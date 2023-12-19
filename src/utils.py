import os
import random
from collections import Counter, defaultdict
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from collections import Counter, defaultdict
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from nltk.corpus import stopwords

class DummyScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {}


def fix_reproducibility(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(*args, **kwargs):
    generator = torch.Generator()
    return DataLoader(
        *args,
        **kwargs,
        worker_init_fn=seed_worker,
        generator=generator
    )


def data_aggregator(splits: List):
    textFields = ['targetMinority','targetCategory', 'targetStereotype']
    classFields = ['whoTarget', 'intentYN', 'sexYN','offensiveYN']

    for split in splits:
        df = pd.read_pickle(os.path.join(CONFIG.dataset.preproc_dir, f"{split}.pkl"))
        aggDict = {
            c: lambda x: sorted(filter(lambda x: x, list(x))) for c in textFields
        }

        aggDict.update({c: lambda x: np.mean(x) for c in classFields})
        df[textFields] = df[textFields].fillna("")

        agg_df = df.groupby("post", as_index=False).agg(aggDict)
        agg_df["hasBiasedImplication"] = (
            agg_df["targetStereotype"].apply(len) == 0
        ).astype(
            int
        )  # ???

        agg_df["targetMinority"] = agg_df["targetMinority"].apply(
            lambda c: [m.lower().strip() for min in c for m in min.split(",")]
        )
        # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: dict(Counter(c)))
        # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: {k: round(v / total,2) for total in (sum(c.values(), 0.0),) for k, v in c.items()})

        agg_df.to_pickle(os.path.join(CONFIG.dataset.aggregated_dir, f"{split}.pkl"))


def df_aggregator(df: pd.DataFrame):
    textFields = ["targetMinority", "targetCategory", "targetStereotype"]
    classFields = ["whoTarget", "intentYN", "sexYN", "offensiveYN"]

    aggDict = {c: lambda x: sorted(filter(lambda x: x, list(x))) for c in textFields}

    aggDict.update({c: lambda x: np.mean(x) for c in classFields})
    df[textFields] = df[textFields].fillna("")

    agg_df = df.groupby("post", as_index=False).agg(aggDict)
    agg_df["hasBiasedImplication"] = (
        agg_df["targetStereotype"].apply(len) == 0
    ).astype(
        int
    )  # ???

    agg_df["targetMinority"] = agg_df["targetMinority"].apply(
        lambda c: [m.lower().strip() for min in c for m in min.split(",")]
    )
    # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: dict(Counter(c)))
    # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: {k: round(v / total,2) for total in (sum(c.values(), 0.0),) for k, v in c.items()})

    return agg_df


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def aggregate_minority(min_counter: Counter, min_emb: dict, tresh=0.9):
    sim_minorities = defaultdict(list)
    minorities_map = {}
    min_counter2 = Counter()

    while any(min_counter):
        min1, _ = min_counter.most_common(1)[0]
        emb1 = min_emb[min1]
        for min2, emb2 in min_emb.copy().items():
            if cosine_similarity(emb1, emb2) >= tresh:
                if min1 != min2:
                    min_counter.update({min1: min_counter.pop(min2)})
                    min_emb.pop(min2)
                sim_minorities[min1].append(min2)
                minorities_map[min2] = min1

        min_counter2.update({min1: min_counter.pop(min1)})

    return min_counter2, min_emb, sim_minorities, minorities_map


def remove_post_from_annotation_count(df: pd.DataFrame, ann_count: int):
    post_counts = df["post"].value_counts().reset_index()
    post_to_remove = post_counts[post_counts["count"].apply(lambda c: c == ann_count)][
        "post"
    ].values.tolist()
    pruned_df = df[~df["post"].isin(post_to_remove)].reset_index()
    print("Removed posts:", len(df) - len(pruned_df))
    return pruned_df


def get_minorities_embedding(df: pd.DataFrame):
    mask = df["targetMinority"].apply(
        lambda x: len(x) > 0
    )  # consider only those they are offensive (have some minority target)
    off_df = df[mask]
    minorities = off_df["targetMinority"].explode().unique().tolist()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    min_counter = Counter()
    for _idx, row in off_df["targetMinority"].items():
        min_counter.update(row)
    # min_counter = dict(sorted(min_counter.items(), key=lambda x:x[1], reverse=True))

    encoded_input = tokenizer(
        minorities, padding=True, truncation=True, max_length=100, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = embeddings.cpu().numpy()
    min_emb = dict(zip(minorities, embeddings, strict=True))

    return min_counter, min_emb



def print_classification_results(tokenizer, labels, predictions, f1_scores, show_cm = True):
    annotation_type = ["Offensive", "Intentional", "Sex/Lewd content", "Group targetted", "Speaker in group"]

    for type, score in zip(annotation_type, f1_scores):
        print(f"{type}: {score:.3f}")

    if show_cm:
        plt.rcParams['font.size'] = '12'
        _, axs = plt.subplots(1, 5, figsize=(35, 15))
        for j in range(5):
            lbl = tokenizer.batch_decode(np.unique(np.concatenate((predictions[j],labels[j]))))
            cm = confusion_matrix(labels[j], predictions[j])
            cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lbl)
            cm_disp.plot(ax=axs[j], cmap='Blues',values_format='0.2f', colorbar=False, xticks_rotation=90)
            axs[j].set_title(f"{annotation_type[j]}")
            axs[j].set(xlabel=None)
            axs[j].set(ylabel=None)
        plt.show()


def print_generations_results(f1_minorities, f1_stereotypes, min_labels, min_pred, sterotype_labels, sterotype_preds, show_dist=True):
    print(f"Minority Rouge-L F1 score: {f1_minorities:.3f}")
    print(f"Stereotype Rouge-L F1 score: {f1_stereotypes:.3f}")
    print()
    if show_dist:
        plt.rcParams['font.size'] = '8'
        plot_minority_distribution(min_pred, min_labels, sterotype_labels, sterotype_preds)
        plt.show()


def plot_minority_distribution(min_pred, min_labels, sterotype_labels, sterotype_preds):
    type = ['Minority', 'Stereotype']
    name = ['predictions', 'labels']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i,(data, ax) in enumerate(zip([min_pred, min_labels, sterotype_labels, sterotype_preds], axes.ravel())):
        _plot_word_bar(data, ax=ax)
        ax.set_title(f'{type[i//2]} {name[i%2]}')

    
def _plot_word_bar(data, ax, n_words=10):
    if isinstance(data[0], list):
        topic_words = [
            y.lower() if y != '' else '\'\'' for x in data for y in x
        ]
    else:  
        topic_words = [
            x.lower() if x != '' else '\'\'' for x in data
        ]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
    popular_words_nonstop = [
        w for w in popular_words if w not in stopwords.words("english")
    ]
    total = sum([word_count_dict[w] for w in reversed(popular_words_nonstop)])
    ax.barh(
        range(n_words),
        [
            word_count_dict[w] / total
            for w in reversed(popular_words_nonstop[0:n_words])
        ],
    )
    ax.set_yticks(
        [x + 0.5 for x in range(n_words)], reversed(popular_words_nonstop[0:n_words])
    )
    for i in ax.containers:
        ax.bar_label(i, padding=2)



def get_predictions(tokenizer, predictions, positive_cls_tokens):
    class_preds = []
    minority_preds = []
    stereotype_preds = []

    # remove from the generated the input prompt
    predictions = [pred[np.where(pred == tokenizer.sep_token_id)[0][0]+1:] for pred in predictions]

    for pred in predictions:
        sep_idx = np.where(pred == tokenizer.sep_token_id)[0]
        eos_idx = np.where(pred == tokenizer.eos_token_id)[0][0]

        # --- get classification tokens --- 
        # concatenate first 4 tokens with the token generated before the eos
        cls_preds = np.concatenate((pred[:4], [pred[eos_idx-1]]))
        bin_cls_preds = [int(pred==pos_token) for pred,pos_token in zip(cls_preds, positive_cls_tokens)]

        # if the model predict not offensive or not to a group, ignore the generation
        if pred[0] == 0 or pred[-2]==0:
            bin_cls_preds[-2] = 0
            bin_cls_preds[-1] = 0
            class_preds.append(bin_cls_preds)
            minority_preds.append([])
            stereotype_preds.append([])
            continue

        class_preds.append(bin_cls_preds)
        
        # --- get minority and stereotype tokens ---
        if len(sep_idx) > 2: # if there are at least 3 sep
            # select as minority tokens, those tokens that are between first 2 sep
            minority_preds.append(pred[sep_idx[0]+1:sep_idx[1]])
            stereotype_preds.append(pred[sep_idx[1]+1:sep_idx[2]])
        elif len(sep_idx) > 1: # if there are at least 2 sep
            minority_preds.append(pred[sep_idx[0]+1:sep_idx[1]])
            stereotype_preds.append(pred[sep_idx[1]+1:-2])
        else:  # if there is only 1 sep
            # minority are those tokens betwen sep and second-to-last token
            # for stereotypes no tokens are selected
            minority_preds.append(pred[sep_idx[0]+1:eos_idx-2])
            stereotype_preds.append([])

    minority_preds = tokenizer.batch_decode(minority_preds)
    stereotype_preds = tokenizer.batch_decode(stereotype_preds)

    return  class_preds, minority_preds, stereotype_preds


def init_cross_entropy_weights(tokenizer, weight_dict):
    pass 