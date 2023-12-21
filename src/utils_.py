import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader


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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

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
    return DataLoader(*args, **kwargs, worker_init_fn=seed_worker, generator=generator)


def remove_post_from_annotation_count(df: pd.DataFrame, ann_count: int):
    post_counts = df["post"].value_counts().reset_index()
    post_to_remove = post_counts[post_counts["count"].apply(lambda c: c == ann_count)][
        "post"
    ].values.tolist()
    pruned_df = df[~df["post"].isin(post_to_remove)].reset_index()
    print("Removed posts:", len(df) - len(pruned_df))
    return pruned_df


def print_classification_results(
    tokenizer, labels, predictions, f1_scores, show_cm=True
):
    annotation_type = [
        "Offensive",
        "Intentional",
        "Sex/Lewd content",
        "Group targetted",
        "Speaker in group",
    ]

    for type, score in zip(annotation_type, f1_scores, strict=True):
        print(f"{type}: {score:.3f}")

    if show_cm:
        plt.rcParams["font.size"] = "12"
        _, axs = plt.subplots(1, 5, figsize=(35, 15))
        for j in range(5):
            lbl = tokenizer.batch_decode(
                np.unique(np.concatenate((predictions[j], labels[j])))
            )
            cm = confusion_matrix(labels[j], predictions[j])
            cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lbl)
            cm_disp.plot(
                ax=axs[j],
                cmap="Blues",
                values_format="0.2f",
                colorbar=False,
                xticks_rotation=90,
            )
            axs[j].set_title(f"{annotation_type[j]}")
            axs[j].set(xlabel=None)
            axs[j].set(ylabel=None)
        plt.show()


def print_generations_results(
    f1_minorities,
    f1_stereotypes,
    min_labels,
    min_pred,
    sterotype_labels,
    sterotype_preds,
    show_dist=True,
):
    print(f"Minority Rouge-L F1 score: {f1_minorities:.3f}")
    print(f"Stereotype Rouge-L F1 score: {f1_stereotypes:.3f}")
    print()
    if show_dist:
        plt.rcParams["font.size"] = "8"
        plot_minority_distribution(
            min_pred, min_labels, sterotype_labels, sterotype_preds
        )
        plt.show()


def plot_minority_distribution(min_pred, min_labels, sterotype_labels, sterotype_preds):
    type = ["Minority", "Stereotype"]
    name = ["predictions", "labels"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, (data, ax) in enumerate(
        zip(
            [min_pred, min_labels, sterotype_labels, sterotype_preds],
            axes.ravel(),
            strict=True,
        )
    ):
        _plot_word_bar(data, ax=ax)
        ax.set_title(f"{type[i//2]} {name[i%2]}")


def _plot_word_bar(data, ax, n_words=10):
    if isinstance(data[0], list):
        topic_words = [y.lower() if y != "" else "''" for x in data for y in x]
    else:
        topic_words = [x.lower() if x != "" else "''" for x in data]
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
    predictions = [
        pred[np.where(pred == tokenizer.sep_token_id)[0][0] + 1 :]
        for pred in predictions
    ]

    for pred in predictions:
        sep_idx = np.where(pred == tokenizer.sep_token_id)[0]
        eos_idx = np.where(pred == tokenizer.eos_token_id)[0][0]

        # --- get classification tokens ---
        # concatenate first 4 tokens with the token generated before the eos
        cls_preds = np.concatenate((pred[:4], [pred[eos_idx - 1]]))
        bin_cls_preds = [
            int(pred == pos_token)
            for pred, pos_token in zip(cls_preds, positive_cls_tokens, strict=True)
        ]

        # if the model predict not offensive or not to a group, ignore the generation
        if pred[0] == 0 or pred[-2] == 0:
            bin_cls_preds[-2] = 0
            bin_cls_preds[-1] = 0
            class_preds.append(bin_cls_preds)
            minority_preds.append([])
            stereotype_preds.append([])
            continue

        class_preds.append(bin_cls_preds)

        # --- get minority and stereotype tokens ---
        if len(sep_idx) > 2:  # if there are at least 3 sep
            # select as minority tokens, those tokens that are between first 2 sep
            minority_preds.append(pred[sep_idx[0] + 1 : sep_idx[1]])
            stereotype_preds.append(pred[sep_idx[1] + 1 : sep_idx[2]])
        elif len(sep_idx) > 1:  # if there are at least 2 sep
            minority_preds.append(pred[sep_idx[0] + 1 : sep_idx[1]])
            stereotype_preds.append(pred[sep_idx[1] + 1 : -2])
        else:  # if there is only 1 sep
            # minority are those tokens betwen sep and second-to-last token
            # for stereotypes no tokens are selected
            minority_preds.append(pred[sep_idx[0] + 1 : eos_idx - 2])
            stereotype_preds.append([])

    minority_preds = tokenizer.batch_decode(minority_preds)
    stereotype_preds = tokenizer.batch_decode(stereotype_preds)

    return class_preds, minority_preds, stereotype_preds


def init_cross_entropy_weights(tokenizer, weight_dict):
    pass
