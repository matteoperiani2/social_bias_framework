from collections import Counter

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import src.models.bart as bart
import src.models.gpt2 as gpt2

from .text_similarity import TextSimilarity


def generate_model_predictions(model, tokenizer, dataloader, split, gen_cfg, config):
    if config.model["name"] == "gpt2":
        return gpt2.generate_predictions(
            model, tokenizer, dataloader, split, gen_cfg, config
        )
    elif config.model["name"] == "bart":
        return bart.generate_predictions(
            model, tokenizer, dataloader, split, gen_cfg, config
        )
    else:
        raise ValueError("Invalid name. Possible values are [gpt2, bart]")


def evaluate_classification(labels, predictions):
    f1 = evaluate.load("f1")
    f1_scores = []
    for lbls, preds in zip(labels, predictions, strict=True):
        score = f1(lbls[lbls != 2], preds[lbls != 2])
        f1_scores.append(score)

    return f1_scores


def evaluate_generation(labels, predictions, config):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    text_similarity = TextSimilarity(config["embedding_model"])
    # stop_words = stopwords.words('english')
    # model = api.load('word2vec-google-news-300')

    rouge_scores = []
    bleu_scores = []
    similarity_scores = []
    for lbls, pred in zip(labels, predictions, strict=True):
        # if post is offensive or it target a group
        if len(lbls) > 0:
            if pred != "":
                r_scores = rouge.compute(pred, lbls)["rougeL"]
                b_scores = bleu.compute(pred, lbls)["bleu"]
                s_scores = [text_similarity.similarity(pred, lbl) for lbl in lbls]

                # pred_split = pred.lower().split()
                # pred_split = [p for p in pred_split if p not in stop_words]
                # for lbl in lbls:
                #     lbl_split = lbl.lower().split()
                #     lbl_split = [l for l in lbl_split if l not in stop_words]
                #     wmd_score = model.wmdistance(pred_split, lbl_split)

                rouge_scores.append(r_scores)
                bleu_scores.append(b_scores)
                similarity_scores.append(s_scores)
            else:
                rouge_scores.append(0.0)
                bleu_scores.append(0.0)
                similarity_scores.append(0.0)

    return {"rouges": rouge_scores, "similarities": similarity_scores}


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
