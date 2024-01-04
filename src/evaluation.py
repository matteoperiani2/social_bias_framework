from collections import defaultdict
import numpy as np
import pandas as pd
import string
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from gensim.models.keyedvectors import KeyedVectors
import spacy

from rouge import Rouge
from typing import Dict, List, Union
import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import os
from sklearn.metrics import f1_score

from .plot import plot_bar_with_bar_labels
from .utils import binarize, create_dirs_for_file
from .config import Config


wmd_useless_tokens = set(string.punctuation)


def tune_threshold(y_true, y_logits, threshold_range=(0.1, 0.9), threshold_step=0.01):
    """
    Tune the threshold to maximize F1 score.

    Parameters:
    - y_true (array-like): True binary labels.
    - y_logits (array-like): Predicted probabilities or scores.
    - threshold_range (tuple): Range of thresholds to search within.
    - threshold_step (float): Step size for searching thresholds.

    Returns:
    - float: Optimal threshold that maximizes F1 score.
    """
    thresholds = np.arange(
        threshold_range[0], threshold_range[1] + threshold_step, threshold_step
    )
    logit_thresholds = np.log(thresholds) - np.log(1 - thresholds)
    best_f1 = 0
    optimal_threshold = threshold_range[0]

    for i, threshold in enumerate(logit_thresholds):
        y_pred = (y_logits >= threshold).astype(int)
        current_f1 = f1_score(
            y_true[y_true != -1], y_pred[y_true != -1], average="binary"
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            optimal_threshold = thresholds[i]

    return round(optimal_threshold / threshold_step) * threshold_step


def tune_cls_thresholds(data, config, threshold_range=(0.1, 0.9), threshold_step=0.01):
    """
    Tune the thresholds to maximize F1 score of each cls label.
    """

    def apply_threshold(cls_column):
        feature = np.asarray(data["cls_logits"])[
            ..., config.classification_columns.index(cls_column)
        ]
        threshold = optimal_thresholds[cls_column]
        threshold = np.log(threshold) - np.log(1 - threshold)
        return (feature >= threshold).astype(int)

    optimal_thresholds = {}
    for i, cls_col in enumerate(config.classification_columns):
        true_labels = np.asarray([binarize(v) for v in data[cls_col]])
        true_labels = np.where(pd.notnull(true_labels), true_labels, -1).astype(int)
        predicted_scores = np.asarray(data["cls_logits"])[..., i]
        if cls_col in ["vs_group", "in_group"]:
            offensive = apply_threshold("offensive")
            predicted_scores[offensive == 0] = -np.inf
        if cls_col == "in_group":
            vs_group = apply_threshold("vs_group")
            predicted_scores[vs_group == 0] = -np.inf

        optimal_threshold = tune_threshold(
            true_labels,
            predicted_scores,
            threshold_range=threshold_range,
            threshold_step=threshold_step,
        )
        optimal_thresholds[cls_col] = optimal_threshold
    return optimal_thresholds


class Evaluator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.__rouge_metric = Rouge(metrics=["rouge-l"], stats="f")
        self.__wmd_model = self.__load_wmd_model(config)
        self.metrics = {
            "rouge": self.__rouge,
            "bleu": self.__bleu,
            "wmd": self.__wmd_distance,
        }

    def evaluate_classification(
        self, labels, predictions, cls_columns
    ) -> Dict[str, float]:
        f1_scores = {}
        for lbls, preds, cols in zip(labels, predictions, cls_columns, strict=True):
            score = f1_score(lbls[lbls != -1], preds[lbls != -1], average="binary")
            f1_scores[cols] = score

        return f1_scores

    def evaluate_generation(self, example):
        group_scores = self.__compute_generation_metrics(
            example["group_preds"], example["group"]
        )
        stereotype_score = self.__compute_generation_metrics(
            example["stereotype_preds"], example["stereotype"]
        )

        return {"group_scores": group_scores, "stereotype_scores": stereotype_score}

    def __compute_generation_metrics(self, prediction, labels):
        if labels is not None and prediction != "":
            labels = np.atleast_1d(labels)
            results = {
                metric_name: self.__compute_metric(prediction, labels, metric=metric)
                for metric_name, metric in self.metrics.items()
            }
        else:
            results = {metric_name: None for metric_name in self.metrics.keys()}
        return results

    def __compute_metric(self, prediction, labels, metric):
        return [metric(prediction, label) for label in labels if label is not None]

    def aggregate_generation_results(self, group_scores, stereotype_scores):
        group_rouge_scores = [
            max(scores["rouge"])
            for scores in group_scores
            if scores["rouge"] is not None
        ]
        group_bleu_score = [
            max(scores["bleu"]) for scores in group_scores if scores["bleu"] is not None
        ]
        group_wmd_score = [
            min(scores["wmd"]) for scores in group_scores if scores["wmd"] is not None
        ]
        group_wmd_score = [score for score in group_wmd_score if score < float("inf")]

        stereotype_rouge_score = [
            max(scores["rouge"])
            for scores in stereotype_scores
            if scores["rouge"] is not None
        ]
        stereotype_bleu_score = [
            max(scores["bleu"])
            for scores in stereotype_scores
            if scores["bleu"] is not None
        ]
        stereotype_wmd_score = [
            min(scores["wmd"])
            for scores in stereotype_scores
            if scores["wmd"] is not None
        ]
        stereotype_wmd_score = [
            score for score in stereotype_wmd_score if score < float("inf")
        ]

        return {
            "group_rouge": np.mean(group_rouge_scores),
            "group_bleu": np.mean(group_bleu_score),
            "group_wmd": np.mean(group_wmd_score),
            "stereotype_rouge": np.mean(stereotype_rouge_score),
            "stereotype_bleu": np.mean(stereotype_bleu_score),
            "stereotype_wmd": np.mean(stereotype_wmd_score),
        }

    def __rouge(self, prediction, label):
        return self.__rouge_metric.get_scores(prediction, label)[0]["rouge-l"]["f"]

    def __bleu(self, prediction, label):
        return corpus_bleu(
            [[label]],
            [prediction],
            weights=(0.5, 0.5),
            smoothing_function=SmoothingFunction().method1,
        )

    def evaluate_per_seed_classification(
        self,
        predictions: Dict[
            str, Union[Dict[str, datasets.Dataset], datasets.DatasetDict]
        ],
    ) -> Dict[str, Dict[str, datasets.Dataset]]:
        results = defaultdict(dict)
        for seed, pred_datasets in predictions.items():
            for split, preds in pred_datasets.items():
                res = evaluate_classification(self, preds, self.config)
                results[seed][split] = res

        return dict(results)

    def evaluate_per_seed_generation(
        self,
        predictions: Dict[
            str, Union[Dict[str, datasets.Dataset], datasets.DatasetDict]
        ],
    ):
        results = defaultdict(dict)
        for seed, pred_datasets in predictions.items():
            for split, preds in pred_datasets.items():
                preds = preds.map(self.evaluate_generation)
                path = os.path.join(self.config.data.prediction, seed, split)
                preds.save_to_disk(path)
                res = self.aggregate_generation_results(
                    preds["group_scores"], preds["stereotype_scores"]
                )

                results[seed][split] = res

        return dict(results)

    def aggregate_per_seed_generation_results(
        self,
        predictions: Dict[
            str, Union[Dict[str, datasets.Dataset], datasets.DatasetDict]
        ],
    ):
        results = defaultdict(dict)
        for seed, pred_datasets in predictions.items():
            for split, preds in pred_datasets.items():
                res = self.aggregate_generation_results(
                    preds["group_scores"], preds["stereotype_scores"]
                )

                results[seed][split] = res

        return dict(results)

    def __load_wmd_model(self, config: Config) -> KeyedVectors:
        """
        Loads a pre-trained word embedding model via gensim library.

        :return
            - pre-trained word embedding model (gensim KeyedVectors object)
        """

        spacy_model = os.path.split(config.wmd_model)[-1]
        if not os.path.exists(config.wmd_model):
            spacy.cli.download(spacy_model)
            nlp = spacy.load(spacy_model)
            wordList = []
            vectorList = []
            for key, vector in nlp.vocab.vectors.items():
                wordList.append(nlp.vocab.strings[key])
                vectorList.append(vector)
            embedding = KeyedVectors(nlp.vocab.vectors_length)
            embedding.add_vectors(wordList, vectorList)

            create_dirs_for_file(config.wmd_model)
            embedding.save(config.wmd_model)

        self.nlp = spacy.load(spacy_model)
        return KeyedVectors.load(config.wmd_model)

    def __wmd_tokenize(self, text: str) -> List[str]:
        # Tokenize and remove punctuation and stopwords
        tokenized_text = self.nlp(text)
        tokens = [
            token.text
            for token in tokenized_text
            if token.text not in wmd_useless_tokens
        ]
        return tokens

    def __wmd_distance(self, prediction, label):
        prediction = self.__wmd_tokenize(str(prediction))
        label = self.__wmd_tokenize(str(label))
        return self.__wmd_model.wmdistance(prediction, label)


def __prepare_cls_labels_for_evaluation(predictions: Dict[str, list], config):
    cls_labels = [
        [binarize(v) for v in predictions[cls_col]]
        for cls_col in config.classification_columns
    ]
    cls_labels = np.where(pd.notnull(cls_labels), cls_labels, -1).astype(int)
    cls_preds = np.asarray(predictions["cls_preds"]).T
    return cls_labels, cls_preds


def evaluate_classification(
    evaluator: Evaluator, predictions: Dict[str, list], config: Config
):
    cls_labels, cls_preds = __prepare_cls_labels_for_evaluation(predictions, config)

    results = evaluator.evaluate_classification(
        cls_labels, cls_preds, config.classification_columns
    )
    return results


def load_per_seed_predictions(
    seeds: List[int], config
) -> Dict[str, Dict[str, datasets.Dataset]]:
    predictions = {}
    for seed in seeds:
        path = os.path.join(config.data.prediction, str(seed))
        data = datasets.load_from_disk(path)
        predictions[str(seed)] = data
    return predictions


def show_classification_results(
    predictions: Dict[str, list], results, config, show_cm=True
):
    annotation_type = [
        "Offensive",
        "Intentional",
        "Sex/Lewd content",
        "Group targetted",
        "Speaker in group",
    ]

    for type, score in zip(annotation_type, results.values(), strict=True):
        print(f"{type}: {score:.3f}")

    if show_cm:
        cls_labels, cls_preds = __prepare_cls_labels_for_evaluation(predictions, config)
        plot_classification_cm(cls_labels, cls_preds, annotation_type)


def print_generation_results(results):
    for score_name, score in results.items():
        s_class, s_type = score_name.split("_")
        print(f"{s_class.title()} {s_type.title()} score:{score:.3f}")


def plot_classification_results(results):
    plot_evaluation_results(results, name="Classification")


def plot_generative_results(results):
    plot_evaluation_results(results, name="Generative")


def plot_evaluation_results(results, name):
    split_dfs = defaultdict(list)
    for seed, seed_res in results.items():
        for split, split_res in seed_res.items():
            for score_name, score_val in split_res.items():
                split_dfs[split].append(
                    {
                        "name": score_name,
                        "score": round(score_val, 4),
                        "seed": seed,
                    }
                )
    split_dfs = {split: pd.DataFrame(values) for split, values in split_dfs.items()}

    _, axs = plt.subplots(1, 2, figsize=(15, 4))
    for i, (split, df) in enumerate(split_dfs.items()):
        plot_bar_with_bar_labels(
            df,
            x="name",
            y="score",
            hue="seed",
            ax=axs[i],
            bar_label_rotation=90,
            bar_label_type="center",
        )
        axs[i].set_title(f"{name} score on {split} set")
        axs[i].set_xlabel(None)
        axs[i].set_ylabel("Score")
        axs[i].tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    plt.plot()


def plot_classification_cm(labels, predictions, annotation_type):
    _, axs = plt.subplots(1, 5, figsize=(35, 15))
    for j in range(5):
        lbls = labels[j]
        preds = predictions[j]
        cm = confusion_matrix(lbls[lbls != -1], preds[lbls != -1], normalize="all")
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
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


def plot_models_metrics_bar(results: dict, split: str, multiple_seeds=False):
    dfs = []
    for model, res in results.items():
        x = __prepare_data(res, splits=split, multiple_seeds=multiple_seeds)
        x["model"] = model
        dfs.append(x)
    data = pd.concat(dfs)

    __plot_metrics_bar(data, hue="model", multiple_seeds=multiple_seeds)


def plot_metrics_bar(results, splits=None, multiple_seeds=False):
    data = __prepare_data(results, splits=splits, multiple_seeds=multiple_seeds)
    __plot_metrics_bar(data, hue="split", multiple_seeds=multiple_seeds)


def __prepare_data(results, splits, multiple_seeds):
    if splits is None:
        x = next(iter(results.values())) if multiple_seeds else results
        splits = x.keys()

    def prepare_data(results, splits):
        x = (
            pd.DataFrame(results)
            .map(lambda t: round(t, 4))
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        return x.melt(id_vars=["metric"], value_vars=splits, var_name="split")

    if multiple_seeds:
        dfs = []
        for seed, res in results.items():
            x = prepare_data(res, splits)
            x["seed"] = seed
            dfs.append(x)
        x = pd.concat(dfs)
    else:
        x = prepare_data(results, splits)
    return x


def __plot_metrics_bar(data, hue: str, multiple_seeds=False):
    errorbar = ("sd", 2) if multiple_seeds else None
    bar_label_type = "center" if multiple_seeds else "edge"

    ax = plot_bar_with_bar_labels(
        data,
        x="metric",
        y="value",
        hue=hue,
        errorbar=errorbar,
        bar_label_rotation=90,
        bar_label_type=bar_label_type,
    )
    ax.tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    plt.show()
