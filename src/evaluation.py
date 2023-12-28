import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge import Rouge

# import nltk
# from nltk.tokenize import word_tokenize
# from gensim.models import KeyedVectors
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
from sklearn.metrics import f1_score

from .utils import binarize


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

    return optimal_threshold


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


def evaluate_classification(labels, predictions, cls_columns):
    f1_scores = {}
    for lbls, preds, cols in zip(labels, predictions, cls_columns, strict=True):
        score = f1_score(lbls[lbls != -1], preds[lbls != -1], average="binary")
        f1_scores[cols] = score

    return f1_scores


__rouge = Rouge(metrics=["rouge-l"], stats="f")


def rouge(prediction, label):
    return __rouge.get_scores(prediction, label)[0]["rouge-l"]["f"]


def bleu(prediction, label):
    return corpus_bleu(
        [[label]],
        [prediction],
        weights=(0.5, 0.5),
        smoothing_function=SmoothingFunction().method1,
    )


def compute_metric(prediction, labels, metric):
    return [metric(prediction, label) for label in labels if label is not None]


# def evaluate_generation(data, config):
#     # stop_words = set(stopwords.words('english'))
#     # word_vectors = KeyedVectors.load_word2vec_format(config.wmd_model, binary=True)
#     similarity = TextSimilarity(config.embedding_model)

#     # all_groups_or_minorities = set()
#     # for cols in ['group', 'stereotype', 'group_preds', 'stereotype_preds']:
#     #     all_groups_or_minorities.update(*[v for v in data[cols] if v is not None and v != ''])

#     # emeddings = dict(zip(all_groups_or_minorities, similarity.generate_embeddings(all_groups_or_minorities)))

#     params = {
#         'rouge': rouge,
#         'emeddings': None
#     }

#     data = data.map(
#         evaluate_generation,
#         load_from_cache_file=False,
#         fn_kwargs=params,
#         batched=False
#     )

#     return data


def compute_generation_metrics(prediction, labels):
    metrics = {"rouge": rouge, "bleu": bleu}
    if labels is not None and prediction != "":
        labels = np.atleast_1d(labels)
        results = {
            metric_name: compute_metric(prediction, labels, metric=metric)
            for metric_name, metric in metrics.items()
        }
    else:
        results = {metric_name: None for metric_name in metrics.keys()}
    return results


def evaluate_generation(example, config):
    group_scores = compute_generation_metrics(example["group_preds"], example["group"])
    stereotype_score = compute_generation_metrics(
        example["stereotype_preds"], example["stereotype"]
    )

    return {"group_scores": group_scores, "stereotype_scores": stereotype_score}


def aggregate_generation_results(group_scores, stereotype_scores):
    group_rouge_scores = [
        max(scores["rouge"]) for scores in group_scores if scores["rouge"] is not None
    ]
    group_bleu_score = [
        max(scores["bleu"]) for scores in group_scores if scores["bleu"] is not None
    ]
    # group_sim_score = [max(scores['similarity']) for scores in group_scores if scores['similarity'] != None]
    # group_wmd_score = [min(scores['wmd']) for scores in group_scores if scores['bleu'] != None]

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
    # stereotype__sim_score = [max(scores['similarity']) for scores in stereotype_scores if scores['similarity'] != None]
    # stereotype_wmd_score = [min(scores['wmd']) for scores in stereotype_scores if scores['bleu'] != None]

    return {
        "group_rouge": np.mean(group_rouge_scores),
        "group_bleu": np.mean(group_bleu_score),
        "stereotype_rouge": np.mean(stereotype_rouge_score),
        "stereotype_bleu": np.mean(stereotype_bleu_score),
    }


def print_generations_results(results):
    for score_name, score in results.items():
        s_class, s_type = score_name.split("_")
        print(f"{s_class.title()} {s_type.title()} score:{score:.3f}")
