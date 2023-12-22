import itertools
from typing import Iterable, List

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
from datasets import DatasetDict
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import BatchSampler


def to_pandas(dataset: DatasetDict, key_name="split") -> pd.DataFrame:
    dataset_ = []
    for split, ds in dataset.items():
        split_df = ds.to_pandas()
        split_df[key_name] = split
        dataset_.append(split_df)
    dataset_ = pd.concat(dataset_)
    dataset_.reset_index(drop=True, inplace=True)

    return dataset_


def from_pandas(df: pd.DataFrame, key_name="split") -> DatasetDict:
    data = datasets.DatasetDict()
    for key in df[key_name].unique():
        data[key] = datasets.Dataset.from_pandas(
            df[df[key_name] == key].reset_index(drop=True)
        )
    data = data.remove_columns(key_name)
    return data


def flatten(list_of_lists: List[List]):
    return [item for list in list_of_lists for item in list]


def replace_str(txt, substitution_map):
    for regex, substitution in substitution_map.items():
        txt = re.sub(regex, substitution, txt)
    return txt


def count_words(sentence: str):
    words = re.findall(r"\b\w+\b", sentence)
    return len(words)


def batch(data: Iterable, batch_size: int) -> Iterable[Iterable]:
    return BatchSampler(data, batch_size=batch_size, drop_last=False)


def batched_function(fn, scalar_output=True):
    def execute_on_batch(batch):
        examples = [
            fn(dict(zip(batch.keys(), values, strict=True)))
            for values in zip(*batch.values(), strict=True)
        ]

        if scalar_output:
            return {
                key: [example[key] for example in examples]
                for key in examples[0].keys()
            }

        return {
            key: list(itertools.chain(*(example[key] for example in examples)))
            for key in examples[0].keys()
        }

    return execute_on_batch


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


def process_gpt2_predictions(tokenizer, predictions, positive_cls_tokens):
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


def print_if_verbose(*values: str, verbose: bool, **kwargs):
    if verbose:
        print(*values, **kwargs)


def init_cross_entropy_weights(tokenizer, weight_dict):
    pass
