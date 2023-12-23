import inspect
import itertools
import os
from typing import Iterable, List

import datasets
import pandas as pd
import regex as re
from datasets import DatasetDict
from torch.utils.data import BatchSampler


def print_if_verbose(*values: str, verbose: bool, **kwargs):
    if verbose:
        print(*values, **kwargs)


def create_dirs_for_file(file_path):
    dir = os.path.dirname(file_path)
    ensure_dir_exists(dir)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def pad_batch(inputs, collator):
    features = [
        dict(zip(inputs.keys(), values, strict=True))
        for values in zip(*inputs.values(), strict=True)
    ]
    features = collator(features)

    return features


def filter_model_inputs(model, inputs):
    forward_signature = set(inspect.signature(model.forward).parameters)
    inputs = {
        argument: value
        for argument, value in inputs.items()
        if argument in forward_signature
    }
    return inputs
