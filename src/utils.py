from typing import Iterable, List

import pandas as pd
import regex as re
from datasets import DatasetDict
from torch.utils.data import BatchSampler


def to_pandas(dataset: DatasetDict, key_name="split"):
    dataset_ = []
    for split, ds in dataset.items():
        split_df = ds.to_pandas()
        split_df[key_name] = split
        dataset_.append(split_df)
    dataset_ = pd.concat(dataset_)
    dataset_.reset_index(drop=True, inplace=True)

    return dataset_


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
