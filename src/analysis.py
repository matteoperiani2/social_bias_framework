import math
import string
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plot import plot_bar_with_bar_labels


def plot_cls_distribution(
    df: pd.DataFrame, cls_cols, n_cols=2, figsize=(20, 15), type="hist"
):
    n_rows = math.ceil(len(cls_cols) / n_cols)

    plt.figure(figsize=figsize)
    for i, col in enumerate(cls_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        plot_distribution(df, x=col, hue="split", ax=ax, type=type)


def plot_cls_counts(df: pd.DataFrame, cls_cols):
    __plot_cls_counts(df, lambda g: g[cls_cols].notnull())


def plot_cls_null_counts(df: pd.DataFrame, cls_cols):
    __plot_cls_counts(df, lambda g: g[cls_cols].isnull())


def __plot_cls_counts(df: pd.DataFrame, filter_fn):
    counts = (
        df.groupby("split")
        .apply(lambda g: filter_fn(g).sum())
        .unstack()
        .reset_index()
        .rename({"level_0": "cls", 0: "counts"}, axis=1)
    )
    ax = plot_bar_with_bar_labels(counts, x="cls", y="counts", hue="split")
    ax.set_xlabel(None)
    ax.set_ylabel(None)


def plot_distribution(
    df: pd.DataFrame, x: str, hue: str = None, ax=None, type="hist", **kwargs
):
    if type == "hist":
        sns.histplot(df, x=x, hue=hue, ax=ax, **kwargs)

    elif type == "bar":
        if hue is not None:
            df = df.groupby(hue)

        distribution = df[x].value_counts(normalize=True)
        distribution = distribution.apply(lambda x: np.round(x, decimals=3) * 100)
        distribution = distribution.rename("frequency").reset_index()
        plot_bar_with_bar_labels(distribution, x=x, y="frequency", hue=hue, ax=ax)

    else:
        raise ValueError("Type must be hist or bar")


def print_n_annotations(df: pd.DataFrame, txt="Total number of annotations:"):
    print(txt, len(df))
    for split in df["split"].unique():
        n_items = (df["split"] == split).sum()
        print(f"- {split}:", n_items)


def print_txt_with_punctuations(
    txt_list: List[str], punctuations: str = string.punctuation + "-"
):
    txt_with_punct = defaultdict(set)
    for group in txt_list:
        for punctuation in punctuations:
            if punctuation in group:
                txt_with_punct[punctuation].add(group)

    for punctuation, groups in txt_with_punct.items():
        print("Punctuation:", punctuation)
        print("\n".join(groups))
        print("=" * 20 + "\n")


def print_n_items(items: List, n=10, separator: Optional[str] = None, shuffle=False):
    end = "\n"
    if separator is not None:
        end += separator + "\n"
    n = min(n, len(items))
    if shuffle:
        items = np.random.choice(items, size=n, replace=False)
    for item in items[:n]:
        print(item, end=end)


def print_mapping(
    items_from, items_to, n=10, separator: Optional[str] = None, shuffle=False
):
    items = list(zip(items_from, items_to, strict=True))
    n = min(n, len(items))
    if shuffle:
        indices = np.arange(len(items))
        indices = np.random.choice(indices, size=n, replace=False)
        items = np.asarray(items)[indices]
    for before, after in items[:n]:
        print("from:\t", before)
        print("to:\t", after)
        if separator is not None:
            print(separator)
