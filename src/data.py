import html
import pickle
from collections import defaultdict
from typing import Counter, Dict, Iterable, Literal, Optional, Set, Tuple, Union

import datasets
import emoji
import numpy as np
import pandas as pd
import regex as re

from src.config import Config
from src.helper import GPT2TrainHelper
from src.train_utils import get_model_helper
from src.utils import (
    count_words,
    flatten,
    from_pandas,
    print_if_verbose,
    replace_str,
    to_pandas,
)


class GroupLabelProcessor:
    def __init__(self, groups) -> None:
        self.processed_labels = set(groups)
        self.groups_count = Counter(groups)
        self.group_map = {}
        self.__reverse_group_map = defaultdict(list)
        self.label_to_groups_map = {}

    def label_list(self):
        return [group for group, _ in self.groups_count.most_common()]

    def rename_groups(self, substitution_map, groups: Optional[Iterable[str]] = None):
        group_labels = groups if groups is not None else self.label_list()
        for group in group_labels:
            new_name = replace_str(group, substitution_map)
            self.map_group(group, new_name)

    def aggregate(
        self,
        aggregation_proposals: Dict[str, Set[str]],
        aggregation_blacklist: Optional[Iterable[Tuple[str]]] = None,
        aggregation_preferences: Optional[Dict[str, Iterable[str]]] = None,
    ):
        visited = set()
        group_labels = self.label_list()

        # remove aggregation proposals in blacklist
        if aggregation_blacklist is not None:
            self.__remove_blacklisted_proposals(
                aggregation_proposals, aggregation_blacklist
            )

        if aggregation_preferences is not None:
            # add aggregation preferences to proposals
            for dest_group, groups in aggregation_preferences.items():
                inner_groups = aggregation_proposals.setdefault(dest_group, set())
                inner_groups.update(groups)
            # give priority to preferences labels
            group_labels = list(aggregation_preferences.keys()) + group_labels

        # aggregate labels
        for group in group_labels:
            groups_to_aggregate = self.__get_groups_to_aggregate(
                group, aggregation_proposals, visited
            )
            self.map_groups(groups_to_aggregate, group)

    def remove_double_spaces(self):
        for group in self.label_list():
            fixed_str = white_space_fix(group)
            self.map_group(group, fixed_str)

    def map_groups(self, current_groups, new_group):
        for group in current_groups:
            self.map_group(group, new_group)

    def map_group(self, current_group, new_group):
        if current_group == new_group:
            return

        # add occurrences of current group to new group
        occurrence = self.groups_count.pop(current_group)
        self.groups_count[new_group] += occurrence

        # add current group and groups aggregated into it to the reverse map of new group
        aggregated_into_current_group = self.__reverse_group_map.pop(current_group, [])
        self.__reverse_group_map[new_group].extend(aggregated_into_current_group)
        self.__reverse_group_map[new_group].append(current_group)

        # change group_map s.t. current_group and groups aggregated into it are mapped to new_group
        self.group_map[current_group] = new_group
        for aggregated_group in aggregated_into_current_group:
            self.group_map[aggregated_group] = new_group

        self.processed_labels.add(new_group)

    def map_split_to_groups(
        self, current_group: str, destination_groups: Iterable[str]
    ):
        self.label_to_groups_map[current_group] = destination_groups
        for dest_group in destination_groups:
            self.groups_count[dest_group] += 1

    def map_split_if_all_groups_are_one_word_longs(
        self,
        split_map,
        similars,
        threshold,
        verbose=False,
    ):
        split_map_ = split_map.copy()
        for label, groups in split_map.items():
            if all(count_words(group) == 1 for group in groups):
                dest_groups = [
                    self.get_dest_group(
                        similars[group][0] if similars[group][1] > threshold else group
                    )
                    for group in groups
                ]
                self.map_split_to_groups(label, dest_groups)
                split_map_.pop(label)

                if verbose:
                    print(label, "-->", groups)
                    for i, group in enumerate(groups):
                        sim_group, sim_score = similars[group]
                        dest_group = dest_groups[i]
                        print("  ", group, (sim_group, sim_score), "-->", dest_group)
                    print()
        return split_map_

    def __remove_blacklisted_proposals(self, aggregation_proposals, blacklist):
        for a, b in blacklist:
            if a in aggregation_proposals:
                aggregation_proposals[a].discard(b)
            if b in aggregation_proposals:
                aggregation_proposals[b].discard(a)

    def __get_groups_to_aggregate(
        self, group, aggregation_proposals: Dict[str, Set], visited: Set
    ):
        if group not in aggregation_proposals:
            return ()

        visited.add(group)
        groups_to_aggregate = aggregation_proposals.pop(group)
        groups_to_aggregate.difference_update(visited)
        visited.update(groups_to_aggregate)

        for group_to_aggregate in list(groups_to_aggregate):
            inner_groups = self.__get_groups_to_aggregate(
                group_to_aggregate, aggregation_proposals, visited=visited
            )
            groups_to_aggregate.update(inner_groups)
        return groups_to_aggregate

    def get_dest_group(self, group) -> str:
        return self.group_map.get(group, group)

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    def transform(self, groups: Iterable[str]):
        def transform_group(group: str):
            group = lower(group)
            group = self.get_dest_group(group)
            groups = self.label_to_groups_map.get(group, (group,))
            return groups

        groups = flatten(transform_group(group) for group in groups)
        groups = np.unique(groups).tolist()
        return groups

    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            processor = pickle.load(handle)
        return processor


def aggregation_proposals_by_similarity(group_labels, similarity_matrix, threshold=0.9):
    similarity_matrix = similarity_matrix.copy()
    np.fill_diagonal(similarity_matrix, 0)
    group_labels = np.asarray(group_labels)

    agg_proposals = {}
    for group, similarity in zip(group_labels, similarity_matrix, strict=True):
        indexes = np.argwhere(similarity > threshold).flatten()
        agg_proposals[group] = set(group_labels[indexes].tolist())
    return agg_proposals


def print_aggregation_proposals(aggregates, groups_count):
    for group, similars in aggregates.items():
        if len(similars) > 0:
            occurrence = groups_count[group]
            print(f"{group} ({occurrence}):", similars)


def filter_out_labels_with_multiple_concepts(groups: Iterable[str]):
    return [group for group in groups if re.search(r"\b(?:and|or)\b|\/", group) is None]


def split_on_and_or_slash(labels: Iterable[str]):
    split_map = {}
    for label in labels:
        pieces = re.split(r"\b(?:and|or)\b|\/", label)
        split_map[label] = pieces
    return split_map


def print_split_group_map(split_group_map, split_map, most_similar):
    for label, dest_groups in split_group_map.items():
        groups = split_map[label]
        print(label, "-->", groups)
        for i, group in enumerate(groups):
            sim_group, sim_score = most_similar[group]
            dest_group = dest_groups[i]
            print("  ", group, (sim_group, sim_score), "-->", dest_group)
        print()


def is_annotation_valid(example):
    # discard if offensive is null
    if example["offensive"] is None:
        return False

    # ok if not offensive
    if example["offensive"] == 0.0:
        return True

    # discard if offensive, but vs_group is null
    if example["vs_group"] is None:
        return False

    # ok if not vs_group
    if example["vs_group"] == 0.0:
        return True

    # discard if vs_group, but in_group is null
    if example["in_group"] is None:
        return False

    # discard if vs_group, but group is null  or stereotype is null
    if example["group"] is None or example["stereotype"] is None:
        return False

    return True


def set_features_to_null_wrt_rules(example):
    if example["offensive"] == 0.0:
        example["vs_group"] = None

    if (example["vs_group"] is None) or (example["vs_group"] == 0.0):
        example["in_group"] = None
        example["group"] = None
        example["stereotype"] = None

    return example


def white_space_fix(text: str):
    return " ".join(text.split())


def lower(text):
    return text.lower()


def load_raw_data(config):
    data = datasets.load_dataset(
        "csv",
        data_files={
            "train": config.data.raw.train,
            "val": config.data.raw.val,
            "test": config.data.raw.test,
        },
    )
    return data


def rename_data_columns(raw_data, config):
    data = raw_data.rename_columns(
        {
            "sexYN": "sex",
            "offensiveYN": "offensive",
            "intentYN": "intentional",
            "whoTarget": "vs_group",
            "speakerMinorityYN": "in_group",
            "targetCategory": "group_category",
            "targetMinority": "group",
            "targetStereotype": "stereotype",
            "dataSource": "source",
        }
    )
    cols = data.column_names["train"]
    relevant_cols = (
        ["post"]
        + config.classification_columns
        + ["group_category"]
        + config.generative_columns
    )
    cols = relevant_cols + [col for col in cols if col not in relevant_cols]
    data = data.select_columns(cols)
    return data


def clean_data(config, verbose=True):
    group_lbl_processor = GroupLabelProcessor.load(config.group_lbl_processor)

    def clean_groups(example):
        group = example["group"]
        if group is not None:
            groups = split_group_labels(group)
            groups = group_lbl_processor.transform(groups)
            example["group"] = groups

        return example

    def split_groups(example):
        group = example["group"]
        if group is not None:
            example["group"] = split_group_labels(group)

        return example

    print_if_verbose("Loading raw data ...", verbose=verbose)
    data = load_raw_data(config)
    data = rename_data_columns(data, config)

    print_if_verbose("Removing invalid annotations ...", verbose=verbose)
    data = data.filter(is_annotation_valid, num_proc=4)

    print_if_verbose("Setting features to None ...", verbose=verbose)
    data = data.map(set_features_to_null_wrt_rules, num_proc=4)

    print_if_verbose("Cleaning groups ...", verbose=verbose)
    test_data = data.pop("test").map(split_groups, num_proc=4)
    data = data.map(clean_groups, num_proc=4)
    data["test"] = test_data

    print_if_verbose("Saving data to", config.data.clean, "...", verbose=verbose)
    data.save_to_disk(config.data.clean)

    print_if_verbose("Complete!", verbose=verbose)


def split_group_labels(group: str):
    return re.split(r"[,;]", group)


def constains_html_escaped_str(string):
    if "&" not in string:
        return False

    return bool(re.search(html._charref.pattern, string))


def unescape_html_str(string: str):
    return html.unescape(string)


def contains_links(text: str):
    """Check if there are links in the text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return bool(url_pattern.search(text))


def remove_links(text: str):
    """Remove links from the text."""
    url_pattern = re.compile(r"(?:https?://)[-a-zA-Z0-9@:%._\\+~#?&//=]+")
    return re.sub(url_pattern, "", text)


__reddit_RT_at_author_regex = re.compile(r"^\s*RT @\S+:\s*")


def starts_with_RT_at_author(text: str):
    return bool(re.search(__reddit_RT_at_author_regex, text))


def remove_RT_at_author(text: str):
    return re.sub(__reddit_RT_at_author_regex, "", text)


__char_map = {
    r"\p{Cn}": "",  # any invalid unicode char
    r"[’‘]": "'",
    "\ufeff": "",
    r"[”“]": '"',
    "[—―–─]": "-",
    "…": "...",
    "\\'": "'",
}


def remove_leading_non_alphanumeric_chars(text: str) -> str:
    """
    Removes leading non alpha-numeric characters from a given string.
    """
    start_index = 0
    while start_index < len(text) and not text[start_index].isalnum():
        start_index += 1

    end_index = len(text) - 1
    while end_index > start_index and not text[end_index].isalnum():
        end_index -= 1

    return text[start_index : end_index + 1]


def remove_overlapping_posts(master_data, slave_data):
    """Remove data in slave_data if post is present in master_data"""
    posts = set(master_data["post"])
    return slave_data.filter(lambda example: example["post"] not in posts, num_proc=4)


def preprocess_post(post: str) -> str:
    pipeline = (
        unescape_html_str,
        unescape_html_str,
        remove_links,
        emoji.demojize,
        remove_RT_at_author,
        lambda txt: replace_str(txt, __char_map),
        white_space_fix,
    )
    for fn in pipeline:
        post = fn(post)
    return post


def preprocess_stereotype(stereotype: str) -> str:
    pipeline = (
        remove_leading_non_alphanumeric_chars,
        white_space_fix,
    )
    for fn in pipeline:
        stereotype = fn(stereotype)
    return stereotype


def preprocess_data(config, verbose=True):
    def _preprocess_post(example):
        example["post"] = preprocess_post(example["post"])
        return example

    def _preprocess_stereotype(example):
        stereotype = example["stereotype"]
        if stereotype is not None:
            example["stereotype"] = preprocess_stereotype(stereotype)
        return example

    print_if_verbose("Loading clean data ...", verbose=verbose)
    data = datasets.load_from_disk(config.data.clean)

    print_if_verbose("Preprocessing posts ...", verbose=verbose)
    data = data.map(_preprocess_post, num_proc=4)

    print_if_verbose("Removing overlapping posts ...", verbose=verbose)
    data["val"] = remove_overlapping_posts(data["train"], data["val"])
    data["test"] = remove_overlapping_posts(data["train"], data["test"])
    data["val"] = remove_overlapping_posts(data["test"], data["val"])

    print_if_verbose("Preprocessing stereotypes ...", verbose=verbose)
    data = data.map(_preprocess_stereotype, num_proc=4)

    print_if_verbose("Saving data to", config.data.processed, "...", verbose=verbose)
    data.save_to_disk(config.data.processed)
    print_if_verbose("Complete!", verbose=verbose)


def aggregate_data(config, verbose=True):
    def unique(x: pd.Series):
        return x.unique().tolist() if x.notnull().any() else None

    print_if_verbose("Loading processed data ...", verbose=verbose)
    data = datasets.load_from_disk(config.data.processed)
    cols = (
        ["post"]
        + config.classification_columns
        + config.generative_columns
        + ["source"]
    )
    data = data.select_columns(cols)
    df = to_pandas(data)

    print_if_verbose("Aggregating data ...", verbose=verbose)
    agg_functions = {col: "mean" for col in config.classification_columns}
    agg_functions.update(
        {col: unique for col in config.generative_columns + ("source",)}
    )
    agg_functions["split"] = lambda x: x.iloc[0]
    df["group"] = df["group"].str.join(", ")
    df = df.groupby("post").aggregate(agg_functions).reset_index()

    print_if_verbose("Saving data to", config.data.aggregated, "...", verbose=verbose)
    dataset = from_pandas(df)
    dataset.save_to_disk(config.data.aggregated)
    print_if_verbose("Complete!", verbose=verbose)


class GPT2DataHelper:
    def __init__(self, config) -> None:
        self.config = config
        helper = GPT2TrainHelper(Config.to_dict(config))
        self.tokenizer = helper.make_tokenizer()

    def tokenize(
        self,
        example: dict,
        task: Union[Literal["train"], Literal["inference"]],
        ignore_labels=False,
    ):
        input_str = example["post"] + self.tokenizer.sep_token
        label_str = None
        if not ignore_labels:
            cls_features = []
            for cls_idx, cls_name in enumerate(self.config.classification_columns):
                value = example[cls_name]
                if value is not None:
                    value = int(value > 0.5)  # binarize
                    cls_token = self.config.model.cls_token_map[cls_idx][value]
                else:
                    cls_token = (
                        self.tokenizer.pad_token
                    )  # trick: pad token will be ignored by loss
                cls_features.append(cls_token)

            generative_features = ""
            if example["group"] is not None:
                assert example["stereotype"] is not None

                generative_features = (
                    self.tokenizer.sep_token
                    + ", ".join(example["group"])
                    + self.tokenizer.sep_token
                    + example["stereotype"]
                    + self.tokenizer.sep_token
                )

            cls_str = "".join(cls_features[:-1])
            in_group_token = cls_features[-1] if example["in_group"] is not None else ""

            label_str = cls_str + generative_features + in_group_token

        input_ids = self.tokenizer(
            text=input_str,
            text_pair=label_str if task == "train" else None,
            padding=False,
            truncation="only_first",
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=False,
        )["input_ids"]
        if task == "train":
            # shift tokens to the left
            input_ids = input_ids[:-1]

        attention_mask = [
            0 if token == self.tokenizer.pad_token_id else 1 for token in input_ids
        ]

        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not ignore_labels:
            labels = self.tokenizer(label_str, padding=False, truncation=False)[
                "input_ids"
            ]
            labels = [
                -100 if token == self.tokenizer.pad_token_id else token
                for token in labels
            ]
            if len(labels) > 4:
                labels[4] = -100  # ignore first sep token
            outputs["labels"] = labels

        return outputs


def tokenize_data(
    config,
    task: Union[Literal["train"], Literal["inference"]],
    use_aggregate=False,
    verbose=True,
):
    if task == "train" and use_aggregate:
        raise NotImplementedError("Train with aggregated dataset is not supported.")

    data_helper = GPT2DataHelper(config) if config.model.name == "gpt2" else None

    print_if_verbose("Loading processed data ...", verbose=verbose)
    path = config.data.aggregated if use_aggregate else config.data.processed
    data = datasets.load_from_disk(path)

    print_if_verbose("Tokenizing data ...", verbose=verbose)

    remove_columns = []
    if task == "train":
        data.pop("test")
        remove_columns = data["train"].column_names
    data = data.map(
        data_helper.tokenize,
        fn_kwargs={"task": task, "ignore_labels": use_aggregate},
        remove_columns=remove_columns,
        num_proc=4,
    )

    path = (
        config.data.train
        if task == "train"
        else config.data.eval
        if not use_aggregate
        else config.data.aggregated_eval
    )
    print_if_verbose("Saving data to", path, "...", verbose=verbose)
    data.save_to_disk(path)

    print_if_verbose("Complete!", verbose=verbose)


def print_tokenized_dataset(data, config, n=10):
    helper = get_model_helper(Config.to_dict(config))
    tokenizer = helper.make_tokenizer()
    examples = data.shuffle().select(range(n))
    for example in examples:
        for key, value in example.items():
            print(f"{key}:", value)
            if key != "attention_mask":
                value = [
                    token if token != -100 else tokenizer.pad_token_id
                    for token in value
                ]
                print(f"{key}:", tokenizer.decode(value, skip_special_tokens=False))
        print("-" * 50)
