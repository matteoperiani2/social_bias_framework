import codecs
import html
import pickle
from collections import defaultdict
from typing import Counter, Dict, Iterable, Optional, Set, Tuple

import datasets
import emoji
import numpy as np
import regex as re

from .utils import count_words, flatten, replace_str

classification_cols = ["offensive", "intentional", "sex", "vs_group", "in_group"]
generative_cols = ["group", "stereotype"]


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

    # of if not vs_group
    if example["vs_group"] == 0.0:
        return True

    # discard if vs_group, but in_group is null
    if example["in_group"] is None:
        return False

    # discard if vs_group, but group is null
    if example["group"] is None:
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


def rename_data_columns(raw_data):
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
        ["post"] + classification_cols + ["group_category"] + generative_cols
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

    if verbose:
        print("Loading raw data...")
    data = load_raw_data(config)
    data = rename_data_columns(data)

    if verbose:
        print("Removing invalid annotation...")
    data = data.filter(is_annotation_valid, num_proc=4)

    if verbose:
        print("Setting features to None...")
    data = data.map(set_features_to_null_wrt_rules, num_proc=4)

    if verbose:
        print("Cleaning groups...")
    data = data.map(clean_groups, num_proc=4)

    if verbose:
        print("Saving data to", config.data.clean)
    data.save_to_disk(config.data.clean)
    if verbose:
        print("Complete!")


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


def decode_unicode_escape_sequences(text):
    """Convert a string containing Unicode escape sequences to Unicode."""
    return codecs.decode(text, "unicode_escape")


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


def preprocess_data(config, verbose=True):
    def _preprocess_post(example):
        example["post"] = preprocess_post(example["post"])
        return example

    if verbose:
        print("Loading clean data...")
    data = datasets.load_from_disk(config.data.clean)

    if verbose:
        print("Preprocessing posts...")
    data = data.map(_preprocess_post, num_proc=4)

    if verbose:
        print("Saving data to", config.data.processed)
    data.save_to_disk(config.data.processed)
    if verbose:
        print("Complete!")
