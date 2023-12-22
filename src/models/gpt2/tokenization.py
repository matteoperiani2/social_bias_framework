from typing import Literal, Union

from ..tokenization import TokenizationHelper
from .helper import GPT2Helper


class GPT2TokenizationHelper(TokenizationHelper):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        helper = GPT2Helper(config)
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
            for cls_idx, cls_name in enumerate(self.config["classification_columns"]):
                value = example[cls_name]
                if value is not None:
                    value = int(value > 0.5)  # binarize
                    cls_token = self.config["model"]["cls_token_map"][cls_idx][value]
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
