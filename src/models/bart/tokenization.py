from typing import Literal, Union

from ..tokenization import TokenizationHelper
from .helper import BartHelper


class BartTokenizationHelper(TokenizationHelper):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        helper = BartHelper(config)
        self.tokenizer = helper.make_tokenizer()

    def tokenize(
        self,
        example: dict,
        task: Union[Literal["train"], Literal["inference"]] = None,
        ignore_labels=False,
    ):
        def tokenize(input_str):
            data = self.tokenizer(
                input_str,
                truncation=True,
                padding=False,
            )
            data["input_ids"] = data["input_ids"][1:-1]
            data["attention_mask"] = data["attention_mask"][1:-1]
            return data

        def map_to_minus_100_if_None(value):
            return value if value is not None else -100

        input_str = example["post"]
        model_inputs = tokenize(input_str)

        if example["group"] is not None and not ignore_labels:
            group = ", ".join(example["group"])
            output_str = group + self.tokenizer.sep_token + example["stereotype"]
            outputs = tokenize(output_str)
            decoder_input_ids = [self.tokenizer.bos_token_id] + outputs["input_ids"]
            decoder_attention_mask = outputs["attention_mask"] + [1]
            labels = outputs["input_ids"] + [self.tokenizer.eos_token_id]
        else:
            decoder_input_ids = [self.tokenizer.bos_token_id]
            decoder_attention_mask = [0]
            labels = [-100]

        if not ignore_labels:
            model_inputs["decoder_input_ids"] = decoder_input_ids
            model_inputs["decoder_attention_mask"] = decoder_attention_mask
            labels = [
                label if label != self.tokenizer.pad_token_id else -100
                for label in labels
            ]
            model_inputs["labels"] = labels

        if not ignore_labels:
            cls_labels = [
                map_to_minus_100_if_None(example[cls_name])
                for cls_name in self.config["classification_columns"]
            ]
            model_inputs["cls_labels"] = cls_labels

        return model_inputs
