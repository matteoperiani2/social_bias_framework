from typing import Literal

from ..tokenization import TokenizationHelper
from .helper import BartHelper


class BartTokenizationHelper(TokenizationHelper):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        helper = BartHelper(config)
        self.tokenizer = helper.make_tokenizer()

    def tokenize(
        self, example: dict, task: Literal["train", "inference"], ignore_labels=False
    ):
        raise NotImplementedError()
