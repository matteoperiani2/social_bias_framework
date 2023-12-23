import os

import datasets
import transformers

from ..helper import ModelHelper
from .data_collator import BartDataCollator
from .model import BartSBF, loss


class BartHelper(ModelHelper):
    def __init__(self, config: dict):
        super(BartHelper, self).__init__(config)

    def make_model(self):
        self.model = BartSBF.from_pretrained(
            self.config["model"]["checkpoint_name"],
            num_labels=5,
            classifier_dropout=0.1,
        )
        return self.model

    def make_tokenizer(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config["model"]["checkpoint_name"]
        )
        return self.tokenizer

    def get_data(self, split):
        path = os.path.join(self.config["data"], split)
        self.data = datasets.load_from_disk(path)
        return self.data

    def make_collator(self):
        self.collator = BartDataCollator(tokenizer=self.tokenizer, model=self.model)
        return self.collator

    def make_loss(self):
        return loss
