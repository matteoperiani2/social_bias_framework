import torch
import transformers
from datasets.arrow_dataset import Dataset

from ..helper import ModelHelper
from .data_collator import BartDataCollator
from .model import BartSBF, loss


class BartHelper(ModelHelper):
    def __init__(self, config: dict):
        super().__init__(config)

    def make_tokenizer(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config["model"]["checkpoint_name"],
            padding_side=self.config["model"]["padding_side"],
        )
        tokenizer.sep_token = tokenizer.bos_token
        tokenizer.sep_token_id = tokenizer.bos_token_id
        return tokenizer

    def make_model(self, tokenizer):
        model = BartSBF.from_pretrained(
            self.config["model"]["checkpoint_name"],
            num_labels=5,
            classifier_dropout=0.1,
        )
        self.__init_mlp_bias(model)

        model.config.sep_token_id = tokenizer.sep_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        return model

    def __init_mlp_bias(self, model):
        # we initialize bias to -log((1-freq)/freq)
        f = torch.FloatTensor(self.model_config["classification_pos_freq"])
        bias_values = -torch.log(1 - f) + torch.log(f)
        params = model.state_dict()
        params["classification_head.out_proj.bias"] = bias_values
        model.load_state_dict(params)

    def get_data(self, split) -> Dataset:
        cols = [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
            "cls_labels",
        ]
        return super().get_data(split).select_columns(cols)

    def make_data_collator(self, tokenizer, model):
        collator = BartDataCollator(tokenizer=tokenizer, model=model)
        return collator

    def make_loss(self):
        return loss(self.config)
