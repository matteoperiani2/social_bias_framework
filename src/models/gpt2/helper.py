import os

import datasets
import torch
import transformers

from ..helper import ModelHelper
from .data_collator import GPT2DataCollator
from .model import GPT2SBF, GPT2Loss


class GPT2Helper(ModelHelper):
    def __init__(self, config: dict):
        super().__init__(config)

    def get_data(self, split, aggregated=False):
        if not aggregated:
            path = os.path.join(self.config["data"]["train"], split)
        else:
            path = os.path.join(self.config["data"]["eval"], split)
        self.data = datasets.load_from_disk(path)

        return self.data

    def make_data_collator(self, tokenizer, model):
        self.collator = GPT2DataCollator(tokenizer=tokenizer, model=model)
        return self.collator

    def make_tokenizer(self, verbose=False):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config["model"]["checkpoint_name"],
            padding_side=self.config["model"]["padding_side"],
            use_fast=True,
        )
        tokenizer.add_special_tokens(self.config["model"]["special_tokens"])
        if verbose:
            print("List of all special token and its token_id:")
            print(" -", tokenizer.all_special_tokens)
            print(" -", tokenizer(tokenizer.all_special_tokens)["input_ids"])

        return tokenizer

    def make_model(self, tokenizer):
        model = GPT2SBF.from_pretrained(self.config["model"]["checkpoint_name"])
        # init new embedding
        new_tokens = len(tokenizer) - model.config.vocab_size
        self.model.resize_token_embeddings(len(tokenizer))
        self.__init_new_tokens_embeddings(model, new_tokens)
        # self.__init_lm_bias(model, tokenizer)

        model.transformer.config.pad_token_id = tokenizer.pad_token_id
        model.transformer.config.sep_token_id = tokenizer.sep_token_id
        model.transformer.config.eos_token_id = tokenizer.eos_token_id

        return model

    def __init_new_tokens_embeddings(self, model, new_tokens):
        params = model.state_dict()
        embeddings = params["transformer.wte.weight"]
        pre_expansion_embeddings = embeddings[:-new_tokens, :]
        mu = torch.mean(pre_expansion_embeddings, dim=0)
        n = pre_expansion_embeddings.size()[0]
        sigma = (
            (pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)
        ) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma
        )
        pad_embedding = torch.zeros_like(mu).unsqueeze(0)  # (1, 768)
        other_embes = torch.stack(
            tuple(dist.sample() for _ in range(new_tokens - 1)), dim=0
        )  # (11,768)
        new_embeddings = torch.cat((pad_embedding, other_embes), dim=0)  # (12, 768)
        embeddings[-new_tokens:, :] = new_embeddings
        params["transformer.wte.weight"][-new_tokens:, :] = new_embeddings
        model.load_state_dict(params)

    def __init_lm_bias(self, model, tokenizer):
        params = model.state_dict()
        lm_bias = params["lm_logits_bias"]
        lm_bias[..., tokenizer.pad_token_id] = torch.finfo(torch.float16).min
        for cls_class, freq in self.config["model"]["classification_pos_freq"].items():
            idx = tokenizer.encode(f"<|{cls_class}|>")[0]
            lm_bias[..., idx] = torch.log(torch.tensor(freq))
            lm_bias[..., idx + 1] = torch.log(torch.tensor(1 - freq))
        params["lm_logits_bias"][..., :] = lm_bias
        model.load_state_dict(params)

    def make_loss(self, tokenizer):
        return GPT2Loss(tokenizer, self.config)
