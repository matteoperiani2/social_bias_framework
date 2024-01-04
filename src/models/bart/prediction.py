from typing import Union

import os
import datasets
import numpy as np
import torch

from ...config import Config
from ...utils import pad_batch, print_if_verbose
from .helper import BartHelper


class BartInference:
    def __init__(self, checkpoint_name, config: Config) -> None:
        self.config = config
        helper = BartHelper(Config.to_dict(config))
        self.tokenizer = helper.make_tokenizer()

        self.model = helper.make_model(self.tokenizer)
        checkpoint_path = os.path.join(
            config.model.checkpoint_dir, checkpoint_name + ".pt"
        )
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        self.data_collator = helper.make_data_collator(self.tokenizer, self.model)

    def generate_predictions(
        self,
        dataset: Union[datasets.Dataset, datasets.DatasetDict],
        cls_threshold=0.5,
        verbose=True,
    ):
        self.prepare_model()

        print_if_verbose("Predicting classification features ...", verbose=verbose)
        dataset = dataset.map(
            self.predict_cls_features,
            batched=True,
            batch_size=self.config.model.generate_batch_size,
        )

        print_if_verbose("Binarizing classification features ...", verbose=verbose)
        dataset = dataset.map(
            self.binarize_cls_preds,
            fn_kwargs={"threshold": cls_threshold},
            batched=True,
            num_proc=4,
        )

        print_if_verbose("Generating groups and stereotypes ...", verbose=verbose)
        if isinstance(dataset, datasets.DatasetDict):
            first_split = next(iter(dataset.values()))
            features = first_split.features
        else:
            features = dataset.features
        features["pred_tokens"] = datasets.features.Sequence(
            feature=datasets.features.Value(dtype="int64")
        )
        dataset = dataset.map(
            self.predict_gen_features,
            batched=True,
            batch_size=self.config.model.generate_batch_size,
            features=features,
        )

        print_if_verbose("Decoding generated tokens ...", verbose=verbose)
        dataset = dataset.map(self.decode_pred_tokens, batched=True, num_proc=4)

        return dataset

    def predict(self, batch, cls_threshold=0.5):
        self.prepare_model()
        batch.update(self.predict_cls_features(batch))
        batch.update(self.binarize_cls_preds(batch, threshold=cls_threshold))
        batch.update(self.predict_gen_features(batch))
        batch.update(self.decode_pred_tokens(batch))

        return batch

    def prepare_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()

    def predict_cls_features(self, batch):
        batch = self.__prepare_inputs_for_generation(batch)

        with torch.no_grad():
            outputs = self.model(**batch, return_dict=True)
        del batch

        return {"cls_logits": outputs.cls_logits.tolist()}

    def binarize_cls_preds(self, batch, threshold=0.5):
        cls_logits = np.asarray(batch["cls_logits"])
        # map in the logit space
        threshold = np.log(threshold) - np.log(1 - threshold)
        bin_cls_preds = (cls_logits >= threshold).astype(int)

        # if not offensive or not vs_group, we set vs_group and in_group to 0.
        not_off_or_not_vs_group = (bin_cls_preds[..., 0] == 0) | (
            bin_cls_preds[..., 3] == 0
        )
        bin_cls_preds[not_off_or_not_vs_group, -2:] = 0

        return {"cls_preds": bin_cls_preds.tolist()}

    def predict_gen_features(self, batch):
        # create a map between batch index and generate batch index,
        # where the latter is obtained by removing samples from the input batch
        # that are not offensive or are not vs_group
        in_gen_batch_map = self.__create_input_gen_batch_map(batch)

        # if generative batch is non empty, generate on generative samples
        if len(in_gen_batch_map) > 0:
            gen_batch = {
                k: [v for i, v in enumerate(values) if i in in_gen_batch_map]
                for k, values in batch.items()
            }
            gen_batch = self.__prepare_inputs_for_generation(gen_batch)

            pred_tokens = (
                self.model.generate(
                    **gen_batch,
                    **self.config.model.generation_params,
                    return_dict=False
                )
                .detach()
                .cpu()
                .tolist()
            )
            del gen_batch

        batch_size = len(batch["input_ids"])
        pred_tokens = [
            pred_tokens[in_gen_batch_map[i]] if i in in_gen_batch_map else []
            for i in range(batch_size)
        ]
        return {"pred_tokens": pred_tokens}

    def __create_input_gen_batch_map(self, batch):
        cls_preds = np.asarray(batch["cls_preds"])
        # Put in the generative batch, if offensive and vs_group
        off_and_vs_group = (cls_preds[..., 0] == 1) & (cls_preds[..., 3] == 1)
        batch_size = len(batch["input_ids"])
        in_gen_map = {}
        current_idx = 0
        for i in range(batch_size):
            if off_and_vs_group[i]:
                in_gen_map[i] = current_idx
                current_idx += 1
        return in_gen_map

    def decode_pred_tokens(self, batch):
        def __decode_pred_tokens(cls_preds, lm_pred):
            group_pred = []
            stereotype_pred = []
            # if offensive and vs_group, takes the generative part
            off_and_not_vs_group = (cls_preds[0] == 1) and (cls_preds[3] == 1)
            if off_and_not_vs_group:
                lm_pred = lm_pred[1:]  # remove <s>
                # --- get minority and stereotype tokens ---
                (sep_idx,) = np.nonzero(lm_pred == self.tokenizer.sep_token_id)
                if len(sep_idx) > 0:  # if there is a sep
                    sep_idx = sep_idx[0]
                    group_pred = lm_pred[:sep_idx]
                    stereotype_pred = lm_pred[sep_idx + 1 :]
                else:  # there is no stereotype
                    stereotype_pred = lm_pred
            return group_pred, stereotype_pred

        group_preds = []
        stereotype_preds = []
        for cls_pred, lm_pred in zip(
            batch["cls_preds"], batch["pred_tokens"], strict=True
        ):
            group_pred, stereotype_pred = __decode_pred_tokens(
                np.asarray(cls_pred), np.asarray(lm_pred)
            )
            group_preds.append(group_pred)
            stereotype_preds.append(stereotype_pred)

        group_preds = self.tokenizer.batch_decode(group_preds, skip_special_tokens=True)
        stereotype_preds = self.tokenizer.batch_decode(
            stereotype_preds, skip_special_tokens=True
        )
        return {
            "group_preds": group_preds,
            "stereotype_preds": stereotype_preds,
        }

    def __prepare_inputs_for_generation(self, batch):
        input_features = {
            "input_ids",
            "attention_mask",
        }
        batch = {k: v for k, v in batch.items() if k in input_features}
        batch = pad_batch(batch, self.data_collator)

        batch_size = batch["input_ids"].size(0)
        decoder_start_token_id = self.model.config.decoder_start_token_id
        batch["decoder_input_ids"] = (
            torch.ones((batch_size, 1), dtype=torch.long) * decoder_start_token_id
        )

        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        return batch
