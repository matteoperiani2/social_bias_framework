from typing import Union

import os
import datasets
import numpy as np
import torch
import transformers

from .logit_processor import GPT2RestrictTokensLogitsProcessor

from ...config import Config
from ...utils import filter_model_inputs, pad_batch, print_if_verbose
from .helper import GPT2Helper


class GPT2Inference:
    def __init__(self, checkpoint_name, config: Config) -> None:
        self.config = config
        helper = GPT2Helper(Config.to_dict(config))
        self.tokenizer = helper.make_tokenizer()

        self.model = helper.make_model(self.tokenizer)
        checkpoint_path = os.path.join(
            config.model.checkpoint_dir, checkpoint_name + ".pt"
        )
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        self.data_collator = helper.make_data_collator(self.tokenizer, self.model)

        self.__cls_tokens = self.__create_cls_tokens()
        self.__positive_cls_tokens = [tokens[0] for tokens in self.__cls_tokens]

    def __create_cls_tokens(self):
        cls_position_tokens = [
            ["<|offY|>", "<|offN|>"],
            ["<|intY|>", "<|intN|>"],
            ["<|sexY|>", "<|sexN|>"],
            ["<|grpY|>", "<|grpN|>"],
            ["<|ingrpY|>", "<|ingrpN|>"],
        ]
        special_tokens_dict = dict(
            zip(
                self.tokenizer.additional_special_tokens,
                self.tokenizer.additional_special_tokens_ids,
                strict=True,
            )
        )
        cls_tokens = [
            special_tokens_dict[token]
            for pos_tokens in cls_position_tokens
            for token in pos_tokens
        ]
        cls_tokens = [
            [cls_tokens[i], cls_tokens[i + 1]] for i in range(len(cls_tokens))[::2]
        ]
        return cls_tokens

    def generate_predictions(
        self,
        dataset: Union[datasets.Dataset, datasets.DatasetDict],
        verbose=True,
    ):
        self.prepare_model()

        print_if_verbose(
            "Generating output tokens ...",
            verbose=verbose,
        )
        dataset = dataset.map(
            self.generate_batch_predictions,
            batched=True,
            batch_size=self.config.model.generate_batch_size,
        )

        print_if_verbose(
            "Decoding predicted tokens into classification features, groups and stereotypes ...",
            verbose=verbose,
        )
        dataset = dataset.map(
            self.decode_pred_tokens,
            batched=True,
            batch_size=self.config.model.generate_batch_size,
        )

        return dataset

    def generate_batch_predictions(
        self,
        batch,
    ):
        model_inputs = self.__prepare_inputs_for_generation(batch)

        logits_processor = self.__create_logits_processor(self.model.device)
        logits_processor = transformers.LogitsProcessorList([logits_processor])

        pred_tokens = self.model.generate(
            **model_inputs,
            logits_processor=logits_processor,
            **self.config.model["generation_params"]
        )

        pred_tokens = pred_tokens.detach().cpu().tolist()
        return {"pred_tokens": pred_tokens}

    def decode_pred_tokens(self, batch):
        class_preds = []
        group_preds = []
        stereotype_preds = []

        pred_tokens = np.asarray(batch["pred_tokens"])

        # remove from the generated the input prompt
        pred_tokens = [
            pred[np.where(pred == self.tokenizer.sep_token_id)[0][0] + 1 :]
            for pred in pred_tokens
        ]

        for pred in pred_tokens:
            sep_idx = np.where(pred == self.tokenizer.sep_token_id)[0]
            eos_idx = np.where(pred == self.tokenizer.eos_token_id)[0][0]

            # --- get classification tokens ---
            # concatenate first 4 tokens with the token generated before the eos
            cls_preds = np.concatenate((pred[:4], [pred[eos_idx - 1]]))
            bin_cls_preds = [
                int(pred == pos_token)
                for pred, pos_token in zip(
                    cls_preds, self.__positive_cls_tokens, strict=True
                )
            ]

            # if the model predict not offensive or not to a group, ignore the generation
            if bin_cls_preds[0] == 0 or bin_cls_preds[-2] == 0:
                bin_cls_preds[-2] = 0
                bin_cls_preds[-1] = 0
                class_preds.append(bin_cls_preds)
                group_preds.append([])
                stereotype_preds.append([])
                continue

            class_preds.append(bin_cls_preds)

            # --- get group and stereotype tokens ---
            if len(sep_idx) > 2:  # if there are at least 3 sep
                # select as minority tokens, those tokens that are between first 2 sep
                group_preds.append(pred[sep_idx[0] + 1 : sep_idx[1]])
                stereotype_preds.append(pred[sep_idx[1] + 1 : sep_idx[2]])
            elif len(sep_idx) > 1:  # if there are at least 2 sep
                group_preds.append(pred[sep_idx[0] + 1 : sep_idx[1]])
                stereotype_preds.append(pred[sep_idx[1] + 1 : -2])
            else:  # if there is only 1 sep
                # group are those tokens betwen sep and second-to-last token
                # for stereotypes no tokens are selected
                group_preds.append(pred[sep_idx[0] + 1 : eos_idx - 2])
                stereotype_preds.append([])

        group_preds = self.tokenizer.batch_decode(group_preds)
        stereotype_preds = self.tokenizer.batch_decode(stereotype_preds)

        return {
            "cls_preds": class_preds,
            "group_preds": group_preds,
            "stereotype_preds": stereotype_preds,
        }

    def __create_logits_processor(self, device):
        return GPT2RestrictTokensLogitsProcessor(
            step_cls_tokens=self.__cls_tokens,
            sep_token_id=self.tokenizer.sep_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.config.model["generation_params"]["max_new_tokens"],
            device=device,
        )

    def predict(self, batch, cls_threshold=0.5):
        self.prepare_model()
        batch.update(self.generate_batch_predictions(batch))
        batch.update(self.decode_pred_tokens(batch))

        return batch

    def prepare_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()

    def __prepare_inputs_for_generation(self, batch):
        model_inputs = filter_model_inputs(self.model, batch)
        model_inputs = pad_batch(model_inputs, self.data_collator)

        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        return model_inputs
