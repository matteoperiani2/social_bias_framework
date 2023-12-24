import gc
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.modules import Embedding
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from ...utils import filter_model_inputs, pad_batch
from .logit_processor import GPT2RestrictTokensLogitsProcessor


class GPT2SBF(transformers.GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = transformers.GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

class GPT2Loss(nn.Module):

    def __init__(self, tokenizer, config, init_weight) -> None:
        super(GPT2Loss, self).__init__()
        self.weight = torch.ones(len(tokenizer))
        if init_weight:
            special_tokens_dict = {token:token_id for token,token_id in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}
            for cls_name,freq in config['model']['classification_pos_freq'].items():
                pos_id = special_tokens_dict[f'<|{cls_name}Y|>']
                neg_id =  special_tokens_dict[f'<|{cls_name}N|>']
                self.weight[pos_id] = 1/freq
                self.weight[neg_id] = 1/(1-freq)

        print(self.weight[50256:])

    def forward(self, outputs, data):
        logits = outputs.logits.transpose(-1, -2)
        labels = data["labels"]
        loss = F.cross_entropy(logits, labels, reduction="none", weight=self.weight.to(logits.device))  # (BATCH, SEQ_LEN)

        n_valid_tokens = torch.sum(labels != -100, dim=-1)  # (BATCH, )
        loss = torch.sum(loss, dim=-1) / torch.clamp(n_valid_tokens, min=1e-7)  # (BATCH, )
        loss = torch.mean(loss)  # (1,)
        return loss



def generate_predictions(data, model, tokenizer, collator, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    special_tokens_dict = {token:token_id for token,token_id in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
    cls_position_tokens = [
        ["<|offY|>","<|offN|>"],
        ["<|intY|>","<|intN|>"],
        ["<|sexY|>","<|sexN|>"],
        ["<|grpY|>","<|grpN|>"],
        ["<|ingrpY|>","<|ingrpN|>"]
    ]
    cls_tokens = [special_tokens_dict[token] for pos_tokens in cls_position_tokens for token in pos_tokens]
    cls_tokens = [[cls_tokens[i], cls_tokens[i+1]] for i in range(len(cls_tokens))[::2]]

    params = {
        "model": model,
        "tokenizer": tokenizer,
        "collator": collator,
        "cls_tokens": cls_tokens,
        "pos_cls_tokens": [tokens[0] for tokens in cls_tokens],
        "gen_cfg": config.model["generation_params"],
        "device": device,
    }

    with torch.no_grad():
        predictions = data.map(
            _generate_batch_predicitons,
            fn_kwargs=params,
            batched=True,
            batch_size= config.model["generate_batch_size"],
            remove_columns=["input_ids", "attention_mask"],
            load_from_cache_file=False,
        )

    predictions = predictions.map(
        _aggregate_labels,
        load_from_cache_file=False,
        fn_kwargs={'cols': config.classification_columns},
        remove_columns=config.classification_columns
    )

    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    return predictions


def _generate_batch_predicitons(
    data, model, tokenizer, collator, cls_tokens, pos_cls_tokens, gen_cfg, device
):
    inputs = filter_model_inputs(model, data)
    inputs = pad_batch(inputs, collator)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    processor = GPT2RestrictTokensLogitsProcessor(
        step_cls_tokens=cls_tokens,
        sep_token_id=tokenizer.sep_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=gen_cfg.max_new_tokens,
        device=device,
    )
    logits_processor = transformers.LogitsProcessorList([processor])

    generate_out = model.generate(
        **inputs, logits_processor=logits_processor, **gen_cfg
    )

    generate_out = generate_out.detach().cpu().numpy()
    cls_preds, minorities_preds, stereotypes_preds = _process_predictions(
        tokenizer, generate_out, pos_cls_tokens
    )

    return {
        "cls_preds": cls_preds,
        "group_preds": minorities_preds,
        "stereotype_preds": stereotypes_preds,
    }


def _process_predictions(tokenizer, predictions, positive_cls_tokens):
    class_preds = []
    minority_preds = []
    stereotype_preds = []

    # remove from the generated the input prompt
    predictions = [
        pred[np.where(pred == tokenizer.sep_token_id)[0][0] + 1 :]
        for pred in predictions
    ]

    for pred in predictions:
        sep_idx = np.where(pred == tokenizer.sep_token_id)[0]
        eos_idx = np.where(pred == tokenizer.eos_token_id)[0][0]

        # --- get classification tokens ---
        # concatenate first 4 tokens with the token generated before the eos
        cls_preds = np.concatenate((pred[:4], [pred[eos_idx - 1]]))
        bin_cls_preds = [
            int(pred == pos_token)
            for pred, pos_token in zip(cls_preds, positive_cls_tokens, strict=True)
        ]

        # if the model predict not offensive or not to a group, ignore the generation
        if bin_cls_preds[0] == 0 or bin_cls_preds[-2] == 0:
            bin_cls_preds[-2] = 0
            bin_cls_preds[-1] = 0
            class_preds.append(bin_cls_preds)
            minority_preds.append([])
            stereotype_preds.append([])
            continue

        class_preds.append(bin_cls_preds)

        # --- get minority and stereotype tokens ---
        if len(sep_idx) > 2:  # if there are at least 3 sep
            # select as minority tokens, those tokens that are between first 2 sep
            minority_preds.append(pred[sep_idx[0] + 1 : sep_idx[1]])
            stereotype_preds.append(pred[sep_idx[1] + 1 : sep_idx[2]])
        elif len(sep_idx) > 1:  # if there are at least 2 sep
            minority_preds.append(pred[sep_idx[0] + 1 : sep_idx[1]])
            stereotype_preds.append(pred[sep_idx[1] + 1 : -2])
        else:  # if there is only 1 sep
            # minority are those tokens betwen sep and second-to-last token
            # for stereotypes no tokens are selected
            minority_preds.append(pred[sep_idx[0] + 1 : eos_idx - 2])
            stereotype_preds.append([])

    minority_preds = tokenizer.batch_decode(minority_preds)
    stereotype_preds = tokenizer.batch_decode(stereotype_preds)

    return class_preds, minority_preds, stereotype_preds


def _aggregate_labels(data, cols):
    values = [int(v>=0.5) if v is not None else -1 for k,v in data.items() if k in cols]
    return {'labels': values}