from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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
    def __init__(self, tokenizer, config) -> None:
        super(GPT2Loss, self).__init__()
        self.weight = torch.ones(len(tokenizer))
        if config["model"].get("weight_loss", False):
            special_tokens_dict = dict(
                zip(
                    tokenizer.all_special_tokens, tokenizer.all_special_ids, strict=True
                )
            )
            for cls_name, freq in config["model"]["classification_pos_freq"].items():
                pos_id = special_tokens_dict[f"<|{cls_name}Y|>"]
                neg_id = special_tokens_dict[f"<|{cls_name}N|>"]
                self.weight[pos_id] = 1 / freq
                self.weight[neg_id] = 1 / (1 - freq)

    def forward(self, outputs, data):
        logits = outputs.logits.transpose(-1, -2)
        labels = data["labels"]
        loss = F.cross_entropy(
            logits, labels, reduction="none", weight=self.weight.to(logits.device)
        )  # (BATCH, SEQ_LEN)

        n_valid_tokens = torch.sum(labels != -100, dim=-1)  # (BATCH, )
        loss = torch.sum(loss, dim=-1) / torch.clamp(
            n_valid_tokens, min=1e-7
        )  # (BATCH, )
        loss = torch.mean(loss)  # (1,)
        return loss


# def generate_predictions(data, model, tokenizer, collator, config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     model.to(device)

#     special_tokens_dict = dict(
#         zip(
#             tokenizer.additional_special_tokens,
#             tokenizer.additional_special_tokens_ids,
#             strict=True,
#         )
#     )
#     cls_position_tokens = [
#         ["<|offY|>", "<|offN|>"],
#         ["<|intY|>", "<|intN|>"],
#         ["<|sexY|>", "<|sexN|>"],
#         ["<|grpY|>", "<|grpN|>"],
#         ["<|ingrpY|>", "<|ingrpN|>"],
#     ]
#     cls_tokens = [
#         special_tokens_dict[token]
#         for pos_tokens in cls_position_tokens
#         for token in pos_tokens
#     ]
#     cls_tokens = [
#         [cls_tokens[i], cls_tokens[i + 1]] for i in range(len(cls_tokens))[::2]
#     ]

#     params = {
#         "model": model,
#         "tokenizer": tokenizer,
#         "collator": collator,
#         "cls_tokens": cls_tokens,
#         "pos_cls_tokens": [tokens[0] for tokens in cls_tokens],
#         "gen_cfg": config.model["generation_params"],
#         "device": device,
#     }

#     with torch.no_grad():
#         predictions = data.map(
#             _generate_batch_predicitons,
#             fn_kwargs=params,
#             batched=True,
#             batch_size=config.model["generate_batch_size"],
#             remove_columns=["input_ids", "attention_mask"],
#             load_from_cache_file=False,
#         )

#     predictions = predictions.map(
#         _aggregate_labels,
#         load_from_cache_file=False,
#         fn_kwargs={"cols": config.classification_columns},
#         remove_columns=config.classification_columns,
#     )

#     model.cpu()
#     gc.collect()
#     torch.cuda.empty_cache()

#     return predictions


def _aggregate_labels(data, cols):
    values = [
        int(v >= 0.5) if v is not None else -1 for k, v in data.items() if k in cols
    ]
    return {"labels": values}
