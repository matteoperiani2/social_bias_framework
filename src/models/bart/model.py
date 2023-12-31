from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import ModelOutput
from transformers.models.bart.modeling_bart import (
    BartClassificationHead,
    shift_tokens_right,
)

from ...logging import WandbLogger
from ...losses import binary_kl_div_with_logits


@dataclass
class BartSBFOutput(ModelOutput):
    """
    Base class for outputs of BartSBF model.

    Args:
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores of the post (before SoftMax).
        logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    cls_logits: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BartSBF(transformers.BartPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    ]
    _keys_to_ignore_on_load_missing = ["lm_logits_bias"]

    def __init__(self, config: transformers.BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = transformers.BartModel(config)
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )
        self.register_parameter(
            "lm_logits_bias",
            nn.Parameter(torch.zeros((1, self.model.shared.num_embeddings))),
        )
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BartSBFOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        ## classification head
        encoder_hidden_state = outputs.encoder_last_hidden_state  # (B, S, E)
        pooled_outputs = torch.mean(encoder_hidden_state, dim=1)  # (B, E)
        cls_logits = self.classification_head(pooled_outputs)

        # lm head
        decoder_hidden_state = outputs.last_hidden_state
        lm_logits = self.lm_head(decoder_hidden_state)
        lm_logits = lm_logits + self.lm_logits_bias.to(lm_logits.device)

        if not return_dict:
            return (
                cls_logits,
                lm_logits,
            ) + outputs[1:]

        output = BartSBFOutput(
            cls_logits=cls_logits,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        return output

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past


def loss(config):
    wandb_logger = WandbLogger()
    cls_weight = config["model"].get("cls_weight", 1.0)
    gen_weight = config["model"].get("gen_weight", 1.0)

    freq = config["model"]["classification_pos_freq"]
    freq = torch.tensor(freq).float()
    freq = torch.stack((1 - freq, freq), dim=0)  # 2 x cls_features
    cls_weights = 1 / freq

    def __kldiv(outputs, data):
        cls_weights = None
        cls_logits = outputs.cls_logits  # batch_size x cls_features
        cls_loss = binary_kl_div_with_logits(
            cls_logits, data["cls_labels"], weight=cls_weights, reduction="batchmean"
        )
        return cls_loss

    def __bce(outputs, data):
        cls_logits = outputs.cls_logits  # batch_size x cls_features
        labels = data["cls_labels"]

        is_cls_valid = labels != -100

        # binarize
        labels = (labels > 0.5).float()  # batch_size x cls_features

        weight = torch.gather(
            cls_weights.to(cls_logits.device), dim=0, index=labels.long()
        )
        weight[~is_cls_valid] = 0.0
        n_batches = torch.any(is_cls_valid, dim=-1).sum()

        loss = F.binary_cross_entropy_with_logits(
            cls_logits, labels, weight=weight, reduction="none"
        )  # batch_size x cls_features
        loss = loss.sum() / n_batches
        return loss

    def __gen_loss(outputs, data):
        logits = outputs.logits  # batch_size x seq_length x vocab_size
        logits = logits.transpose(-1, -2)  # batch_size x vocab_size x seq_length
        labels = data["labels"]
        mask = labels != -100
        n_valid_tokens = torch.sum(mask, dim=-1)  # batch_size
        n_valid_batches = torch.any(mask, dim=-1).sum()  # (1, )

        gen_loss = F.cross_entropy(
            logits, labels, reduction="none"
        )  # batch_size x seq_length
        gen_loss = torch.sum(gen_loss, dim=-1) / torch.clamp(
            n_valid_tokens, min=1e-6
        )  # batch_size
        gen_loss = torch.sum(gen_loss) / torch.clamp(n_valid_batches, min=1e-6)  # (1,)

        return gen_loss

    cls_loss_name = config["model"].get("cls_loss", "kldiv")
    cls_loss_fn = __bce if cls_loss_name == "bce" else __kldiv

    def loss(outputs, data):
        cls_loss = cls_loss_fn(outputs, data)
        gen_loss = __gen_loss(outputs, data)
        loss = cls_weight * cls_loss + gen_weight * gen_loss

        wandb_logger.log_step(
            classification_loss=cls_loss.item(),
            generative_loss=gen_loss.item(),
            cls_weight=cls_weight,
            gen_weight=gen_weight,
        )
        return loss

    return loss
