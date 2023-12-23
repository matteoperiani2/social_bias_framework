from typing import Any, Optional, Union

import numpy as np
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class BartDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
                sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # We have to pad the labels and other features not in  `tokenizer.model_input_names` before calling `tokenizer.pad`
        # as `tokenizer.pad` method will pad only features in `tokenizer.model_input_names`
        tokenizer_input_names = set(self.tokenizer.model_input_names)
        for feature_name in features[0].keys():
            if feature_name not in tokenizer_input_names and isinstance(
                features[0][feature_name], list
            ):
                pad_id = 0
                if feature_name.endswith("labels"):
                    pad_id = self.label_pad_token_id
                elif "ids" in feature_name:
                    pad_id = self.tokenizer.pad_token_id

                self.pad_feature(feature_name, features, pad_id=pad_id)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            "labels" in features
            and "decoder_input_ids" not in features
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features

    def pad_feature(self, feature_name, features, pad_id=0):
        items = (
            [feature[feature_name] for feature in features]
            if feature_name in features[0].keys()
            else None
        )
        # We have to pad the feature before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if items is not None:
            max_item_length = max(len(item) for item in items)
            if self.pad_to_multiple_of is not None:
                max_item_length = (
                    (max_item_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [pad_id] * (max_item_length - len(feature[feature_name]))
                if isinstance(feature[feature_name], list):
                    feature[feature_name] = (
                        feature[feature_name] + remainder
                        if padding_side == "right"
                        else remainder + feature[feature_name]
                    )
                elif padding_side == "right":
                    feature[feature_name] = np.concatenate(
                        [feature[feature_name], remainder]
                    )
                else:
                    feature[feature_name] = np.concatenate(
                        [remainder, feature[feature_name]]
                    )
