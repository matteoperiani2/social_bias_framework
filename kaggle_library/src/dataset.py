import numpy as np
from typing import Any, Optional, Union
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

class SBICDataset(Dataset):
     
    def __init__(self, data, tokenizer, cls_token_map, is_training=True, max_sequence_length=None):
        super(SBICDataset).__init__()
        self.data = data #numpy array
        self.tokenizer = tokenizer
        self.cls_token_map = cls_token_map
        self.max_length = max_sequence_length if max_sequence_length is not None else tokenizer.model_max_length
        self.is_training = is_training

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row  = self.data[idx]
        post = row[0]

        # classification features
        class_features = np.append(row[1:5], [row[-1]], axis=0)

        # free-text features
        mionority = row[5]
        stereotype = row[6]

        # INPUT: post <|sep|> offY/N intY/N sexY/N [grpY/N] <|sep|> minority <|sep|> stereotype <|sep|> ingrpY/N

        input_str = post + self.tokenizer.sep_token # post <|sep|>

        class_features_enc = [self.cls_token_map[idx][val] for idx,val in enumerate(class_features)]
        
        if self.is_training:

            label_str = "".join(class_features_enc[:4]) + self.tokenizer.sep_token
        
            if mionority != "" and  stereotype!= "":
                label_str += mionority + self.tokenizer.sep_token + stereotype + self.tokenizer.sep_token  # minority <|sep|> stereotype <|sep|>
                label_str += class_features_enc[-1] # ingrpY/N
 
            # input encoding
            input_ids = self.tokenizer.encode(
                text=input_str,
                text_pair=label_str,
                truncation="only_first",
                max_length=self.max_length,
            )
            attention_mask = np.ones_like(input_ids, dtype=np.uint8)
            attention_mask[np.asarray(input_ids) == self.tokenizer.pad_token_id] = 0

            # output encoding
            labels = self.tokenizer.encode(text=label_str)
            labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
            labels[4] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask.tolist(),
                "labels": labels,
            }
        else:
            inputs = self.tokenizer(input_str, truncation=True, max_length=self.max_length)

            return {
                "input_ids": inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                "class_labels": class_features,
                "minority_labels": mionority,
                "stereotype_labels": stereotype
            }


class SBICDataCollator(DataCollatorForSeq2Seq):
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

        # labels_present = "classification_labels" in features[0].keys() | 
        max_ids_len = max([len(feature["input_ids"]) for feature in features])
        max_labels_len = max([len(feature["labels"]) for feature in features]) if "labels" in features[0].keys() else -1
        max_pad_length = max_ids_len if max_ids_len > max_labels_len else max_labels_len

        if max_labels_len != -1:
            padding_side = self.tokenizer.padding_side
            present_labels = list(features[0].keys())[2:]
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_pad_length - len(feature[present_labels[0]]))
                if isinstance(feature[present_labels[0]], list):
                    for label in present_labels:   
                        feature[label] = (
                            feature[label] + remainder if padding_side == "right" else remainder + feature[label]
                        )
                elif padding_side == "right":
                    for label in present_labels:
                        feature[label] = np.concatenate([feature[label], remainder]).astype(np.int32)
                else:
                    for label in present_labels:
                        feature[label] = np.concatenate([remainder, feature[label]]).astype(np.int32)

        if 'labels' in features[0].keys():
            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        else:
            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )


        return features