import numpy as np
from typing import Any, Optional, Union
from torch.utils.data import Dataset
import transformers 

from .config import Config

CONFIG: Config = Config()

class SBICDataset(Dataset):
     
    def __init__(self, data, tokenizer, is_training=True, max_sequence_length=None):
        super(SBICDataset).__init__()
        self.data = data #numpy array
        self.tokenizer = tokenizer
        self.labels_encoder = {0: 'no', 1:'yes', 2: ''}
        self.max_length = max_sequence_length if max_sequence_length is not None else tokenizer.model_max_length
        self.is_training = is_training

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row  = self.data[idx]
        post = row[0]

        # classification features
        class_features = np.append(row[1:5], [row[-2]], axis=0)

        # free-text features
        mionority = row[5]
        stereotype = row[6]

        # PROMPT
        # post: post
        # offensive: yes/no
        # intentional: yes/no
        # sex: yes/no
        # group: yes/no

        input_prompt = 'post: ' + post + '\n'

        label_prompt = f'offensive: {self.labels_encoder[class_features[0]]}\n'
        label_prompt += f'intentional: {self.labels_encoder[class_features[1]]}\n'
        label_prompt += f'sex: {self.labels_encoder[class_features[2]]}\n'
        label_prompt += f'group: {self.labels_encoder[class_features[3]]}\n'
        label_prompt += f'in-group: {self.labels_encoder[class_features[4]]}\n'
        label_prompt += f'minority: {mionority}\n'
        label_prompt += f'stereotype: {stereotype}\n'
    
        inputs = self.tokenizer(
            text=input_prompt,
            text_pair=label_prompt,
            truncation="only_first",
            max_length=self.max_length,
        )

        labels = self.tokenizer.encode(label_prompt)
        labels.append(self.tokenizer.eos_token_id)

        return {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "labels": labels
        }
    

class SBICDataCollator(transformers.DataCollatorForSeq2Seq):
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

    tokenizer: transformers.PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        max_ids_len = max([len(feature["input_ids"]) for feature in features])
        max_labels_len = max([len(feature["labels"]) for feature in features]) if "labels" in features[0].keys() else -1
        max_pad_length = max_ids_len if max_ids_len > max_labels_len else max_labels_len

        if max_labels_len != -1:
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_pad_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int32)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int32)

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