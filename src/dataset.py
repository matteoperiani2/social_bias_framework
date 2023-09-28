import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from .config import Config

CONFIG: Config = Config()

class SBICDataset(Dataset):
     
    def __init__(self, data, tokenizer, max_sequence_length=None):
        super(SBICDataset).__init__()
        self.data = data #numpy array
        self.tokenizer = tokenizer
        self.labels_encoder = CONFIG.utils.label_encoder
        self.max_length = max_sequence_length if max_sequence_length is not None else tokenizer.model_max_lenght
        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row  = self.data[idx]
        post = row[5]

        # classification features
        class_features= row[:5]

        # generative features
        mionority = row[6]
        stereotype = row[7]

        input_str = self.tokenizer.bos_token + post + self.tokenizer.sep_token # [STR] post [SEP]

        class_features_enc = [self.labels_encoder[idx][val] for idx,val in enumerate(class_features)]
        label_str = self.tokenizer.sep_token.join(class_features_enc[:4]) # [SEP] grpY/N [SEP] intY/N [SEP] ... [STR] ingrpY/N (5 class)
        label_str += self.tokenizer.sep_token + mionority + self.tokenizer.sep_token + stereotype + self.tokenizer.sep_token  #[SEP] minority [SEP] stereotype [SEP]
        label_str += class_features_enc[-1] + self.tokenizer.eos_token

        # input encoding
        inputs = self.tokenizer(
           input_str+label_str, truncation=True, padding="max_length", max_length=self.max_length,
        )

        gen_inputs = self.tokenizer(
           input_str, truncation=True, padding="max_length", max_length=self.max_length,
        )

        #output encoding
        labels = self.tokenizer(
            label_str, truncation=True, padding="max_length", max_length=self.max_length
        )
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels["input_ids"]]

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "gen_input_ids": torch.tensor(gen_inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "gen_attention_mask": torch.tensor(gen_inputs["attention_mask"]),
            "labels": torch.tensor(labels),
        }