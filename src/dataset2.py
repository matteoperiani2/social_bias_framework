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
        # post = row[5]
        post = "Stupid fucking nigger LeBron. You flopping stupid jungle bunny monkey faggot."

        # classification features
        off= row[3]

        # generative features
        mionority = row[6]
        stereotype = row[8]

        input_str = post

        # input encoding
        inputs = self.tokenizer(
           post+"[SEP][offY][intN][sexN][ingrpY]<|endoftext|>", truncation=True, padding="max_length", max_length=self.max_length,
        )

        labels = self.tokenizer(
            "[SEP][offY][intN][sexN][ingrpY]<|endoftext|>", truncation=True, padding="max_length", max_length=self.max_length,
        )

        #output encoding
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels["input_ids"]]

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "labels": torch.tensor(labels),
        }