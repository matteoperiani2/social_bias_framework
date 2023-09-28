import torch
from torch.utils.data import Dataset

class SBICDataset(Dataset):
     
    def __init__(self, data, tokenizer, labels_encoder, max_sequence_length=None):
        super(SBICDataset).__init__()
        self.data = data #numpy array
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length if max_sequence_length is not None else tokenizer.model_max_length
        self.labels_encoder = labels_encoder
        self.sep_token = self.tokenizer.sep_token

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

        # input encoding
        inputs = self.tokenizer(
            post+self.sep_token, truncation=True, padding="max_length", max_length=self.max_sequence_length,
        )

        # creatig labels string
        class_features_enc = [self.labels_encoder[idx][val] for idx,val in enumerate(class_features)]
        labels_str = class_features_enc[0] + self.sep_token.join(class_features_enc[:4])
        labels_str += self.sep_token + mionority + self.sep_token + stereotype + self.sep_token
        labels_str += class_features_enc[-1]

        #output encoding
        labels = self.tokenizer(
            labels_str, truncation=True, padding="max_length", max_length=self.max_sequence_length
        )
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels["input_ids"]]

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "labels": torch.tensor(labels),
        }