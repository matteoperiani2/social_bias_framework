from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    
    def __init__(self, df, model, history=False, encoder_max_len=512, decoder_max_len=32):
        super(TokenizedDataset).__init__()
        self.df = df
        self.tokenizer = model.tokenizer 

        
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, idx):

        span_start, span_end = find_indexes(self.tokenizer, passage, start_idx, end_idx, self.encoder_max_len, self.name)

        if self.history is False:
          # INPUT:  [CLS] QUESTION [SEP] PASSAGE [SEP] *[PAD]
          input_encoding = self.tokenizer(
              question, 
              passage, 
              padding='max_length',
              truncation=True,
              max_length=self.encoder_max_len, 
          )
          rationale_encoding = self.tokenizer(
              question, 
              rationale, 
              padding='max_length',
              truncation=True,
              max_length=self.encoder_max_len, 
          )          
        else:
          # INPUT:  [CLS] HISTORY + QUESTION [SEP] PASSAGE [SEP] *[PAD]
          input_encoding = self.tokenizer(
              history + question, 
              passage, 
              padding='max_length',
              truncation=True,
              max_length=self.encoder_max_len, 
          )
          rationale_encoding = self.tokenizer(
              history + question, 
              rationale, 
              padding='max_length',
              truncation=True,
              max_length=self.encoder_max_len, 
          )

        # OUTPUT: [CLS] ANSWER [SEP] *[PAD]
        output_encoding = self.tokenizer(
            answer, truncation=True, padding='max_length', max_length=self.decoder_max_len 
        )
                      
        labels = output_encoding.input_ids
        
        # Ignore the loss of the [PAD] labels by setting them to -100
        labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
        labels = torch.tensor(labels, dtype=torch.int64)
          
        inputs = {
            'input_ids': torch.tensor(input_encoding.input_ids, dtype=torch.int64),
            'attention_mask': torch.tensor(input_encoding.attention_mask, dtype=torch.int64),

            'rationale_ids': torch.tensor(rationale_encoding.input_ids, dtype=torch.int64),
            'rationale_attention_mask': torch.tensor(rationale_encoding.attention_mask, dtype=torch.int64),

            'start_positions': torch.tensor(span_start, dtype=torch.int64),
            'end_positions': torch.tensor(span_end, dtype=torch.int64)
        }
          
        return inputs, labels

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

        inputs = self.tokenizer(post+self.tokenizer.sep_token, padding="max_length", max_length=self.max_sequence_length, return_tensor="pt")
        class_features_enc = [self.labels_encoder[idx][val] for idx,val in enumerate(class_features)]

        # creatig labels string
        labels_str = self.sep_token.join(class_features_enc[:4])
        labels_str += self.sep_token + mionority + self.sep_token + stereotype + self.sep_token
        labels_str += class_features_enc[-1]

        return self.data[idx]



# class SBFGpt2Collator(object):
#     r"""
#     Data Collator used for GPT2 in a classificaiton task. 

#     It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
#     can go straight into a GPT2 model.

#     This class is built with reusability in mind: it can be used as is as long
#     as the `dataloader` outputs a batch in dictionary format that can be passed 
#     straight into the model - `model(**batch)`.

#     Arguments:

#         use_tokenizer (:obj:`transformers.tokenization_?`):
#             Transformer type tokenizer used to process raw text into numbers.

#         labels_ids (:obj:`dict`):
#             Dictionary to encode any labels names into numbers. Keys map to 
#             labels names and Values map to number associated to those labels.

#         max_sequence_len (:obj:`int`, `optional`)
#             Value to indicate the maximum desired sequence to truncate or pad text
#             sequences. If no value is passed it will used maximum sequence size
#             supported by the tokenizer and model.

#     """

#     def __init__(self, tokenizer, class_label_encoder, max_sequence_len=None):
#         # Tokenizer to be used inside the class.
#         self.use_tokenizer = tokenizer
#         # Check max sequence length.
#         self.max_sequence_len = tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
#         # Label encoder used inside the class.
#         self.class_label_encoder = class_label_encoder


#     def __call__(self, sequences):
#         r"""
#         This function allowes the class objesct to be used as a function call.
#         Sine the PyTorch DataLoader needs a collator function, I can use this 
#         class as a function.

#         Arguments:

#             item (:obj:`list`):
#                 List of texts and labels.

#         Returns:
#             :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
#             It holddes the statement `model(**Returned Dictionary)`.
#         """

#         # Get all texts from sequences list.
#         posts = [sequence[0] for sequence in sequences]

#         # Get all labels from sequences list.
#         class_label = [(sequence[4:8]) for sequence in sequences]
#         # Encode all labels using label encoder.
#         labels = [self.labels_encoder[label] for label in labels]
#         # Call tokenizer on all texts to convert into tensors of numbers with 
#         # appropriate padding.
#         inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
#         # Update the inputs with the associated encoded labels as tensor.

#         return inputs