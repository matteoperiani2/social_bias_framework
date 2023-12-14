import os
import random
import torch
from transformers import LogitsProcessorList

from src.config import CONFIG
from src.dataset import SBICDataset
# from src.dataset_prompt import SBICDataset
from src.train_utils import *
from src.utils import fix_reproducibility
from src.processor import RestrictClassificationTokensProcessor

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

config=CONFIG.hp
fix_reproducibility(config.seed)

# Make the model
tokenizer = make_tokinzer(config, add_special_tokens=True)
model = make_model(config, tokenizer, init_new_tokens=False)

restricted_token_ids = [5, 10, 15]  # Replace with your specific token IDs
inputs = tokenizer(['Ciao sono Matteo!', 'Ciao Matteo!'], padding=True, return_tensors='pt')

allowed_tokens = ['<|offY|><|offN|>', '<|intY|><|intN|>', '<|sexY|><|sexN|>', '<|grpY|><|grpN|>']
allowed_tokens_ids = tokenizer(allowed_tokens)['input_ids']

processor = RestrictClassificationTokensProcessor(step_cls_tokens=allowed_tokens_ids, sep_token_id=tokenizer.sep_token_id)
logits_processor = LogitsProcessorList([processor])

model.generate(**inputs, logits_processor=logits_processor)

processor.gen_token


gc.collect()
torch.cuda.empty_cache()




