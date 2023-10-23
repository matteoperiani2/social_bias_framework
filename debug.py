# %%
import numpy as np
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import wandb

import torch
from transformers import set_seed

from src.config import CONFIG
from src.dataset import SBICDataset
from src.train_utils import *

# %%
tokenizer = make_tokinzer(CONFIG.hp)
model = make_model(CONFIG.hp, tokenizer)
checkpoints = "distilgpt2_full_8_1"
# checkpoints = "distilgpt2_30720_16"
model.load_state_dict(torch.load(f"checkpoints/{checkpoints}.pt"))

# %%
n_samples = {
    "train": 128,
    "validation": 32
}

for split in ["train", "validation"]:
    data = get_data(split)[:n_samples[split]]
    dataset = SBICDataset(data, tokenizer, is_training=False)
    dataloader = make_dataloader(dataset, model, tokenizer, CONFIG.hp, split="validation")

    res = evaluate(model, tokenizer, dataloader, CONFIG.hp)

    annotation_type = ["Offensive", "Intent", "Sex", "Group", "In-Group"]
    print(f"Classification F1 on {split} set: avg={np.mean(res['clasification_f1_score']):.3f}")
    for type, score in zip(annotation_type, res['clasification_f1_score']):
        print(f" - {type}: {score:.3f}")
    print(f"Minority RougeL-f1 on {split} set: {res['minority_rouge_f1_score']:.3f}")
    print(f"Stereotype RougeL-f1 on {split} set: {res['stereotype_rouge_f1_score']:.3f}")


