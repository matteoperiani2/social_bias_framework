import os, random
import wandb
import numpy as np

import torch
from transformers import set_seed

from src.config import CONFIG
from src.dataset import SBICDataset
from src.train_utils import *

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
set_seed(42)
# torch.autograd.set_detect_anomaly(True)

config = CONFIG.hp

# Make the model
tokenizer = make_tokinzer(config)
model = make_model(config, tokenizer)

# Make the data
train_data = get_data("train")[:10832]
val_data = get_data("validation")[:1024]

train_dataset = SBICDataset(train_data, tokenizer)
val_dataset = SBICDataset(val_data, tokenizer)

train_dataloader = make_dataloader(train_dataset, model, tokenizer, config, split="train")
val_dataloader = make_dataloader(val_dataset, model, tokenizer, config, split="validation")

# Make the loss, the optimizer and the scheduler
optimizer = make_optimizer(model, config)
scheduler = make_scheduler(
    optimizer, steps_per_epoch=len(train_dataloader), config=config
)

train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    config, 
    monitor=False
)

torch.save(model.state_dict(), f"checkpoints/{config.checkpoint_name}_10832_{config.num_epochs}_sl2-half.pt")

gc.collect()
torch.cuda.empty_cache()