import os
import wandb

import torch
from transformers import set_seed

from src.config import CONFIG
from src.dataset import SBICDataset
from src.train_utils import *

wandb.login()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

with wandb.init(project=CONFIG.wandbConfig.project, config=CONFIG.hp):
    config = wandb.config

    set_seed(config.seed)

    # Make the model
    tokenizer = make_tokinzer(config)
    model = make_model(config, tokenizer)

    # Make the data
    train_data = get_data("train")
    val_data = get_data("validation")
    
    if config.train_perc != 1 or config.val_perc != 1:
        train_size = int((config.train_perc * len(train_data)) / config.batch_size) * config.batch_size
        val_size = int((config.val_perc * len(val_data)) / config.batch_size) * config.batch_size
        train_idxs = np.random.choice(train_data.shape[0], train_size, replace=False)
        val_idxs = np.random.choice(val_data.shape[0], val_size, replace=False)
        
        train_data = train_data[train_idxs]
        val_data = val_data[val_idxs]


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
        use_def_loss=False
    )

    torch.save(model.state_dict(), f"checkpoints/{config.checkpoint_name}_{train_size}_{config.num_epochs}_sl.pt")

gc.collect()
torch.cuda.empty_cache()




