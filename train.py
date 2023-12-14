import os
import random
import wandb

import torch
from transformers import set_seed

from src.config import CONFIG
from src.dataset import SBICDataset
# from src.dataset_prompt import SBICDataset
from src.train_utils import *
from src.utils import fix_reproducibility

wandb.login()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

with wandb.init(project=CONFIG.wandbConfig.project, config=CONFIG.hp):
    config = wandb.config
    fix_reproducibility(config.seed)

    # Make the model
    tokenizer = make_tokinzer(config, add_special_tokens=True)
    model = make_model(config, tokenizer, init_new_tokens=True)
    # tokenizer = make_tokinzer(config, cross_attn=False, add_special_tokens=True)
    # model = make_model(config, tokenizer, add_cross_attn=False, add_special_tokens=True)

    # Make the data
    train_data = get_data("train")
    val_data = get_data("validation")
    
    # train_data = np.array(random.choices(train_data, k=10832))
    val_data = np.array(random.choices(val_data, k=2048))
    
    train_dataset = SBICDataset(train_data, tokenizer)
    val_dataset = SBICDataset(val_data, tokenizer)

    train_dataloader = make_dataloader(train_dataset, model, tokenizer, config)
    val_dataloader = make_dataloader(val_dataset, model, tokenizer, config, shuffle=False)

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
        config
    )

torch.save(model.state_dict(), f"checkpoints/{config.checkpoint_name}_full_init.pt")

gc.collect()
torch.cuda.empty_cache()




