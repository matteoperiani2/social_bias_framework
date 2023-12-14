import os, random
import wandb
import numpy as np

import torch
import transformers

from src.config import CONFIG
from src.dataset_cls_head import SBICDataset
# from src.dataset_prompt import SBICDataset
from src.train_utils_head import *
from src.utils import fix_reproducibility
from src.model import GPT2WithClassificationHead

wandb.login()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

with wandb.init(project=CONFIG.wandbConfig.project, config=CONFIG.hp):

    seed=42
    fix_reproducibility(seed)

    config = CONFIG.hp

    # Make the model
    tokenizer = make_tokinzer(config, add_special_tokens=False)
    # model = make_model(config, tokenizer, init_new_tokens=True)

    gpt2_config = transformers.GPT2Config.from_pretrained('distilgpt2')
    model = GPT2WithClassificationHead(gpt2_config)
    model.resize_token_embeddings(len(tokenizer))
    params = model.state_dict()
    embeddings = params['gpt2.wte.weight']
    pre_expansion_embeddings = embeddings[:-2,:]
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    pad_emb = torch.ones_like(mu)*torch.finfo(torch.float16).min
    new_embeddings = torch.stack((pad_emb, mu), dim=0)
    embeddings[-2:,:] = new_embeddings
    params['gpt2.wte.weight'][-2:,:] = new_embeddings
    model.load_state_dict(params)

    train_data = get_data("train")[:10832]
    val_data = get_data("validation")[:2048]

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