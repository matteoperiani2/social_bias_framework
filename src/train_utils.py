import numpy as np
import pandas as pd
import gc
import inspect
import os
import torch
import torch.nn as nn
import warnings 

from tqdm import tqdm

import wandb
from transformers import GPT2LMHeadModel, AutoTokenizer, get_scheduler
from accelerate import Accelerator

from .utils import create_reproducible_dataloader, DummyScheduler
from .dataset import SBICDataCollator
# from .dataset_prompt import SBICDataCollator
from .losses import gpt2_llm_loss

warnings.filterwarnings('ignore') 

def make_tokenizer(config:dict,add_special_tokens=True):
    tokenizer = AutoTokenizer.from_pretrained(config['checkpoint_name'],
                                              padding_side=config['padding_side'],
                                              use_fast=True)
    
    if add_special_tokens:
        tokenizer.add_special_tokens(config['special_tokens'])
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print("List of all special token and its token_id:")
    print(" -", tokenizer.all_special_tokens)
    print(" -",tokenizer(tokenizer.all_special_tokens)["input_ids"])
    return tokenizer


def make_model(config:dict,
               tokenizer:AutoTokenizer,
               init_new_tokens = True):
    model = GPT2LMHeadModel.from_pretrained(config['checkpoint_name'])

    # init new embedding
    new_tokens = len(tokenizer) - model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    if init_new_tokens:
        model = _init_new_tokens_embs(model, new_tokens)

    # for name, para in model.named_parameters():
    #     if 'ln_' in name:
    #         para.requires_grad = False

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.sep_token_id = tokenizer.sep_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model


def _init_new_tokens_embs(model, new_tokens):
    params = model.state_dict()
    embeddings = params['transformer.wte.weight']
    pre_expansion_embeddings = embeddings[:-new_tokens,:]
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5*sigma)
    # pad_embedding = (torch.ones_like(mu)*torch.finfo(torch.float16).min).unsqueeze(0) # (1, 768)
    # other_embes = torch.stack(tuple(dist.sample() for _ in range(new_tokens-1)), dim=0) # (11,768)
    # new_embeddings = torch.cat((pad_embedding,other_embes ), dim=0) # (12, 768)
    new_embeddings = torch.stack(tuple((dist.sample() for _ in range(new_tokens))), dim=0)
    embeddings[-new_tokens:,:] = new_embeddings
    params['transformer.wte.weight'][-new_tokens:,:] = new_embeddings
    model.load_state_dict(params)
    return model


def get_data(split: str, config, aggregated=False):
    if not aggregated:
        path = os.path.join(config['data']['processed'], f"{split}.pkl")
    else:
        path = os.path.join(config['data']['aggregated'], f"{split}.pkl")
    data = pd.read_pickle(path).to_numpy()
    return data


def make_dataloader(dataset, model, tokenizer, config, shuffle=True):
    data_collator = SBICDataCollator(tokenizer=tokenizer, model=model)
    dataloader = create_reproducible_dataloader(
        dataset,
        batch_size=config['batch_size'],
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=True
    )
    return dataloader


def make_optimizer(model, config):
   return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])


def make_scheduler(optimizer, steps_per_epoch, config):
    total_steps = steps_per_epoch * config['num_epochs']
    warmup_steps = int(config['warmup_fraction'] * total_steps)
    if config.get("scheduler", "none") != "none":
        return get_scheduler(
            config['scheduler'],
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    return DummyScheduler(optimizer=optimizer)


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    config,
    monitor=True,
):
    model.train()
    if monitor:
        watch_list = [model]

    accelerator = Accelerator(mixed_precision=config['mixed_precision'], cpu=config.get('cpu', False))
    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if monitor:
        wandb.watch(watch_list, log="all", log_freq=config['log_interval'])

    # loss_fn = CustomLoss([1.,1.,1.], sep_token=torch.tensor(50258))
    
    forward_signature = set(inspect.signature(model.forward).parameters)
    step = 0
    max_iters = config['num_epochs'] * len(train_dataloader)
    print("Training...")
    with tqdm(total=max_iters, unit="batch") as pbar:
        for epoch in range(config['num_epochs']):
            for data in train_dataloader:
                pbar.set_description(F"Epoch {epoch}")
                lr = lr_scheduler.get_last_lr()[0]

                inputs = {
                    argument: value
                    for argument, value in data.items()
                    if argument in forward_signature
                }

                loss = _train_batch(
                    inputs=inputs,
                    data=data,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=gpt2_llm_loss,
                    lr_scheduler=lr_scheduler,
                    accelerator=accelerator,
                    config=config
                )

                step += 1
                pbar.update(1)

                if monitor:
                    wandb.log({
                                "train_loss": loss,
                                "lr": lr 
                              },
                              step=step)

                if (step % config['eval_interval'] == 0) or max_iters == step:
                    print(f"Evaluation after the {step} steps...")
                    # Evaluate the model
                    avg_val_loss = _train_evaluation(
                        model,
                        val_dataloader,
                        loss_fn=gpt2_llm_loss,
                        step=step
                    )
                    model.train()
                    if monitor:
                        wandb.log({
                                    'val_loss': avg_val_loss,
                                  },
                                  step=step)

    if monitor:
        wandb.unwatch(watch_list)
    
    gc.collect()
    accelerator.free_memory()
    torch.cuda.empty_cache()

def _train_batch(
    inputs,
    data,
    step,
    model,
    optimizer,
    loss_fn,
    lr_scheduler,
    config,
    accelerator
):

    outputs = model(**inputs, return_dict=True)
    
    # loss, losses = loss_fn(outputs.logits, data["classification_labels"], data["generative_labels"])
    loss = loss_fn(outputs.logits, data['labels'])

    accelerator.backward(loss)

    if config.get("gradient_clip", "none") != "none":
        accelerator.clip_grad_norm_(model.parameters(), config['gradient_clip'])

    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    return loss.item()


def _train_evaluation(
    model,
    dataloader,
    loss_fn,
    step
):
    model.eval()
    device = torch.device("cuda")
    forward_signature = set(inspect.signature(model.forward).parameters)
    avg_tot_loss = 0
    model.to(device)
    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="batch", desc=f'Eval step {step}') as pbar:
            for data in dataloader:
                inputs_kwargs = {
                    argument: value
                    for argument, value in data.items()
                    if argument in forward_signature
                }

                outputs = model(**inputs_kwargs, return_dict=True)

                loss = loss_fn(outputs.logits, data["labels"])
                # loss, split_loss = loss_fn(outputs.logits, data["classification_labels"], data["generative_labels"])

                pbar.update(1)
                avg_tot_loss += loss.item()
            
            avg_tot_loss /= len(dataloader)
            pbar.set_postfix({'avg_val_loss': avg_tot_loss})

    return avg_tot_loss