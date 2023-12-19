import pandas as pd
import gc
import inspect
import os
import torch
import torch.nn as nn
import warnings 
import datasets

from tqdm import tqdm

import wandb
from transformers import get_scheduler
from accelerate import Accelerator

from .utils import create_reproducible_dataloader, DummyScheduler, init_cross_entropy_weights
from .helper import GPT2TrainHelper, BartTrainHelper
import src.models.gpt2 as gpt2
import src.models.bart as bart


def get_model_helper(config):
    if config['name'] == 'gpt2':
        return GPT2TrainHelper(config)
    elif config['name'] == 'bart':
        return BartTrainHelper(config)
    else:
        raise ValueError("Invalid name. Possible values are [gpt2, bart]")


def make_dataloader(dataset, collator, config, shuffle=True):
    dataloader = create_reproducible_dataloader(
        dataset,
        batch_size=config['batch_size'],
        collate_fn=collator,
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


def make_loss(confg):
    if confg['name'] == 'gpt2':
        return gpt2.loss
    elif confg['name'] == 'bart':
        return bart.loss

def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
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
                    if argument in forward_signature and argument != 'labels'
                }

                loss = _train_batch(
                    inputs=inputs,
                    data=data,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
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
                        loss_fn=loss_fn,
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
    model,
    optimizer,
    loss_fn,
    lr_scheduler,
    config,
    accelerator
):

    outputs = model(**inputs, return_dict=True)
    
    loss = loss_fn(outputs, data, config)

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

                loss =  loss_fn(outputs, data)

                pbar.update(1)
                avg_tot_loss += loss.item()
            
            avg_tot_loss /= len(dataloader)
            pbar.set_postfix({'avg_val_loss': avg_tot_loss})

    return avg_tot_loss