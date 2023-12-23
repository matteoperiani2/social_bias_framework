import gc
import inspect
import os
import random

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .logging import WandbLogger
from .utils import create_dirs_for_file


class DummyLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {}


def fix_reproducibility(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(*args, **kwargs):
    generator = torch.Generator()
    return DataLoader(*args, **kwargs, worker_init_fn=seed_worker, generator=generator)


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
    wandb_logger = WandbLogger()
    watch_list = [model]

    accumulation_steps = config.model.get("accumulation_steps", 1)
    mixed_precision = config.model["mixed_precision"]
    cpu = config.get("cpu", False)
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision=mixed_precision,
        cpu=cpu,
    )
    (
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer, lr_scheduler
    )

    if monitor:
        wandb.watch(watch_list, log="all", log_freq=config.model["log_interval"])

    # Run training and track with wandb
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.model["num_epochs"]
    step = 1
    model.train()

    forward_signature = set(inspect.signature(model.forward).parameters)
    print("Training...")
    with tqdm(total=total_steps, unit="batch") as pbar:
        for epoch in range(config.model["num_epochs"]):
            for data in train_dataloader:
                pbar.set_description(f"Epoch {epoch}")
                lr = lr_scheduler.get_last_lr()[0]

                inputs = {
                    argument: value
                    for argument, value in data.items()
                    if argument in forward_signature
                }

                with accelerator.accumulate(model):
                    loss = _train_batch(
                        inputs=inputs,
                        data=data,
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        accelerator=accelerator,
                        config=config,
                    )

                if monitor:
                    wandb_logger.log_step(train_loss=loss, lr=lr)

                # Evaluate the model and save checkpoints
                if (step % config.model["log_interval"] == 0) or (step == total_steps):
                    print(f"Evaluation after {step} steps...")
                    # Evaluate the model
                    avg_val_loss = _train_evaluation(
                        model, val_dataloader, loss_fn=loss_fn, step=step
                    )
                    model.train()

                    if monitor:
                        wandb_logger.log_step(val_loss=avg_val_loss)

                    save_model_checkpoint(
                        accelerator.unwrap_model(model),
                        step,
                        config,
                    )

                step += 1
                pbar.update(1)
                wandb_logger.update_step(1)
            gc.collect()
            torch.cuda.empty_cache()

    if monitor:
        wandb.unwatch(watch_list)
    accelerator.free_memory()


def _train_batch(
    inputs,
    data,
    model,
    optimizer,
    loss_fn,
    lr_scheduler,
    config,
    accelerator: Accelerator,
):
    outputs = model(**inputs, return_dict=True)
    loss = loss_fn(outputs, data)
    accelerator.backward(loss)

    if config.get("gradient_clip", "none") != "none":
        accelerator.clip_grad_norm_(model.parameters(), config.model["gradient_clip"])
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.item()


def _train_evaluation(model, dataloader, loss_fn, step):
    model.eval()
    avg_tot_loss = 0

    forward_signature = set(inspect.signature(model.forward).parameters)
    with torch.no_grad():
        with tqdm(
            total=len(dataloader), unit="batch", desc=f"Eval step {step}"
        ) as pbar:
            for data in dataloader:
                inputs_kwargs = {
                    argument: value
                    for argument, value in data.items()
                    if argument in forward_signature
                }

                outputs = model(**inputs_kwargs, return_dict=True)
                loss = loss_fn(outputs, data)

                pbar.update(1)
                avg_tot_loss += loss.item()

            avg_tot_loss /= len(dataloader)
            pbar.set_postfix({"avg_val_loss": avg_tot_loss})

    return avg_tot_loss


def save_model_checkpoint(model, step, config):
    checkpoint_dir = config.model["checkpoint_dir"]
    filename = f"checkpoint_{config.seed}_{step}.pt"
    checkpoint_file = os.path.join(checkpoint_dir, filename)

    create_dirs_for_file(checkpoint_file)
    torch.save(model.state_dict(), checkpoint_file)
    wandb.save(checkpoint_file)
