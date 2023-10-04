import pandas as pd
import gc
import inspect
import os
import torch
import torch.nn as nn

from tqdm import tqdm

import wandb
from transformers import GPT2LMHeadModel, AutoTokenizer, get_scheduler
from accelerate import Accelerator

from .config import Config
from .utils import create_reproducible_dataloader, DummyScheduler
from .dataset import SBICDataCollator

CONFIG: Config = Config()

def make(config):
    # Make the model
    tokenizer = make_tokinzer(config)
    model = make_model(config, tokenizer)

    # Make the data
    train_data, val_data = get_data("train"), get_data("validation")
    train_dataloader = make_dataloader(train_data, tokenizer, config, split="train")
    val_dataloader = make_dataloader(val_data, tokenizer, config, split="validation")

    # Make the loss, the optimizer and the scheduler
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(
        optimizer, steps_per_epoch=len(train_dataloader), config=config
    )

    # # Make the evaluation metrics
    # metrics = make_metrics(tokenizer, config)

    return (
        model,
        tokenizer,
        train_data,
        val_data,
        train_dataloader,
        val_dataloader,
        # loss_fn,
        # optimizer,
        scheduler,
        # metrics,
    )

def make_tokinzer(config:dict):
    checkpoint = CONFIG.checkpoints.__dict__[config.checkpoint_name]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                              padding_side=config.padding_side,
                                              use_fast=True)
    tokenizer.add_special_tokens(CONFIG.train_params.special_tokens)
    print("List of all special token and its token_id:")
    print(" -", tokenizer.all_special_tokens)
    print(" -",tokenizer(tokenizer.all_special_tokens)["input_ids"])

    return tokenizer

def make_model(config:dict,
               tokenizer:AutoTokenizer):
    checkpoint = CONFIG.checkpoints.__dict__[config.checkpoint_name]
    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.sep_token_id = tokenizer.sep_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print("Model vocab resize:", model.config.vocab_size)
    print("Model eos token:", model.config.eos_token_id)
    print("Model pad token:", model.config.pad_token_id)
    print("Model sep token:", model.config.sep_token_id)

    return model

def get_data(split: str):
    path = os.path.join(CONFIG.dataset.raw_dir, f"{split}.pkl")
    data = pd.read_pickle(path).to_numpy()

    return data


def make_dataloader(dataset, model, tokenizer, config, split: str):
    data_collator = SBICDataCollator(tokenizer=tokenizer, model=model)
    dataloader = create_reproducible_dataloader(
        dataset,
        batch_size=config.val_batch_size
        if split != "train" and "val_batch_size" in config
        else config.batch_size,
        collate_fn=data_collator,
        num_workers=config.val_num_workers
        if split != "train" and "val_num_workers" in config
        else config.num_workers,
        pin_memory=True,
        shuffle=split == "train",
    )
    return dataloader


def make_optimizer(model, config):
   return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def make_scheduler(optimizer, steps_per_epoch, config):
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(config.warmup_fraction * total_steps)
    if config.get("scheduler", "none") != "none":
        return get_scheduler(
            config.scheduler,
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
    config
):
    watch_list = [model]

    accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.cpu)
    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    wandb.watch(watch_list, log="all", log_freq=config.log_interval)

    # Run training and track with wandb
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.num_epochs

    step = 0
    model.train()

    forward_signature = set(inspect.signature(model.forward).parameters)
    progress_bar = tqdm(range(total_steps))
    for epoch in range(config.num_epochs):
        for data in train_dataloader:

            lr = lr_scheduler.get_last_lr()[0]

            inputs = {
                argument: value
                for argument, value in data.items()
                if argument in forward_signature
            }

            loss = train_batch(
                inputs=inputs,
                data=data,
                step=step,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                config=config,
            )
            progress_bar.update(1)
            step += 1

            wandb.log(
                {
                    "train_loss": loss,
                    "lr": lr
                },
                step=step,
            )

            # if step % 10 == 0:
            #     print(f"Loss at epoch {epoch}: {loss} (iter n° {step+1})")
           
            # # Evaluate the model and save checkpoints
            # if (step % config.log_interval == 0) or (step == total_steps):
            #     # Evaluate the model
            #     # val_loss, val_inner_losses, val_metrics = train_evaluation(
            #     #     model,
            #     #     val_dataloader,
            #     #     loss_fn,
            #     #     # metrics=metrics,
            #     # )
            #     model.train()

            #     train_log(
            #         loss,
            #         # val_loss,
            #         # val_inner_losses,
            #         # val_metrics,
            #         lr=lr_scheduler.get_last_lr()[0],
            #         step=step,
            #     )
            #     avg_loss = AvgValue()
            #     avg_inner_losses = defaultdict(AvgValue)

            # if step % config.checkpoint_interval == 0:
            #     # Saving checkpoint
            #     save_model_checkpoint(
            #         model,
            #         optimizer,
            #         lr_scheduler,
            #         epoch,
            #         step,
            #         checkpoint_counter,
            #         config,
            #     )
            #     wandb.log(
            #         {
            #             "checkpoint_counter": checkpoint_counter,
            #         },
            #         step=step,
            #     )
            #     checkpoint_counter += 1

        gc.collect()
        torch.cuda.empty_cache()

    wandb.unwatch(watch_list)
    accelerator.free_memory()


def train_batch(
    inputs,
    data,
    step,
    model,
    optimizer,
    lr_scheduler,
    config,
    accelerator=None,
    device=None,
):
    assert (
        accelerator is not None or device is not None
    ), "One between accelerator and device must be set."

    if accelerator is None:
        data = {key: value.to(device) for key, value in data.items()}

    outputs = model(**inputs, return_dict=True)

    loss = outputs.loss
    if accelerator is not None:
        accelerator.backward(loss)
    else:
        loss.backward()

    if config.get("gradient_clip", "none") != "none":
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

    if step % config.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    lr_scheduler.step()

    return loss.item()


# def train_evaluation(
#     model,
#     dataloader,
#     compute_loss: ComputeLoss = None
#     # , metrics: Dict[str, Metric] = {}
# ) -> Tuple[AvgValue, Dict[str, AvgValue], Dict[str, AvgValue]]:
#     model.eval()
#     avg_loss = AvgValue()
#     avg_inner_losses = defaultdict(AvgValue)
#     avg_metrics = defaultdict(AvgValue)

#     forward_signature = set(inspect.signature(model.forward).parameters)
#     with torch.no_grad():
#         for data in dataloader:
#             inputs_kwargs = {
#                 argument: value
#                 for argument, value in data.items()
#                 if argument in forward_signature
#             }
#             n_samples = len(next(iter(data.values())))

#             outputs = model(**inputs_kwargs, return_dict=True)
#             if compute_loss is not None:
#                 loss, inner_losses = compute_loss(outputs, data)

#                 avg_loss.update(loss.item(), n_samples)
#                 for loss_name, loss_value in inner_losses.items():
#                     avg_inner_losses[loss_name].update(loss_value, n_samples)

#             # for metric_name, metric in metrics.items():
#             #     metric_value = metric(outputs, data)
#             #     avg_metrics[metric_name].update(metric_value, n_samples)

#     return avg_loss, avg_inner_losses, avg_metrics


# def train_log(
#     train_loss: AvgValue,
#     train_inner_losses: Dict[str, AvgValue],
#     val_loss: AvgValue,
#     val_inner_losses: Dict[str, AvgValue],
#     val_metrics: Dict[str, AvgValue],
#     lr,
#     step,
# ):
#     train_loss = train_loss.avg()
#     train_inner_losses = {
#         f"{loss_name}": loss_value.avg()
#         for loss_name, loss_value in train_inner_losses.items()
#     }

#     val_loss = val_loss.avg()
#     val_inner_losses = {
#         f"val_{loss_name}": loss_value.avg()
#         for loss_name, loss_value in val_inner_losses.items()
#     }

#     val_metrics = {
#         f"val_{metric_name}": metric_value.avg()
#         for metric_name, metric_value in val_metrics.items()
#     }

#     wandb.log(
#         {
#             "avg_train_loss": train_loss,
#             **train_inner_losses,
#             "val_loss": val_loss,
#             **val_inner_losses,
#             **val_metrics,
#             "lr": lr,
#         },
#         step=step,
#     )
#     print(
#         f"Iteration: {step:6}",
#         f"train loss: {train_loss:.4f}",
#         f"val loss: {val_loss:.4f}",
#         f"lr: {lr:.6f}",
#         sep="\t",
#     )


# # def save_model_checkpoint(
# #     model, optimizer, lr_scheduler, epoch, step, checkpoint_counter, config
# # ):
# #     checkpoint_file = os.path.join(
# #         CONFIG.models.checkpoints_dir(
# #             config.model_name, config.get("history_length", 0) > 0
# #         ),
# #         config.model_name,
# #         f"checkpoint_{checkpoint_counter}.pt",
# #     )
# #     create_dirs_for_file(checkpoint_file)

# #     save_checkpoint(
# #         model,
# #         optimizer,
# #         lr_scheduler,
# #         epoch,
# #         step,
# #         checkpoint_counter,
# #         checkpoint_path=checkpoint_file,
# #     )
# #     wandb.save(f"{config.model_name}_{checkpoint_counter}.pt")


# # def load_model_checkpoint(
# #     checkpoint_counter, config, model, optimizer=None, lr_scheduler=None
# # ):
# #     checkpoint_file = os.path.join(
# #         CONFIG.models.checkpoints_dir(
# #             config.model_name, config.get("history_length", 0) > 0
# #         ),
# #         config.model_name,
# #         f"checkpoint_{checkpoint_counter}.pt",
# #     )
# #     return load_checkpoint(
# #         checkpoint_file, model, optimizer=optimizer, scheduler=lr_scheduler
# #     )


# def evaluate(model, tokenizer, train_data: SBICDataset, val_data: SBICDataset, test_data: SBICDataset, config):
#     datasets = [("train", train_data), ("val", val_data), ("test", test_data)]
#     results = {}
#     for dataset_name, dataset in datasets:
#         print(f"eval  {dataset_name}")
#         outputs, metrics = eval_classification_token(model, tokenizer, dataset, config)
#         results[dataset_name] = (outputs, metrics)

#         gc.collect()
#         torch.cuda.empty_cache()

#     return results