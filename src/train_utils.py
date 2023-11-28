import numpy as np
import pandas as pd
import gc
import inspect
import os
import torch
import torch.nn as nn
import warnings 

from collections import deque
from tqdm import tqdm

import wandb
import transformers
from transformers import GPT2LMHeadModel, AutoTokenizer, get_scheduler
from transformers.generation import GenerationConfig
from accelerate import Accelerator

from .config import Config
from .utils import create_reproducible_dataloader, DummyScheduler
from .dataset import SBICDataCollator
from .evaluation import evaluate_predictions, eval_classifications
from .losses import CEwithStructureImportance

warnings.filterwarnings('ignore') 

CONFIG: Config = Config()


def make_tokinzer(config:dict):
    # checkpoint = CONFIG.checkpoints.__dict__[config.checkpoint_name]
    tokenizer = AutoTokenizer.from_pretrained(config.get("checkpoint_name"),
                                              padding_side=config.padding_side,
                                              use_fast=True)
    tokenizer.add_special_tokens(CONFIG.model_params.special_tokens)
    print("List of all special token and its token_id:")
    print(" -", tokenizer.all_special_tokens)
    print(" -",tokenizer(tokenizer.all_special_tokens)["input_ids"])

    return tokenizer


def make_model(config:dict,
               tokenizer:AutoTokenizer):
    model = GPT2LMHeadModel.from_pretrained(config.get("checkpoint_name"))
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
    path = os.path.join(CONFIG.dataset.preproc_dir, f"{split}.pkl")
    data = pd.read_pickle(path).to_numpy()
    return data


def make_dataloader(dataset, model, tokenizer, config, split: str):
    data_collator = SBICDataCollator(tokenizer=tokenizer, model=model)
    dataloader = create_reproducible_dataloader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        num_workers=0,
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
    config,
    monitor=True,
):
    model.train()
    if monitor:
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

    if monitor:
        wandb.watch(watch_list, log="all", log_freq=config.log_interval)

    loss_fn = CEwithStructureImportance(alpha=config.loss_alpha)

    forward_signature = set(inspect.signature(model.forward).parameters)
    step = 0
    max_iters = config.num_epochs * len(train_dataloader)
    print("Training...")
    with tqdm(total=max_iters, unit="batch") as pbar:
        for epoch in range(config.num_epochs):
            epoch_loss = 0
            for data in train_dataloader:
                pbar.set_description(F"Epoch {epoch}")
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
                    loss_fn=loss_fn,
                    lr_scheduler=lr_scheduler,
                    accelerator=accelerator,
                    config=config
                )

                epoch_loss += loss

                step += 1
                pbar.update(1)

                if monitor:
                    wandb.log({"train_loss": loss,
                               "lr": lr },
                              step=step)

                if (step % config.eval_interval == 0) or max_iters == step:
                    print(f"Evaluation after the {step} steps...")
                    # Evaluate the model
                    _, avg_val_loss = train_evaluation(
                        model,
                        val_dataloader,
                        loss_fn=loss_fn
                    )
                    model.train()
                    if monitor:
                        wandb.log(
                            {
                                "val_loss": avg_val_loss,
                                "lr": lr
                            },
                            step=step,
                        )
                    print(f"Epoch {epoch}:\n - loss={loss:.4f} - avg_val_loss={avg_val_loss:.4f}")

                pbar.set_postfix(loss=loss)

    if monitor:
        wandb.unwatch(watch_list)
    
    gc.collect()
    accelerator.free_memory()
    torch.cuda.empty_cache()

def train_batch(
    inputs,
    data,
    step,
    model,
    optimizer,
    loss_fn,
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
    
    loss = loss_fn(outputs.logits, data["labels"])

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


def train_evaluation(
    model,
    dataloader,
    loss_fn
):
    model.eval()

    forward_signature = set(inspect.signature(model.forward).parameters)
    avg_loss = 0
    losses = deque(maxlen=len(dataloader))
    with torch.no_grad():
        for data in tqdm(dataloader, leave=False, total=len(dataloader)):
            inputs_kwargs = {
                argument: value
                for argument, value in data.items()
                if argument in forward_signature
            }

            outputs = model(**inputs_kwargs, return_dict=True)
            loss = loss_fn(outputs.logits, data["labels"])
            losses.append(loss)
            avg_loss += loss

    return losses, avg_loss/len(dataloader)


def evaluate(
    model:transformers.AutoModel, 
    tokenizer:transformers.AutoTokenizer, 
    dataloader:torch.utils.data.DataLoader, 
    config:dict
):  
    model.eval()
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = 100
    gen_cfg.max_length = 512
    
    class_f1s = deque(maxlen=len(dataloader))
    minor_rouge = deque(maxlen=len(dataloader))
    strtp_rouge = deque(maxlen=len(dataloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data in tqdm(dataloader, leave=False, total=len(dataloader)):
            input_ids = torch.as_tensor(data.pop("input_ids"), device=device)
            generate_out = model.generate(inputs = input_ids,
                                          generation_config=gen_cfg)
            generate_out = generate_out.cpu().numpy()
            
            # remove from the output the input prompt
            preds = [gen[np.where(gen == tokenizer.sep_token_id)[0][0]+1:] for gen in generate_out]
            
            score = evaluate_predictions(tokenizer, preds, data)
            class_f1s.append(score["class_f1"])
            minor_rouge.extend(score["rouge_minor"])
            strtp_rouge.extend(score["rouge_strtp"])

    clasification_f1_score = np.mean(class_f1s, axis=0)
    minority_rouge_f1_score = np.mean(minor_rouge, axis=0)
    stereotype_rouge_f1_score = np.mean(strtp_rouge, axis=0)

    return {
        "out_tokens": generate_out,
        "clasification_f1_score": clasification_f1_score,
        "minority_rouge_f1_score": minority_rouge_f1_score,
        "stereotype_rouge_f1_score": stereotype_rouge_f1_score
    }


def evaluate_classification(
    model:transformers.AutoModel, 
    tokenizer:transformers.AutoTokenizer, 
    dataloader:torch.utils.data.DataLoader, 
    config:dict
):  
    model.eval()
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = 100
    gen_cfg.max_length = 512
    
    labels = None
    predictions = None
    batchs_f1 = deque(maxlen=len(dataloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data in tqdm(dataloader, leave=False, total=len(dataloader)):
            input_ids = torch.as_tensor(data.pop("input_ids"), device=device)
            generate_out = model.generate(inputs = input_ids,
                                          generation_config=gen_cfg)
            generate_out = generate_out.cpu().numpy()
            
            # remove from the output the input prompt
            preds = [gen[np.where(gen == tokenizer.sep_token_id)[0][0]+1:] for gen in generate_out]
            
            res = eval_classifications(tokenizer, preds, data)
            if labels is None:
                labels = res["labels"]
            else:
                labels = np.concatenate((labels, res["labels"]), axis=1)
            if predictions is None:
                predictions = res["predictions"]
            else:
                predictions = np.concatenate((predictions, res["predictions"]), axis=1)
            batchs_f1.append(res["f1"])

    f1_score = np.mean(batchs_f1, axis=0)

    return {
        'labels': labels,
        "predictions": predictions,
        "f1": f1_score
    }