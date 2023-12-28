import os
from abc import ABC, abstractmethod
from typing import Any

import datasets
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..train_utils import DummyLRScheduler, create_reproducible_dataloader


class ModelHelper(ABC):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model_config = config["model"]

    def make_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(self.model_config["checkpoint_name"])
        return tokenizer

    @abstractmethod
    def make_model(self, tokenizer) -> torch.nn.Module:
        raise NotImplementedError()

    def get_data(self, split) -> datasets.Dataset:
        path = os.path.join(self.config["data"]["train"], split)
        data = datasets.load_from_disk(path)
        return data

    def make_data_collator(self, tokenizer, model):
        return None

    def make_dataloader(self, data, split: str, collate_fn=None) -> DataLoader:
        batch_size = (
            self.model_config["val_batch_size"]
            if split != "train" and "val_batch_size" in self.model_config
            else self.model_config["batch_size"]
        )

        dataloader = create_reproducible_dataloader(
            data,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=self.model_config.get("num_workers", 0),
            pin_memory=True,
            shuffle=(split == "train"),
        )
        return dataloader

    def make_optimizer(self, model) -> torch.optim.Optimizer:
        optimizer_cls = getattr(
            torch.optim, self.model_config.get("optimizer_name", "AdamW")
        )
        parameters = [{"params": model.parameters()}]

        return optimizer_cls(
            parameters,
            lr=self.model_config["learning_rate"],
            **self.model_config.get("optimizer_args", {}),
        )

    def make_scheduler(self, optimizer, steps_per_epoch) -> Any:
        total_steps = steps_per_epoch * self.model_config["num_epochs"]
        warmup_steps = int(self.model_config["warmup_fraction"] * total_steps)
        if self.model_config.get("scheduler", "none") != "none":
            return transformers.get_scheduler(
                self.model_config["scheduler"],
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        return DummyLRScheduler(optimizer=optimizer)

    def make_loss(self, tokenizer):
        raise NotImplementedError()
