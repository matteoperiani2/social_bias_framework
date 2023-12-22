import gc
import os

import torch

import wandb
from src.config import Config
from src.models import model_helper_factory
from src.train_utils import fix_reproducibility, train

if __name__ == "__main__":
    wandb.login()
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    config = Config.load_config(model_name="gpt2")
    config = Config.to_dict(config)
    config["seed"] = 42

    with wandb.init(project=config["wandb"]["project"], config=config):
        config = wandb.config
        fix_reproducibility(config.seed)

        # get the train helper
        train_helper = model_helper_factory(config)

        # Make the tokenizer and the model
        tokenizer = train_helper.make_tokenizer()
        model = train_helper.make_model()

        # Make the data
        train_dataset = train_helper.get_data("train")
        val_dataset = train_helper.get_data("val")

        collator = train_helper.make_data_collator(tokenizer, model)

        train_dataloader = train_helper.make_dataloader(
            train_dataset, collate_fn=collator, split="train"
        )
        val_dataloader = train_helper.make_dataloader(
            val_dataset, collate_fn=collator, split="val"
        )

        # Make the loss, the optimizer and the scheduler
        optimizer = train_helper.make_optimizer(model)
        scheduler = train_helper.make_scheduler(
            optimizer, steps_per_epoch=len(train_dataloader)
        )

        loss_fn = train_helper.make_loss(tokenizer)

        train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            loss_fn,
            scheduler,
            config,
        )

    gc.collect()
    torch.cuda.empty_cache()
