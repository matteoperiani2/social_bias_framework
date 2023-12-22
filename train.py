import os, wandb, torch
from src.train_utils import *
from src.config import Config

wandb.login()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

config = Config.load_config(model_name='gpt2')
config = Config.to_dict(config)
config['seed'] = 42

with wandb.init(project=config['wandb']['project'], config=config):

    config = wandb.config
    fix_reproducibility(config.seed)

    # get the train helper
    train_helper = get_model_helper(config)

    # Make the model and the tokenizer
    model, tokenizer = train_helper.make_model_and_tokenizer()

    # Make the data
    train_dataset = train_helper.get_data("train")
    val_dataset = train_helper.get_data("val")

    collator = train_helper.make_collator()

    train_dataloader = make_dataloader(train_dataset, collator, config)
    val_dataloader = make_dataloader(val_dataset, collator, config, shuffle=False)

    # Make the loss, the optimizer and the scheduler
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(
        optimizer, steps_per_epoch=len(train_dataloader), config=config
    )

    loss_fn = make_loss(config)

    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        scheduler,
        config, 
    )

    torch.save(model.state_dict(), f'{config.model.name}_{config.seed}.pt')

gc.collect()
torch.cuda.empty_cache()




