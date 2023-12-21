import os
import torch

from src.dataset import SBICDataset
from src.train_utils import *
from src.config import Config
from src.utils import fix_reproducibility

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

config = Config.load_config(model_name='gpt2')
config = Config.to_dict(config.model)
config['seed'] = 42

fix_reproducibility(config['seed'])

# get the train helper
train_helper = get_model_helper(config)

# Make the model and the tokenizer
model, tokenizer = train_helper.make_model_and_tokenizer()

# Make the data
train_dataset = train_helper.get_data("train").select(range(10832))
val_dataset = train_helper.get_data("val").select(range(2048))

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
    monitor=False
)

gc.collect()
torch.cuda.empty_cache()