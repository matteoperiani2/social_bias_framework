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

# Make the model
tokenizer = make_tokenizer(config)
model = make_model(config, tokenizer)

# Make the data
train_data = get_data("train", config)[:10832]
val_data = get_data("validation",config)[:1024]

train_dataset = SBICDataset(train_data, tokenizer, cls_token_map=config['cls_token_map'])
val_dataset = SBICDataset(val_data, tokenizer, cls_token_map=config['cls_token_map'])

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
    config, 
    monitor=False
)

# torch.save(model.state_dict(), f"checkpoints/{config.checkpoint_name}_10832_{config.num_epochs}_sl2-half.pt")

gc.collect()
torch.cuda.empty_cache()