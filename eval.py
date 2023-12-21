import os
import torch

from src.dataset import SBICDataset
from src.train_utils import *
from src.config import Config
from src.utils import fix_reproducibility
from src.evaluation import generate_predictions

from transformers import GenerationConfig

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

config = Config.load_config(model_name='gpt2')
config = Config.to_dict(config.model)
config['seed'] = 42

fix_reproducibility(config['seed'])

# Make the model
tokenizer = make_tokenizer(config)
model = make_model(config, tokenizer)

model.load_state_dict(torch.load('checkpoints/distilgpt2_full_init.pt'))

# Make the data
data = get_data("train", config, aggregated=False)[:1024]
dataset = SBICDataset(data, tokenizer, cls_token_map=config['cls_token_map'], is_training=False)
dataloader = make_dataloader(dataset, model, tokenizer, config, shuffle=True)

gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.max_new_tokens = 100
gen_cfg.do_sample = False
gen_cfg.num_beams = 1

generate_predictions(
    model, 
    tokenizer,
    dataloader,
    'val',
    gen_cfg,
    config
)

# torch.save(model.state_dict(), f"checkpoints/{config.checkpoint_name}_10832_{config.num_epochs}_sl2-half.pt")

gc.collect()
torch.cuda.empty_cache()