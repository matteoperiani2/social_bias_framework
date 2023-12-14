import os
import torch

class PropertyDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def __setattr__(self, key, value):
        self[key] = value

class Config:
    
    class Dataset:
        
        data_dir: str = "data"
        
        raw_dir: str = os.path.join(data_dir, "raw")
        train_data_raw: str = os.path.join(raw_dir, "train.csv")
        val_data_raw: str = os.path.join(raw_dir, "validation.csv")
        test_data_raw: str = os.path.join(raw_dir, "test.csv")

        preproc_dir: str = os.path.join(data_dir, "preproc")
        train_data_preproc: str = os.path.join(preproc_dir, "train.pkl")
        val_data_preproc:str = os.path.join(preproc_dir, "validation.pkl")
        test_data_preproc: str = os.path.join(preproc_dir, "test.pkl")

        aggregated_dir: str = os.path.join(data_dir, "aggregated")
        train_data_agg: str = os.path.join(aggregated_dir, "train.pkl")
        val_data_agg: str = os.path.join(aggregated_dir, "validation.pkl")
        test_data_agg: str = os.path.join(aggregated_dir, "test.pkl")

        train_dir: str = os.path.join(data_dir, "train")
        train_data: str = os.path.join(train_dir, "train.pkl")
        val_data: str = os.path.join(train_dir, "validation.pkl")
        test_data: str = os.path.join(train_dir, "test.pkl")

    class ModelParameters:
        model_name:str = "distilgpt2"
        checkpoint_name:str = "distilgpt2"
        epochs:int = 3
        batch_size:int = 2
        lr:int = 1e-5
        padding_side:str = "left"
        device:str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        special_tokens = {
            "pad_token": "<|pad|>",
            "sep_token": "<|sep|>",
            "additional_special_tokens": [
                "<|offY|>",
                "<|offN|>",
                "<|intY|>",
                "<|intN|>",
                "<|sexY|>",
                "<|sexN|>",
                "<|grpY|>",
                "<|grpN|>",
                "<|ingrpY|>",
                "<|ingrpN|>",
            ],
        }

    class Utils:
        label_encoder = {
            0: {0: "<|offN|>", 1: "<|offY|>"},
            1: {0: "<|intN|>", 1: "<|intY|>"},
            2: {0: "<|sexN|>", 1: "<|sexY|>"},
            3: {0: "<|grpN|>", 1: "<|grpY|>", 2:"<|pad|>"},    
            4: {0: "<|ingrpN|>", 1: "<|ingrpY|>", 2:"<|pad|>"},
        }

    class Checkpoints:
        def __init__(
            self, gpt2="gpt2", distilgpt2="distilgpt2"
        ) -> None:
            self.gpt2 = gpt2
            self.distilgpt2 = distilgpt2

    class WandbConfig:
        """Specify the parameters of `wandb.init`"""

        project: str = "social_bias_framework"
        entity: str = "matteoperiani"
        
    dataset: Dataset = Dataset()
    model_params: ModelParameters = ModelParameters()
    checkpoints: Checkpoints = Checkpoints()
    utils: Utils = Utils()
    wandbConfig:WandbConfig = WandbConfig()

    hp = PropertyDict(
        seed=42,
        checkpoint_name="distilgpt2",
        model_name="distilgpt2",
        padding_side="left",
        batch_size=8,
        num_epochs=1,
        learning_rate=1e-5,
        scheduler="linear",
        warmup_fraction=0.2,
        accumulation_steps=1,
        gradient_clip = 1.0,
        mixed_precision="fp16",
        log_interval=200,
        eval_interval = 100,
        cpu=False
    )


CONFIG:Config = Config()