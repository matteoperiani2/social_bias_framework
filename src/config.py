import os
from typing import Optional

from omegaconf import OmegaConf


class Config:
    class Model:
        gpt2 = "gpt2"

    model = Model()

    @staticmethod
    def load_config(
        config_path="config", config_name="main", model_name: Optional[str] = None
    ):
        config_file = os.path.join(config_path, config_name) + ".yaml"
        config = OmegaConf.load(config_file)

        if model_name is not None:
            model_file = os.path.join(config_path, "model", model_name) + ".yaml"
            model_config = OmegaConf.load(model_file)
            config.merge_with({"model": model_config})

        return config
