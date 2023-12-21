import os
from typing import Optional

from omegaconf import OmegaConf


class Config:
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

    @staticmethod
    def to_dict(config: OmegaConf):
        return OmegaConf.to_container(config)
