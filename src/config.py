import os
from typing import Optional

from omegaconf import OmegaConf


class Config:
    @staticmethod
    def load_config(
        config_path="config", config_name="main", model_name: Optional[str] = None
    ):
        __SUB_CONFIGS = ("model", "data")

        config_file = os.path.join(config_path, config_name) + ".yaml"
        config = OmegaConf.load(config_file)

        def merge_with_subconfig(subconfig_name):
            file = os.path.join(config_path, subconfig_name, model_name) + ".yaml"
            subconfig = OmegaConf.load(file)
            config.merge_with({subconfig_name: subconfig})

        if model_name is not None:
            for subconfig_name in __SUB_CONFIGS:
                merge_with_subconfig(subconfig_name)

        return config

    @staticmethod
    def to_dict(config: OmegaConf):
        return OmegaConf.to_container(config)
