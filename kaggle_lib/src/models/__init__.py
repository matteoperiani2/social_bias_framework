from .bart import BartHelper, BartTokenizationHelper
from .gpt2 import GPT2Helper, GPT2TokenizationHelper
from .helper import ModelHelper
from .tokenization import TokenizationHelper


def model_helper_factory(config: dict) -> ModelHelper:
    if config["model"]["name"] == "gpt2":
        return GPT2Helper(config)
    elif config["model"]["name"] == "bart":
        return BartHelper(config)
    else:
        raise ValueError("Invalid model name. Possible values are [gpt2, bart]")


def tokenization_helper_factory(config: dict) -> TokenizationHelper:
    if config["model"]["name"] == "gpt2":
        return GPT2TokenizationHelper(config)
    elif config["model"]["name"] == "bart":
        return BartTokenizationHelper(config)
    else:
        raise ValueError("Invalid model name. Possible values are [gpt2, bart]")


__all__ = [
    model_helper_factory,
    tokenization_helper_factory,
    ModelHelper,
    TokenizationHelper,
]
