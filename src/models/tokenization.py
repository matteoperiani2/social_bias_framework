from abc import ABC, abstractmethod
from typing import Literal, Union


class TokenizationHelper(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def tokenize(
        self,
        example: dict,
        task: Union[Literal["train"], Literal["inference"]],
        ignore_labels=False,
    ):
        pass
