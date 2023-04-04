from .better_abc import ABCMeta, abstract_attribute
from abc import abstractmethod
from typing import Dict, Any
from .description import ParametersDescription
import numpy as np


class Environment(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.reset()

    @abstract_attribute
    def parameters_description(self) -> ParametersDescription:
        ...

    @abstractmethod
    def score(self, parameters: Dict[str, Any]) -> float:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def current_state(self) -> np.ndarray:
        ...
