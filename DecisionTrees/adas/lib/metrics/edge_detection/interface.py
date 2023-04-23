import numpy as np
from abc import ABC, abstractmethod


class BinaryEdgeMetric(ABC):
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        assert pred.shape == gt.shape, \
            f"Prediction and ground truth shape mismatched. Prediction: {pred.shape}. Gt: {gt.shape}"
        
        assert self._is_binary(pred), "Prediction is not a binary image"
        assert self._is_binary(gt), "Ground truth is not a binary image"

        return self._score(pred, gt)

    def _is_binary(self, edge_map: np.ndarray) -> bool:
        return set(np.unique(edge_map)).issubset({0, 1})

    @abstractmethod
    def _score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        ...
