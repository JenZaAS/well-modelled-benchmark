"""The one interface every model implements so the benchmark can run it.

    class MyModel(BaseContender):
        name = "blue_xgboost"
        workstream = "blue_boosted_trees"
        def fit(self, train_wells, target): ...
        def predict(self, well, target) -> Prediction: ...

`well` passed to predict() never contains the target curve.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from wmb.io import Well


@dataclass
class Prediction:
    mean: np.ndarray                         # same length as well.depth; NaN where the model can't predict
    std: np.ndarray | None = None            # 1-sigma uncertainty (optional but scored)
    quantiles: dict[float, np.ndarray] | None = None   # optional, e.g. {0.05: ..., 0.95: ...}


@runtime_checkable
class Contender(Protocol):
    name: str
    workstream: str
    inputs: list[str]                        # curves the model uses as inputs

    def fit(self, train_wells: list[Well], target: str) -> None: ...

    def predict(self, well: Well, target: str) -> Prediction: ...


class BaseContender(ABC):
    name = "unnamed"
    workstream = "unknown"
    inputs: list[str] = []
    targets: list[str] | None = None         # None = any curve

    def supports(self, target: str) -> bool:
        return self.targets is None or target in self.targets

    @abstractmethod
    def fit(self, train_wells: list[Well], target: str) -> None: ...

    @abstractmethod
    def predict(self, well: Well, target: str) -> Prediction: ...
