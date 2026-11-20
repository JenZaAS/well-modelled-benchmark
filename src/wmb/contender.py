"""The one interface every model implements so the benchmark can run it.

    class MyModel(BaseContender):
        name = "blue_xgboost"
        workstream = "blue_boosted_trees"
        def fit(self, train_wells, target): ...
        def predict(self, well, target) -> Prediction: ...

Optionally override `predict_task(task)` / `predict_batch(tasks)` (see wmb.contract.Task);
the defaults call `predict(task.to_well(), task.target)`. For missing_curve the well omits the target; for gap it includes `target_observed` (NaN on the hidden interval).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from wmb.io import Well

if TYPE_CHECKING:
    from wmb.contract import Task


@dataclass
class Prediction:
    mean: np.ndarray | None = None           # same length as well.depth; NaN where the model can't predict
    std: np.ndarray | None = None            # 1-sigma uncertainty (optional but scored)
    quantiles: dict[float, np.ndarray] | None = None   # optional, e.g. {0.05: ..., 0.95: ...}
    p50: np.ndarray | None = None            # alias of mean
    p05: np.ndarray | None = None            # lower end of the 90% interval
    p95: np.ndarray | None = None            # upper end of the 90% interval
    bin_edges: np.ndarray | None = None      # optional: K-1 increasing internal edges of K ordered value bins
    bin_probs: np.ndarray | None = None      # optional: [len(depth), K] bin probabilities; NaN rows = no forecast

    def __post_init__(self):
        if self.mean is None:
            self.mean = self.p50
        if self.p50 is None:
            self.p50 = self.mean


@runtime_checkable
class Contender(Protocol):
    name: str
    workstream: str
    inputs: list[str]                        # curves the model uses as inputs (required_inputs if set)

    def fit(self, train_wells: list[Well], target: str) -> None: ...

    def predict(self, well: Well, target: str) -> Prediction: ...


class BaseContender(ABC):
    name = "unnamed"
    workstream = "unknown"
    inputs: list[str] = []
    required_inputs: list[str] = []          # empty -> `inputs` is used
    optional_inputs: list[str] = []
    targets: list[str] | None = None         # None = any curve
    licence_name: str | None = None          # e.g. MIT, Apache-2.0; see the workstream LICENSE.md
    licence_url: str | None = None
    licence_restrictions: str | None = None  # one-line summary, or 'none'

    @property
    def team(self) -> str:
        return self.workstream

    def declared_required_inputs(self) -> list[str]:
        return list(self.required_inputs or self.inputs)

    def supports(self, target: str) -> bool:
        return self.targets is None or target in self.targets

    @abstractmethod
    def fit(self, train_wells: list[Well], target: str) -> None: ...

    def predict(self, well: Well, target: str) -> Prediction:
        raise NotImplementedError(f"{type(self).__name__} must implement predict(well, target) or predict_task(task)")

    def predict_task(self, task: "Task") -> Prediction:
        return self.predict(task.to_well(), task.target)

    def predict_batch(self, tasks: list["Task"]) -> list[Prediction]:
        return [self.predict_task(t) for t in tasks]
