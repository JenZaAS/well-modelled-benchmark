"""The fixed input/output contract of the benchmark ("the station").

A `Task` is what a model receives; `validate_prediction` checks what it returns.
Full description: docs/benchmark.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wmb.contender import Prediction
from wmb.io import Well
from wmb.scoring import gaussian_interval

TARGET_CURVES = ("RHOB", "DTC", "DTS")
SCENARIO_MISSING = "missing_curve"
SCENARIO_GAP = "gap"
GAP_LENGTHS_M = (10.0, 50.0, 100.0)
INPUT_CURVES = ("GR", "RDEP", "RMED", "CALI", "NPHI", "PEF", "RHOB", "DTC", "DTS")


@dataclass
class Task:
    well_id: str
    depth: np.ndarray                                   # metres MD, increasing
    inputs: pd.DataFrame                                # DEPTH + canonical curves; NaN allowed; no target column
    target: str
    predict_mask: np.ndarray                            # True = the model must predict
    scenario: str                                       # missing_curve | gap
    gap_length_m: float | None = None
    target_observed: np.ndarray | None = None           # gap only: measured target, NaN on predict_mask
    dataset: str = ""
    split: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def scenario_key(self) -> str:
        return scenario_key(self.scenario, self.gap_length_m)

    def to_well(self) -> Well:
        """Well view of the task: the target is omitted (missing_curve) or holds the observed values (gap)."""
        curves = {c: self.inputs[c].to_numpy(float) for c in self.inputs.columns if c != "DEPTH" and c != self.target}
        if self.target_observed is not None:
            curves[self.target] = np.asarray(self.target_observed, float).copy()
        return Well(self.well_id, np.asarray(self.depth, float), curves, self.dataset, self.split, dict(self.meta))


def scenario_key(scenario: str, gap_length_m: float | None = None) -> str:
    return f"gap_{int(gap_length_m)}" if scenario == SCENARIO_GAP else scenario


def task_input_columns(target: str) -> list[str]:
    return ["DEPTH"] + [c for c in INPUT_CURVES if c != target]


def _quantile(q: dict | None, level: float):
    for k, v in (q or {}).items():
        if abs(float(k) - level) < 1e-9:
            return v
    return None


def _as_array(name: str, x, n: int) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if len(a) != n:
        raise ValueError(f"{name} length {len(a)} != depth length {n}")
    return a


def validate_prediction(task: Task, pred: Prediction) -> Prediction:
    """Check a Prediction against its Task and return a normalised copy.

    mean and p50 are aliases. Missing p05/p50/p95 are taken from `quantiles`; if still missing and
    `std` is given, p05/p95 are filled with the Gaussian rule p50 +- 1.6448536269514722 * std (the one
    place where the 90% interval is derived; wmb.scoring.gaussian_interval). Raises ValueError otherwise.
    """
    n = len(task.depth)
    if not isinstance(pred, Prediction):
        raise ValueError(f"prediction must be a wmb.contender.Prediction, got {type(pred).__name__}")
    p50 = pred.p50 if pred.p50 is not None else pred.mean
    if p50 is None:
        p50 = _quantile(pred.quantiles, 0.5)
    if p50 is None:
        raise ValueError("p50 (or mean) is required")
    p50 = _as_array("p50", p50, n)
    if pred.mean is not None:
        _as_array("mean", pred.mean, n)
    p05, p95 = pred.p05, pred.p95
    p05 = _quantile(pred.quantiles, 0.05) if p05 is None else p05
    p95 = _quantile(pred.quantiles, 0.95) if p95 is None else p95
    std = None if pred.std is None else _as_array("std", pred.std, n)
    p05 = None if p05 is None else _as_array("p05", p05, n)
    p95 = None if p95 is None else _as_array("p95", p95, n)
    if (p05 is None) != (p95 is None):
        raise ValueError("p05 and p95 must be given together")
    if std is not None:
        bad = int((np.isfinite(std) & (std < 0)).sum())
        if bad:
            raise ValueError(f"std < 0 at {bad} samples")
    if p05 is not None:
        for a, b, label in ((p05, p50, "p05 > p50"), (p50, p95, "p50 > p95")):
            both = np.isfinite(a) & np.isfinite(b)
            bad = int((both & (a > b)).sum())
            if bad:
                raise ValueError(f"{label} at {bad} samples")
    elif std is not None:
        p05, p95 = gaussian_interval(p50, std, 0.90)
    quantiles = None
    if pred.quantiles is not None:
        quantiles = {float(k): _as_array(f"quantile {k}", v, n) for k, v in pred.quantiles.items()}
        levels = sorted(quantiles)
        for lo_k, hi_k in zip(levels, levels[1:]):
            both = np.isfinite(quantiles[lo_k]) & np.isfinite(quantiles[hi_k])
            bad = int((both & (quantiles[hi_k] < quantiles[lo_k])).sum())
            if bad:
                raise ValueError(f"quantile {hi_k} < quantile {lo_k} at {bad} samples")
    snap = lambda a: None if a is None else np.array(a, dtype=float, copy=True)  # noqa: E731
    p50c = snap(p50)
    return Prediction(mean=p50c, std=snap(std), quantiles=None if quantiles is None else {k: snap(v) for k, v in quantiles.items()},
                      p50=p50c, p05=snap(p05), p95=snap(p95))
