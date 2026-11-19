"""Read canonical curves from a Well by name. Prediction code reads nothing else."""
from __future__ import annotations

import numpy as np


def curve(well, name: str) -> np.ndarray:
    """Float copy of a canonical curve, or all-NaN of the depth length when the well lacks it."""
    n = len(well.depth)
    x = well.curves.get(name)
    if x is None:
        return np.full(n, np.nan)
    a = np.array(x, dtype=float, copy=True).reshape(-1)
    if len(a) != n:
        raise ValueError(f"curve {name} has {len(a)} samples but the depth index has {n}")
    return a


def positive_target(well, target: str) -> np.ndarray:
    """Boolean mask of finite, strictly positive measured target values (all False if absent)."""
    y = curve(well, target)
    return np.isfinite(y) & (y > 0)


def has_positive_target(well, target: str) -> bool:
    return bool(positive_target(well, target).any())


def pooled(wells, *names: str) -> np.ndarray:
    """Rows (one column per name) from all wells that carry every named curve, finite in every column."""
    rows = [np.column_stack([curve(w, c) for c in names]) for w in wells if all(c in w.curves for c in names)]
    a = np.vstack(rows) if rows else np.empty((0, len(names)))
    return a[np.isfinite(a).all(axis=1)]
