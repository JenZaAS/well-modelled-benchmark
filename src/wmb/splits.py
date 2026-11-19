"""Train/test splits and gap masking. Always split by whole wells (no leakage)."""
from __future__ import annotations

import numpy as np

from wmb.io import Well


def well_split(wells: list[Well], test_fraction: float = 0.2, seed: int = 0) -> tuple[list[Well], list[Well]]:
    """Random split by well name (deterministic for a seed)."""
    names = sorted({w.name for w in wells})
    rng = np.random.default_rng(seed)
    n_test = max(1, round(len(names) * test_fraction))
    test = set(rng.choice(names, n_test, replace=False))
    return [w for w in wells if w.name not in test], [w for w in wells if w.name in test]


def official_split(wells: list[Well]) -> tuple[list[Well], list[Well]]:
    """Use the dataset's own split field: 'train' trains, everything else tests (FORCE 2020: leaderboard + hidden)."""
    return [w for w in wells if w.split in (None, "train")], [w for w in wells if w.split not in (None, "train")]


def gap_mask(depth: np.ndarray, n_gaps: int = 3, min_m: float = 20.0, max_m: float = 50.0, seed: int = 0) -> np.ndarray:
    """Boolean mask, True inside random artificial gaps of min_m..max_m metres."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(depth.shape, bool)
    if depth.size == 0:
        return mask
    lo, hi = float(np.nanmin(depth)), float(np.nanmax(depth))
    for _ in range(n_gaps):
        length = rng.uniform(min_m, max_m)
        if hi - lo <= length:
            break
        top = rng.uniform(lo, hi - length)
        mask |= (depth >= top) & (depth < top + length)
    return mask
