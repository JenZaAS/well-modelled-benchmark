"""Train/test splits and gap masking. Always split by whole wells (no leakage)."""
from __future__ import annotations

import hashlib

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


def gap_seed(spec_seed: int, well_id: str, target: str, length_m: int) -> int:
    """Stable seed for one gap: sha256 of "<well_id>|<target>|<length_m>" (first 8 hex digits) XOR spec_seed.
    Not the builtin hash(), which changes between processes."""
    digest = hashlib.sha256(f"{well_id}|{target}|{int(length_m)}".encode()).hexdigest()
    return (int(digest[:8], 16) ^ int(spec_seed)) & 0xFFFFFFFF


def max_gap_step(depth: np.ndarray) -> float:
    """Largest depth step allowed inside a gap: max(1 m, 3 x median spacing); spacing defaults to 0.15 m."""
    d = np.diff(np.asarray(depth, float))
    d = d[np.isfinite(d) & (d > 0)]
    return max(1.0, 3.0 * (float(np.median(d)) if len(d) else 0.15))


def window_is_continuous(depth: np.ndarray, i0: int, i1: int, max_step: float) -> bool:
    """True when every consecutive depth step within depth[i0:i1] is <= max_step."""
    return i1 - i0 < 2 or bool((np.diff(np.asarray(depth, float)[i0:i1]) <= max_step).all())


def place_gap(depth: np.ndarray, target: np.ndarray, length_m: float, seed: int) -> tuple[int, int] | None:
    """Pick a contiguous gap of `length_m` metres lying fully inside finite target values.

    Returns (start_i, end_i), start inclusive, end exclusive. end_i is the first index at which
    depth[end_i - 1] - depth[start_i] >= length_m, so the gap is the shortest slice covering the length.
    Candidates must be depth-continuous (no step above max_gap_step). Eligible starts are chosen
    uniformly with numpy.random.default_rng(seed); None if there are none.
    """
    depth, target = np.asarray(depth, float), np.asarray(target, float)
    n = len(depth)
    if n == 0:
        return None
    ends = np.searchsorted(depth, depth + length_m - 1e-9, side="left") + 1
    starts = np.nonzero(ends <= n)[0]
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(target))])
    starts = starts[bad[ends[starts]] - bad[starts] == 0]
    # cumulative count of oversized steps: step j joins samples j and j+1
    jumps = np.concatenate([[0], np.cumsum(np.diff(depth) > max_gap_step(depth))])
    starts = starts[jumps[ends[starts] - 1] - jumps[starts] == 0]
    if not len(starts):
        return None
    i = int(np.random.default_rng(seed).choice(starts))
    return i, int(ends[i])
