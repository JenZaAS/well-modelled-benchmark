"""Well split, gap indexing and flank geometry for the 50 m density-gap experiment (NumPy only).

Conventions (indices are into one well's native depth axis, metres MD, increasing):
    gap        [a, b)      the hidden interval; its RHOB is removed before anything else reads the curve
    segment    [s, e)      the continuous acquisition segment holding the gap; a step larger than
                           wmb.splits.max_gap_step(depth) starts a new segment, and no context crosses it
    flank                  measured target samples next to the gap inside the same segment

The station's own placer (`wmb.splits.place_gap` with `gap_seed`) chooses every gap, so A-D gaps obey the
same metre-based length, continuity and finite-target rules as the shipped `rhob_gap50` test.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from wmb import splits
from wmb.contract import SCENARIO_GAP, Task, task_input_columns

GROUPS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------------------------- well split
def split_key(well_id: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{well_id}".encode()).hexdigest()


def split_wells(names, sizes: dict, seed: int) -> dict[str, list[str]]:
    """A/B/C/D well IDs: sort by sha256("<seed>|<well id>"), then cut in group order with the given sizes.

    The sizes must add up to the number of distinct names; the groups are disjoint by construction and checked.
    """
    ids = sorted(set(names), key=lambda n: (split_key(n, seed), n))
    if sum(sizes.values()) != len(ids):
        raise ValueError(f"split sizes {sizes} add up to {sum(sizes.values())}, but there are {len(ids)} wells")
    out, i = {}, 0
    for g in GROUPS:
        out[g] = sorted(ids[i:i + int(sizes.get(g, 0))])
        i += int(sizes.get(g, 0))
    assert_disjoint(out)
    return out


def local_split(names) -> dict[str, list[str]]:
    """Contract-check split for a handful of wells (synthetic `check_model` runs): every group = every well.

    In-sample by design; it only exercises the code path and is never used for a reported number.
    """
    ids = sorted(set(names))
    return {g: list(ids) for g in GROUPS}


def assert_disjoint(groups: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for g, ids in groups.items():
        for n in ids:
            if n in seen:
                raise ValueError(f"well {n} is in both split {seen[n]} and {g}")
            seen[n] = g


def split_hash(groups: dict[str, list[str]]) -> str:
    return hashlib.sha256(json.dumps({g: sorted(v) for g, v in sorted(groups.items())}).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- geometry
def segment_ids(depth) -> np.ndarray:
    """Segment number per sample; a step above max_gap_step(depth) starts a new segment."""
    depth = np.asarray(depth, float)
    if len(depth) < 2:
        return np.zeros(len(depth), int)
    return np.r_[0, np.cumsum(np.diff(depth) > splits.max_gap_step(depth))]


def segment_of(seg: np.ndarray, i: int) -> tuple[int, int]:
    """[s, e) of the segment holding sample i."""
    same = np.flatnonzero(seg == seg[i])
    return int(same[0]), int(same[-1]) + 1


def mask_pieces(depth, mask) -> list[tuple[int, int, int, int]]:
    """(a, b, s, e) for each contiguous True run of `mask`, split where it crosses a segment boundary."""
    depth = np.asarray(depth, float)
    m = np.asarray(mask, bool)
    seg = segment_ids(depth)
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    out = []
    for i0, i1 in zip(edges[::2], edges[1::2]):
        for sid in np.unique(seg[i0:i1]):
            rows = np.arange(i0, i1)[seg[i0:i1] == sid]
            s, e = segment_of(seg, int(rows[0]))
            out.append((int(rows[0]), int(rows[-1]) + 1, s, e))
    return out


def flank_run(obs, a: int, b: int, s: int, e: int, max_len: int, side: str) -> np.ndarray:
    """Indices (increasing depth) of the contiguous measured run touching the gap on one side, at most max_len.

    side "above" reads obs[a-1], obs[a-2], ... ; "below" reads obs[b], obs[b+1], ... ; both stop at the first
    NaN or at the segment edge. An empty array means the side has no adjacent measured context.
    """
    obs = np.asarray(obs, float)
    if side == "above":
        lo = max(s, a - max_len)
        cand = np.arange(a - 1, lo - 1, -1)
    elif side == "below":
        hi = min(e, b + max_len)
        cand = np.arange(b, hi)
    else:
        raise ValueError(side)
    if not len(cand):
        return cand.astype(int)
    finite = np.isfinite(obs[cand])
    n = int(np.logical_and.accumulate(finite).sum())
    run = cand[:n]
    return np.sort(run).astype(int)


def linear_fill(depth, obs, a: int, b: int, s: int, e: int) -> np.ndarray | None:
    """Straight line between the nearest measured samples above and below inside [s, e); None if a side has none.

    Same rule as `validate_force.linear_gap_reference` (the shared interpolation row), for one piece.
    """
    depth, obs = np.asarray(depth, float), np.asarray(obs, float)
    above = np.flatnonzero(np.isfinite(obs[s:a])) + s
    below = np.flatnonzero(np.isfinite(obs[b:e])) + b
    if not len(above) or not len(below):
        return None
    j, k = int(above[-1]), int(below[0])
    return obs[j] + (obs[k] - obs[j]) * (depth[a:b] - depth[j]) / (depth[k] - depth[j])


# ---------------------------------------------------------------------------------------------- gap index
@dataclass(frozen=True)
class Gap:
    well: str
    group: str
    k: int
    start: int          # inclusive index
    end: int            # exclusive index
    start_md: float
    end_md: float       # depth of the last hidden sample
    seg_start: int
    seg_end: int
    spacing_m: float    # median depth step inside the gap

    @property
    def n(self) -> int:
        return self.end - self.start

    @property
    def gap_id(self) -> str:
        return f"{self.well}|{self.start}:{self.end}"

    def as_dict(self) -> dict:
        return {**asdict(self), "n": self.n, "gap_id": self.gap_id}


def _make_gap(well: str, group: str, k: int, depth, a: int, b: int) -> Gap:
    seg = segment_ids(depth)
    s, e = segment_of(seg, a)
    d = np.diff(np.asarray(depth, float)[a:b])
    return Gap(well, group, k, int(a), int(b), round(float(depth[a]), 4), round(float(depth[b - 1]), 4), s, e,
               round(float(np.median(d)) if len(d) else float("nan"), 5))


def index_gaps(well: str, depth, y, *, group: str, n_max: int, overlap: bool, seed: int, length_m: float,
               target: str = "RHOB", attempts_per_gap: int = 16) -> list[Gap]:
    """Up to n_max gaps of `length_m` metres in one well, placed by `wmb.splits.place_gap`.

    overlap=True (fitting wells): distinct start indices, overlaps allowed.
    overlap=False (selection / decision / calibration wells): non-overlapping, one sample kept between gaps.
    Every placement seed is `gap_seed(seed, "<well>#<group>#<k>#<attempt>", target, length_m)`: deterministic,
    and independent of Python's hash. Fewer gaps are returned when no more fit.
    """
    depth, y = np.asarray(depth, float), np.asarray(y, float)
    remaining = y.copy()
    starts: set[int] = set()
    gaps: list[Gap] = []
    k = 0
    misses = 0
    while len(gaps) < n_max and misses < attempts_per_gap:
        label = f"{well}#{group}#{k}#{misses}"
        g = splits.place_gap(depth, remaining if not overlap else y, length_m,
                             splits.gap_seed(seed, label, target, length_m))
        if g is None or g[0] in starts:
            misses += 1
            if g is None and not overlap:
                break       # nothing placeable is left
            continue
        a, b = g
        starts.add(a)
        gaps.append(_make_gap(well, group, len(gaps), depth, a, b))
        if not overlap:
            remaining[max(0, a - 1):b + 1] = np.nan
        k += 1
        misses = 0
    return sorted(gaps, key=lambda g: g.start)


def gaps_hash(gaps: list[Gap]) -> str:
    rows = sorted((g.well, g.group, g.start, g.end) for g in gaps)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def wells_data_hash(wells, target: str = "RHOB") -> str:
    """Hash of the depth and target arrays of the given wells (identity of the data the artifacts saw)."""
    h = hashlib.sha256()
    for w in sorted(wells, key=lambda w: w.name):
        h.update(w.name.encode())
        h.update(np.ascontiguousarray(np.asarray(w.depth, float)).tobytes())
        y = w.curves.get(target)
        if y is not None:
            h.update(np.nan_to_num(np.ascontiguousarray(np.asarray(y, float)), nan=-1e9).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- tasks
def inputs_frame(well, target: str) -> pd.DataFrame:
    n = len(well.depth)
    cols = {}
    for c in task_input_columns(target):
        if c == "DEPTH":
            cols[c] = np.asarray(well.depth, float)
        else:
            cols[c] = np.asarray(well.curves[c], float) if c in well.curves else np.full(n, np.nan)
    return pd.DataFrame(cols)


def gap_task(well, gap: Gap, target: str = "RHOB", frame: pd.DataFrame | None = None, test: str = "",
             length_m: float = 50.0) -> Task:
    """A station Task for one development gap: the hidden RHOB is removed here, before anything reads it."""
    depth = np.asarray(well.depth, float)
    mask = np.zeros(len(depth), bool)
    mask[gap.start:gap.end] = True
    observed = np.asarray(well.curves[target], float).copy()
    observed[mask] = np.nan
    frame = inputs_frame(well, target) if frame is None else frame
    meta = {"gap_id": gap.gap_id, "group": gap.group, "test": test}
    return Task(well_id=well.name, depth=depth, inputs=frame, target=target, predict_mask=mask, scenario=SCENARIO_GAP,
                gap_length_m=float(length_m), target_observed=observed, dataset=well.dataset, split=well.split, meta=meta)
