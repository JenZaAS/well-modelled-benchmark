"""The fixed 15-bin density grid, continuous readouts from a binned distribution, and the scores.

Bins (frozen for every contender and figure): 14 internal edges 1.60, 1.70, ..., 2.90 g/cm3.
Bin 1 is RHOB < 1.60, bin 2 is [1.60, 1.70), ..., bin 15 is RHOB >= 2.90. A value exactly on an edge goes into
the higher bin (np.searchsorted(..., side="right"), the pilot's convention). Indices here are 0-based (0..14).

Continuous readouts use one conversion for every row: uniform mass inside each finite bin; for the two open
tails, the empirical values of the training (C) labels that fell in that tail. If a tail had no training value,
a declared one-bin-width extension ([1.50, 1.60) or [2.90, 3.00)) is used and flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDGES = np.round(np.arange(16, 30) / 10.0, 2)          # 1.60 ... 2.90, 14 internal edges
N_BINS = len(EDGES) + 1                                 # 15
WIDTH = 0.10
EVENT_EDGE = 2.40                                       # "chance that density here is below 2.40 g/cm3"
EVENT_INDEX = int(np.flatnonzero(np.isclose(EDGES, EVENT_EDGE))[0])     # 8: bins 0..8 lie entirely below 2.40
LOW_EXTENSION = (EDGES[0] - WIDTH, EDGES[0])
HIGH_EXTENSION = (EDGES[-1], EDGES[-1] + WIDTH)


def to_bin(y) -> np.ndarray:
    """0-based bin index (0..14); -1 for a missing value. Edge values go into the higher bin."""
    y = np.asarray(y, float)
    out = np.searchsorted(EDGES, y, side="right").astype(int)
    out[~np.isfinite(y)] = -1
    return out


def option_labels(unit: str = "g/cm3") -> list[str]:
    """The 15 answer options, low to high (Decima's option order and the heatmap's column order)."""
    lab = [f"below {EDGES[0]:.2f} {unit}"]
    lab += [f"{a:.2f} to below {b:.2f} {unit}" for a, b in zip(EDGES[:-1], EDGES[1:])]
    lab += [f"{EDGES[-1]:.2f} {unit} or higher"]
    return lab


def display_edges() -> np.ndarray:
    """16 edges for drawing the 15 bins (the open tails drawn one bin wide)."""
    return np.r_[EDGES[0] - WIDTH, EDGES, EDGES[-1] + WIDTH]


@dataclass
class Tails:
    """Values used inside the two open tails, from training labels only."""
    low: np.ndarray
    high: np.ndarray
    low_extended: bool
    high_extended: bool

    @classmethod
    def fit(cls, y_train) -> "Tails":
        y = np.asarray(y_train, float)
        y = y[np.isfinite(y)]
        low, high = np.sort(y[y < EDGES[0]]), np.sort(y[y >= EDGES[-1]])
        return cls(low, high, low_extended=len(low) == 0, high_extended=len(high) == 0)

    def as_dict(self) -> dict:
        return {"low": self.low.tolist(), "high": self.high.tolist(), "low_extended": self.low_extended,
                "high_extended": self.high_extended, "low_n": int(len(self.low)), "high_n": int(len(self.high))}

    @classmethod
    def from_dict(cls, d) -> "Tails":
        return cls(np.asarray(d["low"], float), np.asarray(d["high"], float), d["low_extended"], d["high_extended"])

    def within(self, k: int, u: np.ndarray) -> np.ndarray:
        """Value at within-bin probability u (0..1) of bin k."""
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        if k == 0:
            return (LOW_EXTENSION[0] + WIDTH * u) if self.low_extended else np.quantile(self.low, u)
        if k == N_BINS - 1:
            return (HIGH_EXTENSION[0] + WIDTH * u) if self.high_extended else np.quantile(self.high, u)
        return EDGES[k - 1] + WIDTH * u

    def means(self) -> np.ndarray:
        """Representative (mean) value of each bin under this conversion."""
        m = np.r_[np.nan, (EDGES[:-1] + EDGES[1:]) / 2, np.nan]
        m[0] = np.mean(LOW_EXTENSION) if self.low_extended else self.low.mean()
        m[-1] = np.mean(HIGH_EXTENSION) if self.high_extended else self.high.mean()
        return m


def normalise(p) -> np.ndarray:
    p = np.asarray(p, float)
    return p / p.sum(axis=-1, keepdims=True)


def quantile(p, q: float, tails: Tails) -> np.ndarray:
    """Quantile q of each row's mixture distribution (rows of p: 15 probabilities). NaN rows stay NaN."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p).all(axis=1)
    if not ok.any():
        return out
    pp = p[ok]
    cum = np.cumsum(pp, axis=1)
    cum[:, -1] = 1.0
    k = np.argmax(cum >= q - 1e-12, axis=1)
    below = np.where(k > 0, cum[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
    below[k == 0] = 0.0
    mass = pp[np.arange(len(k)), k]
    u = np.where(mass > 0, (q - below) / np.where(mass > 0, mass, 1.0), 0.5)
    vals = np.empty(len(k))
    for b in np.unique(k):
        sel = k == b
        vals[sel] = tails.within(int(b), u[sel])
    out[ok] = vals
    return out


# ------------------------------------------------------------------------------------- finer nested grids
def fine_edges(step: float) -> np.ndarray:
    """Internal edges every `step` from 1.60 to 2.90 g/cm3. `step` must divide the 0.10 bin width, so each fixed
    bin splits into whole sub-bins and the same two open tails remain (a nested, finer view; see coarsen)."""
    per = WIDTH / step
    if step <= 0 or abs(per - round(per)) > 1e-9:
        raise ValueError(f"step {step} does not divide the {WIDTH} bin width")
    n = int(round((EDGES[-1] - EDGES[0]) / step))
    return np.round(EDGES[0] + step * np.arange(n + 1), 6)


def parent_bins(edges) -> np.ndarray:
    """For each bin on a nested grid (len(edges) + 1 bins), the index of the fixed bin that contains it."""
    e = np.asarray(edges, float)
    if not np.isin(np.round(EDGES, 6), np.round(e, 6)).all() or not (np.isclose(e[0], EDGES[0])
                                                                       and np.isclose(e[-1], EDGES[-1])):
        raise ValueError("edges must contain every fixed edge and share the two open tails")
    inside = np.r_[e[0] - 1.0, (e[:-1] + e[1:]) / 2, e[-1] + 1.0]        # one point inside each finer bin
    return to_bin(inside)


def coarsen(p, edges) -> np.ndarray:
    """Probabilities on a nested grid summed into the 15 fixed bins."""
    p = np.asarray(p, float)
    onehot = np.zeros((p.shape[1], N_BINS))
    onehot[np.arange(p.shape[1]), parent_bins(edges)] = 1.0
    return p @ onehot


def quantile_on(p, edges, q: float, tails: Tails) -> np.ndarray:
    """quantile() for probabilities on a nested grid (e.g. fine_edges): uniform mass inside each finite bin, the
    fixed tails' training values inside the two open bins. NaN rows stay NaN."""
    p, e = np.asarray(p, float), np.asarray(edges, float)
    parent_bins(e)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p).all(axis=1)
    if not ok.any():
        return out
    pp = p[ok]
    cum = np.cumsum(pp, axis=1)
    cum[:, -1] = 1.0
    ar = np.arange(len(pp))
    k = np.argmax(cum >= q - 1e-12, axis=1)
    below = np.where(k > 0, cum[ar, np.maximum(k - 1, 0)], 0.0)
    mass = pp[ar, k]
    u = np.clip(np.where(mass > 0, (q - below) / np.where(mass > 0, mass, 1.0), 0.5), 0.0, 1.0)
    vals = np.empty(len(pp))
    first, last = k == 0, k == len(e)
    mid = ~first & ~last
    lo, hi = e[np.clip(k - 1, 0, len(e) - 1)], e[np.clip(k, 0, len(e) - 1)]
    vals[mid] = lo[mid] + (hi[mid] - lo[mid]) * u[mid]
    if first.any():
        vals[first] = tails.within(0, u[first])
    if last.any():
        vals[last] = tails.within(N_BINS - 1, u[last])
    out[ok] = vals
    return out


def readouts(p, tails: Tails) -> dict:
    """Mean (the pitch's probability-weighted estimate), median and central-interval ends of each row."""
    p = np.asarray(p, float)
    r = {"mean": p @ tails.means()}
    for name, q in (("median", 0.5), ("q05", 0.05), ("q10", 0.10), ("q90", 0.90), ("q95", 0.95)):
        r[name] = quantile(p, q, tails)
    return r


# ------------------------------------------------------------------------------------------------- scores
def rps(p, b) -> np.ndarray:
    """Ranked probability score per row: mean over the 14 edges of (P(y < edge) - 1[y < edge])^2. Lower is better."""
    p = np.asarray(p, float)
    b = np.asarray(b, int)
    cum = np.cumsum(p, axis=1)[:, :-1]
    obs = (b[:, None] <= np.arange(len(EDGES))[None, :]).astype(float)
    return np.mean((cum - obs) ** 2, axis=1)


def prob_below(p, edge: float = EVENT_EDGE) -> np.ndarray:
    """Probability that the value lies below `edge` (an exact internal edge): the mass of bins entirely below it."""
    i = np.flatnonzero(np.isclose(EDGES, edge))
    if not len(i):
        raise ValueError(f"{edge} is not one of the fixed bin edges")
    return np.asarray(p, float)[:, : int(i[0]) + 1].sum(axis=1)


def log_loss(p, b, floor: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, float)
    return -np.log(np.clip(p[np.arange(len(p)), np.asarray(b, int)], floor, 1.0))
