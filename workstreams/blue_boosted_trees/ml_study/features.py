"""Feature matrix: other logs + depth, optionally + context or trend features (neighbouring depths) and stratigraphy.

Context features (``context=True``) let a row-by-row model see the logs above and below each depth. For every input log:
    <log>_above5m / <log>_below5m   the value 5 m shallower / deeper
    <log>_avg5m / <log>_avg20m      centred moving average over 5 m / 20 m
Trend features (``trend=True``) describe how a log changes with depth around each sample (10 m windows). For every input log:
    <log>_above10m / <log>_below10m  mean of the 10 m strictly above / below (the sample itself is excluded)
    <log>_trend10m                   below minus above: positive = the log increases with depth
    <log>_std10m                     standard deviation over a centred 10 m window (blocky vs. noisy)
Windows are in metres (converted with the well's median sample spacing, 0.152 m in FORCE 2020). The log being
predicted is never an input, so it gets no neighbouring-depth features either.

Stratigraphy encodings:
    ordinal  one column per level, coded by rank of median depth in the training wells
             (shallow -> deep). Cheap, and a sensible order for tree models.
    onehot   one 0/1 column per group/formation, for the linear model.
Labels never seen in training become NaN (ordinal) or all zeros (onehot).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wmb.curves import LOG_SCALE
from wmb.io import Well

from .data import UNKNOWN

STRAT_LEVELS = ("group", "formation")
CONTEXT_SHIFT_M = 5.0
CONTEXT_AVG_M = (5.0, 20.0)
CONTEXT_SUFFIXES = ("above5m", "below5m", *[f"avg{m:g}m" for m in CONTEXT_AVG_M])
TREND_WINDOW_M = 10.0
TREND_SUFFIXES = (f"above{TREND_WINDOW_M:g}m", f"below{TREND_WINDOW_M:g}m", f"trend{TREND_WINDOW_M:g}m",
                  f"std{TREND_WINDOW_M:g}m")


def _shift(x: np.ndarray, n: int) -> np.ndarray:
    """Value ``n`` samples shallower (n > 0) or deeper (n < 0); NaN where there is no such sample."""
    out = np.full(x.shape, np.nan)
    if n > 0:
        out[n:] = x[:-n]
    elif n < 0:
        out[:n] = x[-n:]
    return out


def context_blocks(x: np.ndarray, step_m: float) -> list[np.ndarray]:
    """The four context columns of one log, in the order of ``CONTEXT_SUFFIXES``."""
    n = max(1, round(CONTEXT_SHIFT_M / step_m))
    s = pd.Series(x)
    avgs = [s.rolling(max(1, round(m / step_m)), center=True, min_periods=1).mean().to_numpy() for m in CONTEXT_AVG_M]
    return [_shift(x, n), _shift(x, -n), *avgs]


def _mean_above(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the ``n`` samples strictly shallower than each sample (NaN where there are none)."""
    return pd.Series(x).rolling(n, min_periods=1).mean().shift(1).to_numpy()


def trend_blocks(x: np.ndarray, step_m: float) -> list[np.ndarray]:
    """The four trend columns of one log, in the order of ``TREND_SUFFIXES``."""
    n = max(2, round(TREND_WINDOW_M / step_m))
    above, below = _mean_above(x, n), _mean_above(x[::-1], n)[::-1]
    std = pd.Series(x).rolling(n, center=True, min_periods=2).std().to_numpy()
    return [above, below, below - above, std]


class FeatureBuilder:
    """Turn a ``Well`` into a float32 matrix. Call ``fit`` on the training wells first when using stratigraphy."""

    def __init__(self, curves: list[str], use_strat: bool = False, strat_encoding: str = "ordinal",
                 context: bool = False, trend: bool = False) -> None:
        if strat_encoding not in ("ordinal", "onehot"):
            raise ValueError(f"strat_encoding must be 'ordinal' or 'onehot', got {strat_encoding!r}")
        self.curves = list(curves)
        self.use_strat = use_strat
        self.strat_encoding = strat_encoding
        self.context = context
        self.trend = trend
        self.order: dict[str, list[str]] = {}

    def fit(self, wells: list[Well]) -> "FeatureBuilder":
        """Learn the shallow-to-deep order of the groups and formations in the training wells."""
        if not self.use_strat:
            return self
        depth = np.concatenate([w.depth for w in wells])
        for level in STRAT_LEVELS:
            labels = np.concatenate([self._labels(w, level) for w in wells])
            df = pd.DataFrame({"label": labels, "depth": depth})
            median = df[df["label"] != UNKNOWN].groupby("label")["depth"].median().sort_values()
            self.order[level] = list(median.index)
        return self

    @staticmethod
    def _labels(well: Well, level: str) -> np.ndarray:
        if level not in well.meta:
            raise ValueError(f"well {well.name} has no '{level}' in meta; load it with ml_study.data")
        return well.meta[level]

    @property
    def names(self) -> list[str]:
        """Column names of the matrix returned by ``transform``."""
        out = list(self.curves)
        if self.context:
            out += [f"{c}_{s}" for c in self.curves for s in CONTEXT_SUFFIXES]
        if self.trend:
            out += [f"{c}_{s}" for c in self.curves for s in TREND_SUFFIXES]
        out.append("DEPTH")
        if self.use_strat:
            for level in STRAT_LEVELS:
                out += [level] if self.strat_encoding == "ordinal" else [f"{level}={c}" for c in self.order[level]]
        return out

    @property
    def n_continuous(self) -> int:
        """Number of leading log/depth columns (everything before the stratigraphy columns)."""
        extra = len(CONTEXT_SUFFIXES) * self.context + len(TREND_SUFFIXES) * self.trend
        return len(self.curves) * (1 + extra) + 1

    def _curve(self, well: Well, name: str) -> np.ndarray:
        v = well.curves.get(name)
        if v is None:
            return np.full(well.depth.shape, np.nan)
        return np.log10(np.clip(v, 1e-3, None)) if name in LOG_SCALE else v

    def _strat(self, well: Well) -> np.ndarray:
        blocks = []
        for level in STRAT_LEVELS:
            labels = pd.Series(self._labels(well, level))
            if self.strat_encoding == "ordinal":
                codes = labels.map({c: i for i, c in enumerate(self.order[level])})
                blocks.append(codes.to_numpy(float)[:, None])
            else:
                blocks.append(np.column_stack([(labels == c).to_numpy(float) for c in self.order[level]]))
        return np.hstack(blocks)

    def transform(self, well: Well) -> np.ndarray:
        """Feature matrix with one row per depth sample of ``well``."""
        logs = [self._curve(well, c) for c in self.curves]
        step = float(np.median(np.diff(well.depth))) if len(well.depth) > 1 else 0.0
        step = step if step > 0 else 1.0
        cols = list(logs)
        for enabled, blocks in ((self.context, context_blocks), (self.trend, trend_blocks)):
            if enabled:
                for x in logs:
                    cols += blocks(x, step)
        X = np.column_stack([*cols, well.depth])
        if self.use_strat:
            X = np.hstack([X, self._strat(well)])
        return X.astype(np.float32)
