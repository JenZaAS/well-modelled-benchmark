"""Feature matrix: other logs + depth, optionally + stratigraphy (GROUP, FORMATION).

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


class FeatureBuilder:
    """Turn a ``Well`` into a float32 matrix. Call ``fit`` on the training wells first when using stratigraphy."""

    def __init__(self, curves: list[str], use_strat: bool = False, strat_encoding: str = "ordinal") -> None:
        if strat_encoding not in ("ordinal", "onehot"):
            raise ValueError(f"strat_encoding must be 'ordinal' or 'onehot', got {strat_encoding!r}")
        self.curves = list(curves)
        self.use_strat = use_strat
        self.strat_encoding = strat_encoding
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
        out = [*self.curves, "DEPTH"]
        if self.use_strat:
            for level in STRAT_LEVELS:
                out += [level] if self.strat_encoding == "ordinal" else [f"{level}={c}" for c in self.order[level]]
        return out

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
        X = np.column_stack([*[self._curve(well, c) for c in self.curves], well.depth])
        if self.use_strat:
            X = np.hstack([X, self._strat(well)])
        return X.astype(np.float32)
