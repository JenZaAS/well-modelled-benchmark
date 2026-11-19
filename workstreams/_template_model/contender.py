"""Template train: log-linear regression with a Gardner-style relation. Numpy only.

Copy this folder to start a new model. For each target the model regresses ln(target) on
transformed inputs (RHOB: ln Vp from DTC, the Gardner relation; DTC: ln RHOB, its inverse;
DTS: ln DTC and ln RHOB) plus GR, NPHI and depth. Rows with missing inputs use a model fitted
on exactly the inputs that are present (fitted lazily from a training subsample); with no usable
input the prediction is the training mean. The residual standard deviation (log space, scaled by
the prediction) is returned as `std`, which gives Gaussian 90% intervals.
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction

FEATURES = {
    "RHOB": [("DTC", "vp"), ("GR", "id"), ("NPHI", "id"), ("DEPTH", "id")],
    "DTC": [("RHOB", "ln"), ("GR", "id"), ("NPHI", "id"), ("DEPTH", "id")],
    "DTS": [("DTC", "ln"), ("RHOB", "ln"), ("GR", "id"), ("NPHI", "id"), ("DEPTH", "id")],
}
MAX_TRAIN_ROWS = 60_000
MIN_ROWS = 50


def _column(depth: np.ndarray, curves: dict, name: str, kind: str) -> np.ndarray:
    x = depth if name == "DEPTH" else curves.get(name)
    if x is None:
        return np.full(len(depth), np.nan)
    x = np.asarray(x, float)
    if kind == "id":
        return x
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, np.log(304800.0 / x) if kind == "vp" else np.log(x), np.nan)


def _matrix(depth: np.ndarray, curves: dict, target: str) -> np.ndarray:
    return np.column_stack([_column(depth, curves, n, k) for n, k in FEATURES[target]])


class TemplateLinear(BaseContender):
    name, workstream = "template_linear", "_template_model"
    licence_name, licence_url = "MIT", "https://opensource.org/license/mit"
    licence_restrictions = "none"
    required_inputs = ["DEPTH"]
    optional_inputs = ["GR", "NPHI", "DTC", "RHOB", "DTS"]
    targets = ["RHOB", "DTC", "DTS"]

    def fit(self, train_wells, target):
        xs, ys = [], []
        for w in train_wells:
            y = w.curves.get(target)
            if y is None:
                continue
            y = np.asarray(y, float)
            ok = np.isfinite(y) & (y > 0)
            xs.append(_matrix(w.depth, w.curves, target)[ok])
            ys.append(np.log(y[ok]))
        x = np.vstack(xs) if xs else np.empty((0, len(FEATURES[target])))
        y = np.concatenate(ys) if ys else np.empty(0)
        if len(y) > MAX_TRAIN_ROWS:
            keep = np.linspace(0, len(y) - 1, MAX_TRAIN_ROWS).astype(int)
            x, y = x[keep], y[keep]
        self._x, self._y, self._target = x, y, target
        self._mean_ln = float(y.mean()) if len(y) else float(np.log(2.4 if target == "RHOB" else 90.0))
        self._sd_ln = float(y.std()) if len(y) else 0.2
        self._mu = np.nanmean(x, axis=0) if len(y) else np.zeros(x.shape[1])
        self._sd = np.where(np.nanstd(x, axis=0) > 0, np.nanstd(x, axis=0), 1.0) if len(y) else np.ones(x.shape[1])
        self._models: dict[tuple[bool, ...], tuple[np.ndarray, float] | None] = {}

    def _model(self, pattern: tuple[bool, ...]):
        if pattern not in self._models:
            cols = np.nonzero(pattern)[0]
            rows = np.isfinite(self._x[:, cols]).all(axis=1)
            if not cols.size or rows.sum() < MIN_ROWS:
                self._models[pattern] = None
            else:
                a = np.column_stack([np.ones(rows.sum()), (self._x[rows][:, cols] - self._mu[cols]) / self._sd[cols]])
                beta = np.linalg.lstsq(a, self._y[rows], rcond=None)[0]
                self._models[pattern] = (beta, float(np.std(self._y[rows] - a @ beta)))
        return self._models[pattern]

    def predict(self, well, target):
        x = _matrix(well.depth, well.curves, target)
        ln = np.full(len(x), self._mean_ln)
        sd = np.full(len(x), self._sd_ln)
        finite = np.isfinite(x)
        patterns, inverse = np.unique(finite, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        for k, pat in enumerate(patterns):
            fit = self._model(tuple(bool(v) for v in pat))
            if fit is None:
                continue
            rows = inverse == k
            cols = np.nonzero(pat)[0]
            a = np.column_stack([np.ones(rows.sum()), (x[rows][:, cols] - self._mu[cols]) / self._sd[cols]])
            ln[rows], sd[rows] = a @ fit[0], fit[1]
        mean = np.exp(ln)
        return Prediction(mean, sd * mean)


CONTENDERS = [TemplateLinear]
