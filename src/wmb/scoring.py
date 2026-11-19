"""Benchmark metrics. One definition, used by every contender.

Regression (log-value prediction):
    accuracy   MAE, RMSE, R2
    coverage   share of true values inside mean +-1 sigma (ideal 0.683) and +-2 sigma (ideal 0.954)
    sharpness  mean predicted sigma (smaller = sharper; only meaningful with good coverage)
    crps       Gaussian CRPS (one number for accuracy + calibration)
Classification (lithology): accuracy, FORCE 2020 penalty score (matrix loaded from data, not committed).
Speed: wall-clock seconds to fit and to predict, and seconds per 1,000 predicted samples
       (same laptop CPU for the final run; see docs/benchmark.md).
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np


def regression_scores(y: np.ndarray, mean: np.ndarray, std: np.ndarray | None = None) -> dict:
    y, mean = np.asarray(y, float), np.asarray(mean, float)
    ok = np.isfinite(y) & np.isfinite(mean)
    out: dict = {"n": int(ok.sum())}
    if not ok.any():
        return out
    e = mean[ok] - y[ok]
    out["MAE"] = float(np.abs(e).mean())
    out["RMSE"] = float(np.sqrt((e ** 2).mean()))
    var = float(((y[ok] - y[ok].mean()) ** 2).mean())
    out["R2"] = float(1 - (e ** 2).mean() / var) if var > 0 else float("nan")
    if std is not None:
        s = np.asarray(std, float)[ok]
        good = np.isfinite(s) & (s > 0)
        if good.any():
            z = np.abs(e[good]) / s[good]
            out["cover_1sigma"] = float((z <= 1).mean())
            out["cover_2sigma"] = float((z <= 2).mean())
            out["sharpness_mean_sigma"] = float(s[good].mean())
            out["CRPS"] = float(crps_gaussian(y[ok][good], mean[ok][good], s[good]).mean())
    return out


def crps_gaussian(y, mu, sigma) -> np.ndarray:
    """Closed-form CRPS for a normal predictive distribution (Gneiting & Raftery 2007)."""
    z = (np.asarray(y) - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def classification_scores(y: np.ndarray, pred: np.ndarray, penalty_matrix: np.ndarray | None = None,
                          class_index: dict | None = None) -> dict:
    """penalty_matrix: the FORCE 2020 12x12 matrix (from the data folder); class_index: code -> row."""
    y, pred = np.asarray(y), np.asarray(pred)
    out = {"n": int(len(y)), "accuracy": float((y == pred).mean()) if len(y) else float("nan")}
    if penalty_matrix is not None and class_index is not None and len(y):
        i = np.vectorize(class_index.get)(y)
        j = np.vectorize(class_index.get)(pred)
        out["penalty_score"] = float(-penalty_matrix[i, j].mean())
    return out


@contextmanager
def timer(store: dict, key: str):
    """with timer(t, 'predict_s'): ...  -> t['predict_s'] = seconds"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0
