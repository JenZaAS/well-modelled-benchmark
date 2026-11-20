"""Benchmark metrics. One definition, used by every contender.

Regression (log-value prediction):
    accuracy   MAE, RMSE, R2
    coverage   share of true values inside mean +-1 sigma (ideal 0.683) and +-2 sigma (ideal 0.954);
               cover_90 / cover_50 for the 90% and 50% intervals (ideals 0.90 / 0.50)
    sharpness  mean predicted sigma, and interval_width_90 (smaller = sharper; only meaningful with good coverage)
    crps       Gaussian CRPS (one number for accuracy + calibration)
Classification (lithology): accuracy, FORCE 2020 penalty score (matrix loaded from data, not committed).
Speed: wall-clock seconds to fit and to predict, and seconds per 1,000 predicted samples
       (same laptop CPU for the final run; see docs/benchmark.md).
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from statistics import NormalDist

import numpy as np


Z90 = 1.6448536269514722      # two-sided 90% normal quantile
Z50 = 0.6744897501960817      # two-sided 50% normal quantile
_Z = {0.90: Z90, 0.50: Z50}


def gaussian_interval(p50, std, level: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    """Central interval p50 +- z * std for a Gaussian predictive distribution (the benchmark's only definition)."""
    z = _Z.get(level) or NormalDist().inv_cdf(0.5 + level / 2)
    p50, std = np.asarray(p50, float), np.asarray(std, float)
    return p50 - z * std, p50 + z * std


def _pinball(y, q, tau: float) -> float:
    d = y - q
    return float(np.maximum(tau * d, (tau - 1) * d).mean())


def regression_scores(y: np.ndarray, mean: np.ndarray, std: np.ndarray | None = None, *,
                      p05: np.ndarray | None = None, p95: np.ndarray | None = None,
                      p25: np.ndarray | None = None, p75: np.ndarray | None = None) -> dict:
    """Accuracy on finite (y, mean) pairs, plus uncertainty scores when std and/or intervals are given.

    cover_90 / interval_width_90: [p05, p95] if given, else Gaussian from std.
    cover_50: [p25, p75] if given, else Gaussian from std; omitted otherwise.
    Point-only predictions return has_uncertainty False and no coverage keys.
    """
    y, mean = np.asarray(y, float), np.asarray(mean, float)
    ok = np.isfinite(y) & np.isfinite(mean)
    out: dict = {"n": int(ok.sum()), "has_uncertainty": False}
    if not ok.any():
        return out
    yy, mm = y[ok], mean[ok]
    e = mm - yy
    out["MAE"] = float(np.abs(e).mean())
    out["RMSE"] = float(np.sqrt((e ** 2).mean()))
    var = float(((yy - yy.mean()) ** 2).mean())
    out["R2"] = float(1 - (e ** 2).mean() / var) if var > 0 else float("nan")
    sub = lambda a: None if a is None else np.asarray(a, float)[ok]  # noqa: E731
    s, lo, hi, lo5, hi5 = sub(std), sub(p05), sub(p95), sub(p25), sub(p75)
    if s is not None:
        good = np.isfinite(s) & (s > 0)
        if good.any():
            z = np.abs(e[good]) / s[good]
            out["cover_1sigma"] = float((z <= 1).mean())
            out["cover_2sigma"] = float((z <= 2).mean())
            out["sharpness_mean_sigma"] = float(s[good].mean())
            out["CRPS"] = float(crps_gaussian(yy[good], mm[good], s[good]).mean())
    def interval(lo_, hi_, level):
        """Per sample: the given [lo, hi] if both finite, else Gaussian from a finite std > 0, else NaN."""
        n_ = len(mm)
        glo, ghi = np.full(n_, np.nan), np.full(n_, np.nan)
        if s is not None:
            sg = np.where(np.isfinite(s) & (s > 0), s, np.nan)
            glo, ghi = gaussian_interval(mm, sg, level)
        given = np.zeros(n_, bool)
        if lo_ is not None and hi_ is not None:
            given = np.isfinite(lo_) & np.isfinite(hi_)
            glo, ghi = np.where(given, lo_, glo), np.where(given, hi_, ghi)
        return glo, ghi, given

    l90, h90, given90 = interval(lo, hi, 0.90)
    g = np.isfinite(l90) & np.isfinite(h90)
    if g.any():
        out["has_uncertainty"] = True
        out["cover_90"] = float(((yy[g] >= l90[g]) & (yy[g] <= h90[g])).mean())
        out["interval_width_90"] = float((h90[g] - l90[g]).mean())
    if given90.any():
        out["pinball_05"] = _pinball(yy[given90], l90[given90], 0.05)
        out["pinball_50"] = _pinball(yy[given90], mm[given90], 0.5)
        out["pinball_95"] = _pinball(yy[given90], h90[given90], 0.95)
    l50, h50, _ = interval(lo5, hi5, 0.50)
    g = np.isfinite(l50) & np.isfinite(h50)
    if g.any():
        out["has_uncertainty"] = True
        out["cover_50"] = float(((yy[g] >= l50[g]) & (yy[g] <= h50[g])).mean())
    return out

def ranked_probability_score(y, edges, probs) -> np.ndarray:
    """Ranked probability score per row of a binned forecast (lower is better; 0 = all mass in the right bin).

    edges: K-1 increasing internal edges; bin k covers [edges[k-1], edges[k]) and the end bins are open, so a value
    on an edge belongs to the higher bin. probs: [n, K]. The score is the mean over the K-1 edges of
    (P(Y < edge) - 1[y < edge])^2. Rows with a non-finite y or forecast are NaN.
    """
    y = np.asarray(y, float)
    e = np.asarray(edges, float)
    p = np.asarray(probs, float)
    cum = np.cumsum(p, axis=1)[:, :-1]
    with np.errstate(invalid="ignore"):
        obs = (y[:, None] < e[None, :]).astype(float)
    out = np.mean((cum - obs) ** 2, axis=1)
    out[~(np.isfinite(y) & np.isfinite(p).all(axis=1))] = np.nan
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
