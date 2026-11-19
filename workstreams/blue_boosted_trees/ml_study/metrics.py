"""Error statistics for one prediction vs the measured log (builds on ``wmb.scoring``)."""
from __future__ import annotations

import numpy as np

from wmb import scoring
from wmb.curves import LOG_SCALE, UNITS

METRIC_COLUMNS = ["n", "MAE", "RMSE", "R2", "bias", "MedAE", "MAE_rel_std", "pearson_r"]
HIGHER_IS_BETTER = {"R2", "pearson_r", "n"}


def target_space(target: str, x: np.ndarray) -> np.ndarray:
    """Values in the space where a log is modelled and scored: log10 for resistivity (LOG_SCALE), else unchanged."""
    x = np.asarray(x, float)
    return np.log10(np.clip(x, 1e-3, None)) if target in LOG_SCALE else x


def target_unit(target: str) -> str:
    """Unit of the scored values, e.g. 'g/cm3' or 'log10(ohm.m)'."""
    return f"log10({UNITS[target]})" if target in LOG_SCALE else UNITS[target]


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R2 plus bias (mean of predicted - measured), median abs error,
    MAE relative to the spread of the measured log, and Pearson r. NaN pairs are ignored."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    base = scoring.regression_scores(y, pred)
    if not ok.any():
        return {k: float("nan") for k in METRIC_COLUMNS} | {"n": 0}
    e = pred[ok] - y[ok]
    spread = float(np.std(y[ok]))
    pearson = float(np.corrcoef(y[ok], pred[ok])[0, 1]) if ok.sum() > 2 and spread > 0 and np.std(pred[ok]) > 0 else float("nan")
    return {
        "n": base["n"], "MAE": base["MAE"], "RMSE": base["RMSE"], "R2": base["R2"],
        "bias": float(e.mean()), "MedAE": float(np.median(np.abs(e))),
        "MAE_rel_std": base["MAE"] / spread if spread > 0 else float("nan"),
        "pearson_r": pearson,
    }
