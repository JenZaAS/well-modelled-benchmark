"""Constant Vp/Vs control (cf. Pickett 1963, https://doi.org/10.2118/452-PA).

    DTS = r * DTC,   r = DTS / DTC = Vp / Vs   (dimensionless; any common slowness unit)

r is fitted as the median across fitting wells of each well's median positive, finite DTS/DTC ratio.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def well_ratio(dtc, dts) -> float | None:
    """Median DTS/DTC over rows where both are finite and > 0; None when there is no such row."""
    dtc, dts = np.asarray(dtc, float), np.asarray(dts, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(dtc) & np.isfinite(dts) & (dtc > 0) & (dts > 0)
        r = dts[ok] / dtc[ok]
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.median(r)) if r.size else None


def fit_ratio(pairs_by_well: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float | None, int]:
    """(median of per-well median ratios, number of wells with a ratio). (None, 0) without paired wells."""
    ratios = [r for r in (well_ratio(dtc, dts) for dtc, dts in pairs_by_well) if r is not None]
    return (float(np.median(ratios)), len(ratios)) if ratios else (None, 0)


def ratio_dts(dtc_us_ft, ratio: float | None) -> np.ndarray:
    """DTS = ratio * DTC on finite DTC > 0; all NaN when ratio is None or not finite and > 0."""
    d = np.asarray(dtc_us_ft, float)
    out = np.full(d.shape, np.nan)
    if ratio is None or not (np.isfinite(ratio) and ratio > 0):
        return out
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(d) & (d > 0)
    out[ok] = ratio * d[ok]
    out[~np.isfinite(out)] = np.nan
    return out
