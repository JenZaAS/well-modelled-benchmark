"""Decision features for one reconstructed gap, and the Decima text state built from the same values.

Every feature is computed from the reconstruction, the observed flanks and the fixed Gardner estimate,
never from the hidden measurements. The text state is generated from `SCHEMA`, so the numeric classifier and
Decima read the same summary statistic (rounded to three decimals in the text).
"""
from __future__ import annotations

import math

import numpy as np

TEMPLATE_VERSION = "gap-state-v1"

# name, unit, description; the order is the classifier's column order
SCHEMA = [
    ("width", "g/cm3", "TTM: mean raw P90-P10 inside the gap; TSPulse: mean |flank value - flank median| (variability proxy)"),
    ("flank_median_diff", "g/cm3", "|median of the shallow context - median of the deep context| (NaN if one-sided)"),
    ("pred_range", "g/cm3", "max - min of the reconstructed curve inside the gap"),
    ("ctx_above_m", "m", "metres of measured context used above the gap"),
    ("ctx_below_m", "m", "metres of measured context used below the gap"),
    ("gardner_mad", "g/cm3", "median |reconstruction - fixed Gardner from DTC| over gap depths with DTC (NaN if none)"),
    ("dtc_fraction", "1", "fraction of gap depths with usable DTC (a finite Gardner estimate)"),
]
BASE = ["width", "flank_median_diff", "pred_range", "ctx_above_m", "ctx_below_m"]
WITH_GARDNER = BASE + ["gardner_mad", "dtc_fraction"]
WIDTH_LABEL = {"ttm": "Predicted P10-P90 width", "tspulse": "Flank variability"}


def width_proxy(arm: str, piece) -> float:
    if arm == "ttm":
        if piece.q_raw is None:
            return math.nan
        return float(np.mean(piece.raw_hi - piece.raw_lo))
    return float(piece.flank_var)


def gap_features(arm: str, piece, gardner_gap) -> dict:
    """Features of one piece. gardner_gap: fixed-Gardner RHOB on the piece's depths (NaN where DTC is unusable)."""
    p50 = np.asarray(piece.p50, float)
    g = np.asarray(gardner_gap, float)
    finite = np.isfinite(p50)
    both = finite & np.isfinite(g)
    diff = abs(piece.flank_med_above - piece.flank_med_below)
    return {
        "width": width_proxy(arm, piece),
        "flank_median_diff": float(diff) if np.isfinite(diff) else math.nan,
        "pred_range": float(np.ptp(p50[finite])) if finite.any() else math.nan,
        "ctx_above_m": float(piece.ctx_above_m),
        "ctx_below_m": float(piece.ctx_below_m),
        "gardner_mad": float(np.median(np.abs(p50[both] - g[both]))) if both.any() else math.nan,
        "dtc_fraction": float(np.isfinite(g).mean()) if len(g) else 0.0,
        "status": piece.status,
        "complete": bool(finite.all()),
    }


def _num(v: float, unit: str = "g/cm³", nd: int = 3) -> str:
    return f"{v:.{nd}f} {unit}"


def state_text(arm: str, f: dict, gap_length_m: float = 50.0) -> str:
    """Deterministic short description of one gap for Decima. No well name, split, error, label or hidden value."""
    parts = [f"Bulk-density gap, {gap_length_m:g} m."]
    d = f["flank_median_diff"]
    parts.append(f"Shallow/deep flank medians differ by {_num(d)}." if np.isfinite(d)
                 else "Only one flank is measured, so flank medians cannot be compared.")
    parts.append(f"Observed context {f['ctx_above_m']:.0f} m above, {f['ctx_below_m']:.0f} m below.")
    w = f["width"]
    parts.append(f"{WIDTH_LABEL[arm]} {_num(w)}." if np.isfinite(w) else f"{WIDTH_LABEL[arm]} not available.")
    r = f["pred_range"]
    parts.append(f"Predicted density range {_num(r)}." if np.isfinite(r) else "Predicted density range not available.")
    g = f["gardner_mad"]
    parts.append(f"Median disagreement with sonic-derived density {_num(g)}." if np.isfinite(g)
                 else "No sonic-derived density is available in the gap.")
    parts.append(f"Sonic availability {100 * f['dtc_fraction']:.0f}%.")
    return " ".join(parts)


def schema_doc(arm: str) -> dict:
    return {"version": TEMPLATE_VERSION, "arm": arm, "features": [dict(zip(("name", "unit", "description"), s)) for s in SCHEMA],
            "template_example": state_text(arm, {"width": 0.03, "flank_median_diff": 0.08, "pred_range": 0.12,
                                                 "ctx_above_m": 14, "ctx_below_m": 13, "gardner_mad": 0.04,
                                                 "dtc_fraction": 1.0})}
