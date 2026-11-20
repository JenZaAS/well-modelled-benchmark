"""Empirical pointwise intervals, calibrated on the D wells (labelled empirical, not model-native).

    scale   TTM: local raw P90 - P10, floored at `ttm_scale_floor`;  TSPulse: 1 (constant band)
    k50/k90 equal-well-weighted 50th / 90th percentile of |p50 - measured| / scale over D gap depths
    bands   P25/P75 = p50 -/+ k50 * scale,  P05/P95 = p50 -/+ k90 * scale   (nested by construction: k50 <= k90)

Only model-made pieces are calibrated and get bands; fallback pieces keep a point value only. Depth
correlation and a change of basin can change coverage; the report measures it on the test gaps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from workstreams.black_physics.granite.config import CONFIG


def weighted_quantile(values, weights, q: float) -> float:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if not len(v):
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cw, v))


def piece_scale(arm: str, piece, floor: float = CONFIG.ttm_scale_floor) -> np.ndarray:
    if arm == "ttm":
        if piece.q_raw is None:
            return np.full(piece.n, np.nan)
        return np.maximum(piece.raw_hi - piece.raw_lo, floor)
    return np.ones(piece.n)


@dataclass
class IntervalCalibrator:
    arm: str
    k50: float
    k90: float
    floor: float
    n_wells: int
    n_samples: int
    source: str            # "D" (official) or "local" (synthetic contract check)

    def bands(self, piece) -> dict[str, np.ndarray]:
        nan = np.full(piece.n, np.nan)
        if not piece.model_made or not np.isfinite(self.k90):
            return {"p05": nan, "p25": nan.copy(), "p75": nan.copy(), "p95": nan.copy()}
        s = piece_scale(self.arm, piece, self.floor)
        m = piece.p50
        return {"p05": m - self.k90 * s, "p25": m - self.k50 * s, "p75": m + self.k50 * s, "p95": m + self.k90 * s}

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IntervalCalibrator":
        return cls(**d)


def fit(arm: str, records, source: str = "D", floor: float = CONFIG.ttm_scale_floor) -> IntervalCalibrator:
    """records: iterable of (well_id, piece, measured values on the piece). Equal weight per well."""
    rows = []
    for well, piece, y in records:
        if not piece.model_made:
            continue
        y = np.asarray(y, float)
        s = piece_scale(arm, piece, floor)
        r = np.abs(piece.p50 - y) / s
        ok = np.isfinite(r)
        if ok.any():
            rows.append((well, r[ok]))
    if not rows:
        return IntervalCalibrator(arm, float("nan"), float("nan"), floor, 0, 0, source)
    wells = sorted({w for w, _ in rows})
    per_well = {w: int(sum(len(r) for ww, r in rows if ww == w)) for w in wells}
    vals = np.concatenate([r for _, r in rows])
    wts = np.concatenate([np.full(len(r), 1.0 / per_well[w]) for w, r in rows])
    k50, k90 = weighted_quantile(vals, wts, 0.50), weighted_quantile(vals, wts, 0.90)
    return IntervalCalibrator(arm, k50, max(k90, k50), floor, len(wells), int(len(vals)), source)
