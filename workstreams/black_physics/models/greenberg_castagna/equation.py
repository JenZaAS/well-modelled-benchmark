"""Greenberg & Castagna (1992) brine-saturated shear-velocity mixture, https://doi.org/10.1111/j.1365-2478.1992.tb00371.x

For each active solid lithology i (fraction f_i > 0, fractions summing to 1):

    s_i = a2 Vp^2 + a1 Vp + a0                     (Vp and s_i in km/s)
    Vs  = [ sum(f_i s_i) + 1 / sum(f_i / s_i) ] / 2

Only the final Vs is converted to slowness. The coefficients are the published endmember trends for porous,
brine-saturated lithologies. Hydrocarbon-bearing intervals need the full method with Gassmann fluid
substitution (not implemented here; it needs porosity, saturation, mineral and fluid properties).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from workstreams.black_physics.models._shared.units import km_s_to_slowness, slowness_to_km_s

# (a2, a1, a0) for Vs = a2 Vp^2 + a1 Vp + a0, km/s
GC_COEFFICIENTS: dict[str, tuple[float, float, float]] = {
    "sandstone": (0.0, 0.80416, -0.85588),
    "limestone": (-0.05508, 1.01677, -1.03049),
    "dolomite": (0.0, 0.58321, -0.07775),
    "shale": (0.0, 0.76969, -0.86735),
}
FRACTION_SUM_TOL = 1e-9
GR_LOW_PERCENTILE = 10.0
GR_HIGH_PERCENTILE = 90.0


def endmember_vs(vp_km_s, lithology: str) -> np.ndarray:
    """Brine-saturated shear speed (km/s) of one pure lithology; no masking."""
    a2, a1, a0 = GC_COEFFICIENTS[lithology]
    vp = np.asarray(vp_km_s, float)
    with np.errstate(over="ignore", invalid="ignore"):
        return a2 * vp ** 2 + a1 * vp + a0


def gc_brine_vs(vp_km_s, fractions: Mapping[str, object], sum_tol: float = FRACTION_SUM_TOL) -> np.ndarray:
    """Mixture shear velocity (km/s). `fractions` maps lithology name -> solid fraction (scalar or array).

    A row is NaN unless Vp is finite and > 0, every fraction is finite and >= 0, the fractions sum to 1
    within sum_tol, and every active (f > 0) endmember speed is > 0. Zero-weight endmembers are skipped.
    """
    if not fractions:
        raise ValueError("give at least one lithology fraction")
    unknown = sorted(set(fractions) - set(GC_COEFFICIENTS))
    if unknown:
        raise ValueError(f"unknown lithologies {unknown}; choose from {sorted(GC_COEFFICIENTS)}")
    vp = np.asarray(vp_km_s, float)
    fr = {k: np.asarray(v, float) for k, v in fractions.items()}
    shape = np.broadcast_shapes(vp.shape, *(f.shape for f in fr.values()))
    vp = np.broadcast_to(vp, shape)
    fr = {k: np.broadcast_to(f, shape) for k, f in fr.items()}

    valid = np.isfinite(vp) & (vp > 0)
    total = np.zeros(shape)
    for f in fr.values():
        valid &= np.isfinite(f) & (f >= 0)
        total = total + np.where(np.isfinite(f), f, 0.0)
    valid &= np.abs(total - 1.0) <= sum_tol
    speeds = {k: endmember_vs(vp, k) for k in fr}
    for k, f in fr.items():
        with np.errstate(invalid="ignore"):
            valid &= ~((f > 0) & ~(speeds[k] > 0))

    arith, harm = np.zeros(shape), np.zeros(shape)
    for k, f in fr.items():
        active = valid & (f > 0)
        arith[active] += f[active] * speeds[k][active]
        harm[active] += f[active] / speeds[k][active]
    vs = np.full(shape, np.nan)
    vs[valid] = 0.5 * (arith[valid] + 1.0 / harm[valid])
    return vs


def gc_brine_dts(dtc_us_ft, fractions: Mapping[str, object], sum_tol: float = FRACTION_SUM_TOL) -> np.ndarray:
    """Mixture shear slowness (us/ft) from compressional slowness (us/ft)."""
    return km_s_to_slowness(gc_brine_vs(slowness_to_km_s(dtc_us_ft), fractions, sum_tol))


# --- two-lithology gamma-ray adapter (sand and shale, both brine-filled) -----------------------------

def gr_well_endpoints(gr_by_well: Sequence[np.ndarray], low_pct: float = GR_LOW_PERCENTILE,
                      high_pct: float = GR_HIGH_PERCENTILE) -> tuple[float, float, int] | None:
    """Median across wells of each well's low/high percentile of finite GR. None when no well has GR."""
    lows, highs = [], []
    for gr in gr_by_well:
        g = np.asarray(gr, float)
        g = g[np.isfinite(g)]
        if g.size:
            lo, hi = np.percentile(g, [low_pct, high_pct])
            lows.append(lo)
            highs.append(hi)
    if not lows:
        return None
    return float(np.median(lows)), float(np.median(highs)), len(lows)


def gr_shale_fraction(gr, gr_low: float | None, gr_high: float | None) -> np.ndarray:
    """clip((GR - low) / (high - low), 0, 1); NaN where GR is not finite or the endpoints are unusable."""
    g = np.asarray(gr, float)
    out = np.full(g.shape, np.nan)
    if gr_low is None or gr_high is None or not (np.isfinite(gr_low) and np.isfinite(gr_high)) or gr_high <= gr_low:
        return out
    ok = np.isfinite(g)
    out[ok] = np.clip((g[ok] - gr_low) / (gr_high - gr_low), 0.0, 1.0)
    return out


def gc_brine_gr_dts(dtc_us_ft, gr, gr_low: float | None, gr_high: float | None) -> np.ndarray:
    """DTS (us/ft) from DTC and GR, with sand/shale fractions from linear GR scaling between the endpoints."""
    f_shale = gr_shale_fraction(gr, gr_low, gr_high)
    return gc_brine_dts(dtc_us_ft, {"sandstone": 1.0 - f_shale, "shale": f_shale})
