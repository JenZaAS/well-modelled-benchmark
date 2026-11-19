"""Slowness <-> velocity conversion. Canonical sonic slowness is in us/ft.

    V [km/s] = 304.8   / DT [us/ft]
    V [m/s]  = 304800  / DT [us/ft]

The two velocity units differ by exactly 1000x. Gardner's density relation uses m/s; the mudrock line and
Greenberg-Castagna use km/s. Every function returns NaN (never raises) where the input is not a finite,
strictly positive number, or where the result would not be finite.
"""
from __future__ import annotations

import numpy as np

KM_S_PER_INV_US_FT = 304.8        # 1 ft = 0.3048 m, so 1 / (1 us/ft) = 304.8 km/s
M_S_PER_INV_US_FT = 304_800.0


def require_positive(**values: float) -> None:
    """Raise ValueError unless every named constant (an endmember, a coefficient) is finite and > 0."""
    for k, v in values.items():
        if not (np.isfinite(v) and v > 0):
            raise ValueError(f"{k} must be finite and > 0, got {v}")


def finite_or_nan(x) -> np.ndarray:
    """Float copy of x with every non-finite value (inf, -inf, NaN) replaced by NaN."""
    a = np.array(x, dtype=float, copy=True)
    a[~np.isfinite(a)] = np.nan
    return a


def _reciprocal(scale: float, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0)
    with np.errstate(over="ignore", divide="ignore"):
        out[ok] = scale / x[ok]
    return finite_or_nan(out)


def slowness_to_km_s(dt_us_ft) -> np.ndarray:
    """Velocity in km/s from slowness in us/ft."""
    return _reciprocal(KM_S_PER_INV_US_FT, dt_us_ft)


def slowness_to_m_s(dt_us_ft) -> np.ndarray:
    """Velocity in m/s from slowness in us/ft (1000x the km/s value)."""
    return _reciprocal(M_S_PER_INV_US_FT, dt_us_ft)


def km_s_to_slowness(v_km_s) -> np.ndarray:
    """Slowness in us/ft from velocity in km/s."""
    return _reciprocal(KM_S_PER_INV_US_FT, v_km_s)


def m_s_to_slowness(v_m_s) -> np.ndarray:
    """Slowness in us/ft from velocity in m/s."""
    return _reciprocal(M_S_PER_INV_US_FT, v_m_s)
