"""Wyllie, Gregory & Gardner (1956) time average, https://doi.org/10.1190/1.1438217

    DTC = (1 - phi) * dt_matrix + phi * dt_fluid           (all slownesses in us/ft, phi in v/v)

Reference endmembers: quartz matrix 55.5 us/ft, water 189 us/ft (Kansas Geological Survey values).
phi = 0.20 -> 82.2 us/ft.
"""
from __future__ import annotations

import numpy as np

from workstreams.black_physics.models._shared.units import require_positive

QUARTZ_DT_US_FT = 55.5
WATER_DT_US_FT = 189.0


def porosity_ok(phi) -> np.ndarray:
    """Admissible porosity: finite and within [0, 1] v/v."""
    p = np.asarray(phi, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(p) & (p >= 0) & (p <= 1)


def wyllie_dtc(phi, dt_matrix: float = QUARTZ_DT_US_FT, dt_fluid: float = WATER_DT_US_FT) -> np.ndarray:
    """Compressional slowness (us/ft); NaN where phi is not finite or outside [0, 1]."""
    require_positive(dt_matrix=dt_matrix, dt_fluid=dt_fluid)
    p = np.asarray(phi, float)
    ok = porosity_ok(p)
    out = np.full(p.shape, np.nan)
    out[ok] = (1.0 - p[ok]) * dt_matrix + p[ok] * dt_fluid
    return out
