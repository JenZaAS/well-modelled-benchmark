"""Castagna, Batzle & Eastwood (1985) mudrock line, https://doi.org/10.1190/1.1441933

    Vp [km/s] = 304.8 / DTC [us/ft]
    Vs [km/s] = 0.8621 Vp - 1.1724
    DTS [us/ft] = 304.8 / Vs

An empirical trend for water-saturated clastic silicate rocks. Numerical guard (kept from the original
repository code): abstain where Vs <= 0.3 km/s. The line reaches Vs = 0.3 km/s at Vp = 1.7079 km/s
(DTC = 178.46 us/ft) and Vs = 0 at Vp = 1.3600 km/s, so the guard removes the near-zero and negative
shear speeds that would otherwise give DTS above 1016 us/ft or a negative slowness.
"""
from __future__ import annotations

import numpy as np

from workstreams.black_physics.models._shared.units import km_s_to_slowness, slowness_to_km_s

SLOPE = 0.8621
INTERCEPT_KM_S = -1.1724
VS_FLOOR_KM_S = 0.3


def mudrock_vs(vp_km_s) -> np.ndarray:
    """Shear velocity (km/s) on the mudrock line, no guard applied."""
    return SLOPE * np.asarray(vp_km_s, float) + INTERCEPT_KM_S


def mudrock_dts(dtc_us_ft, vs_floor_km_s: float = VS_FLOOR_KM_S) -> np.ndarray:
    """Shear slowness (us/ft) from compressional slowness (us/ft). NaN where DTC is not finite and > 0
    or where the mudrock Vs is not above vs_floor_km_s."""
    vs = mudrock_vs(slowness_to_km_s(dtc_us_ft))
    with np.errstate(invalid="ignore"):
        vs = np.where(vs > vs_floor_km_s, vs, np.nan)
    return km_s_to_slowness(vs)
