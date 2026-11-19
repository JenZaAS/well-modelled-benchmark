"""Gardner, Gardner & Gregory (1974) density-velocity power law, https://doi.org/10.1190/1.1440465

    rho [g/cm3] = a * Vp^b,   Vp [m/s] = 304800 / DTC [us/ft]

Published convention used in this repository: a = 0.31, b = 0.25 (Vp in m/s, *not* km/s).
DTC = 100 us/ft -> Vp = 3048 m/s -> rho = 2.30338 g/cm3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from workstreams.black_physics.models._shared.units import finite_or_nan, slowness_to_m_s

PUBLISHED_A = 0.31
PUBLISHED_B = 0.25
FIT_RANGE_US_FT = (40.0, 200.0)       # open interval used to *fit*; prediction is not limited to it
MIN_FIT_ROWS = 10                     # a fit needs more than this many rows


@dataclass(frozen=True)
class GardnerCoefficients:
    a: float
    b: float
    fitted: bool                       # False = published coefficients
    n_rows: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


PUBLISHED = GardnerCoefficients(PUBLISHED_A, PUBLISHED_B, False, 0, "published coefficients")


def gardner_density(dtc_us_ft, a: float = PUBLISHED_A, b: float = PUBLISHED_B) -> np.ndarray:
    """Bulk density (g/cm3) from compressional slowness (us/ft). NaN where DTC is not finite and > 0."""
    vp = slowness_to_m_s(dtc_us_ft)
    with np.errstate(over="ignore", invalid="ignore"):
        return finite_or_nan(a * vp ** b)


def fit_gardner(dtc_us_ft, rhob, fit_range: tuple[float, float] = FIT_RANGE_US_FT,
                min_rows: int = MIN_FIT_ROWS) -> GardnerCoefficients:
    """Least-squares fit of ln(rho) on ln(Vp) over valid pairs inside fit_range (exclusive).

    Valid pairs: finite DTC strictly inside fit_range, finite rho > 0. Falls back to the published
    coefficients when there are not more than min_rows pairs, when the predictor has a single value,
    or when the fit is not finite with b > 0 (density increasing with velocity).
    """
    dtc, rho = np.asarray(dtc_us_ft, float).reshape(-1), np.asarray(rhob, float).reshape(-1)
    lo, hi = fit_range
    use = np.isfinite(dtc) & np.isfinite(rho) & (rho > 0) & (dtc > lo) & (dtc < hi)
    n = int(use.sum())

    def fallback(why: str) -> GardnerCoefficients:
        return GardnerCoefficients(PUBLISHED_A, PUBLISHED_B, False, n, f"published coefficients: {why}")

    if n <= min_rows:
        return fallback(f"{n} valid pairs (needs more than {min_rows})")
    x = np.log(slowness_to_m_s(dtc[use]))
    y = np.log(rho[use])
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if not np.isfinite(sxx) or sxx <= 0 or np.ptp(x) <= 0:
        return fallback("DTC values are not distinct")
    b = float(xc @ (y - y.mean())) / sxx
    a = float(np.exp(y.mean() - b * x.mean()))
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return fallback(f"fit not finite and increasing (a={a:.4g}, b={b:.4g})")
    return GardnerCoefficients(a, b, True, n, f"fitted on {n} pairs with {lo:g} < DTC < {hi:g} us/ft")
