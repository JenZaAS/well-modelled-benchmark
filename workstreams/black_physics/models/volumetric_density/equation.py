"""Volumetric density mixing.

    rho_bulk = (1 - phi) * sum_i(f_i * rho_i) + phi * rho_fluid

f_i are solid-normalised mineral fractions (summing to 1), rho in g/cm3, phi in v/v.
Reference configuration: quartz 2.65 g/cm3, water 1.00 g/cm3. phi = 0.20 -> 2.32 g/cm3.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from workstreams.black_physics.models._shared.units import require_positive

QUARTZ_DENSITY = 2.65
WATER_DENSITY = 1.00
FRACTION_SUM_TOL = 1e-9


def matrix_density(fractions: Mapping[str, object], densities: Mapping[str, float],
                   sum_tol: float = FRACTION_SUM_TOL) -> np.ndarray:
    """sum(f_i * rho_i). NaN on rows whose fractions are not finite, >= 0 and summing to 1 within sum_tol."""
    if not fractions:
        raise ValueError("give at least one mineral fraction")
    missing = sorted(set(fractions) - set(densities))
    if missing:
        raise ValueError(f"no density given for {missing}")
    for k in fractions:
        require_positive(**{f"density of {k}": float(densities[k])})
    fr = {k: np.asarray(v, float) for k, v in fractions.items()}
    shape = np.broadcast_shapes(*(f.shape for f in fr.values()))
    valid = np.ones(shape, bool)
    total, rho = np.zeros(shape), np.zeros(shape)
    for k, f in fr.items():
        f = np.broadcast_to(f, shape)
        with np.errstate(invalid="ignore"):
            valid &= np.isfinite(f) & (f >= 0)
        f0 = np.where(np.isfinite(f), f, 0.0)
        total += f0
        rho += f0 * float(densities[k])
    valid &= np.abs(total - 1.0) <= sum_tol
    return np.where(valid, rho, np.nan)


def bulk_density(phi, rho_matrix=QUARTZ_DENSITY, rho_fluid: float = WATER_DENSITY) -> np.ndarray:
    """Bulk density (g/cm3). NaN where phi is not finite or outside [0, 1], or rho_matrix is not finite > 0."""
    require_positive(rho_fluid=rho_fluid)
    p, rm = np.asarray(phi, float), np.asarray(rho_matrix, float)
    shape = np.broadcast_shapes(p.shape, rm.shape)
    p, rm = np.broadcast_to(p, shape), np.broadcast_to(rm, shape)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(p) & (p >= 0) & (p <= 1) & np.isfinite(rm) & (rm > 0)
    out = np.full(shape, np.nan)
    out[ok] = (1.0 - p[ok]) * rm[ok] + p[ok] * rho_fluid
    return out


def mixed_density(phi, fractions: Mapping[str, object] | None = None, densities: Mapping[str, float] | None = None,
                  rho_fluid: float = WATER_DENSITY) -> np.ndarray:
    """Full form: matrix from mineral fractions and densities (default: pure quartz), then bulk density."""
    if fractions is None:
        fractions, densities = {"quartz": 1.0}, {"quartz": QUARTZ_DENSITY}
    return bulk_density(phi, matrix_density(fractions, densities or {}), rho_fluid)
