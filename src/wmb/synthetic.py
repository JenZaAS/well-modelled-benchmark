"""Fake but plausible wells for smoke tests and CI. Generated in memory, never saved.
Not real data and not a scientific test."""
from __future__ import annotations

import numpy as np

from wmb.io import Well


def make_wells(n_wells: int = 8, n_samples: int = 1500, seed: int = 0) -> list[Well]:
    rng = np.random.default_rng(seed)
    wells = []
    for k in range(n_wells):
        depth = 1500 + 0.15 * np.arange(n_samples)
        shale = np.clip(np.convolve(rng.normal(0.5, 0.35, n_samples), np.ones(40) / 40, "same"), 0, 1)
        gr = 30 + 90 * shale + rng.normal(0, 4, n_samples)
        dtc = 75 + 35 * shale + 10 * rng.random() + rng.normal(0, 2, n_samples)        # us/ft
        vp = 304800 / dtc                                                                # m/s
        rhob = 0.31 * vp ** 0.25 + rng.normal(0, 0.03, n_samples)                        # Gardner + noise
        nphi = 0.08 + 0.3 * shale + rng.normal(0, 0.01, n_samples)
        rdep = 10 ** (1.3 - 0.9 * shale + rng.normal(0, 0.08, n_samples))
        vs = 0.8621 * vp / 1000 - 1.1724                                                 # mudrock line, km/s
        dts = 304.8 / np.clip(vs, 0.3, None) + rng.normal(0, 3, n_samples)
        curves = {"GR": gr, "DTC": dtc, "RHOB": rhob, "NPHI": nphi, "RDEP": rdep, "DTS": dts}
        wells.append(Well(f"SYN-{k + 1:02d}", depth, curves, dataset="synthetic", split="train"))
    return wells
