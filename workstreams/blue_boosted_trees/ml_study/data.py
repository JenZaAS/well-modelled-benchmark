"""Load FORCE 2020 wells with stratigraphy (GROUP / FORMATION) attached to ``Well.meta``.

The shared ``wmb.io.read_force_csv`` drops the stratigraphy columns, so this workstream
keeps its own loader. Everything else (the ``Well`` type, curve names, official split)
comes from ``wmb``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wmb import paths, splits
from wmb.curves import FORCE_COLUMNS
from wmb.io import SENTINELS, Well

UNKNOWN = "UNKNOWN"
TRAIN_FILE = ("train.csv", "train")
TEST_FILES = (("leaderboard_test_features.csv", "leaderboard"), ("hidden_test.csv", "hidden"))
_CURVE_COLUMNS = {std: col for std, col in FORCE_COLUMNS.items() if std != "LITH"}


def _clean(x: np.ndarray) -> np.ndarray:
    """Replace the usual no-data sentinels with NaN."""
    x = np.asarray(x, dtype=float).copy()
    for s in SENTINELS:
        x[np.isclose(x, s)] = np.nan
    return x


def read_wells(csv_path: Path, split: str) -> list[Well]:
    """Read one FORCE 2020 CSV into Wells; ``meta['group']`` and ``meta['formation']`` hold strings per sample."""
    cols = {"WELL", "DEPTH_MD", "GROUP", "FORMATION", *_CURVE_COLUMNS.values()}
    df = pd.read_csv(csv_path, sep=";", usecols=lambda c: c in cols)
    wells = []
    for name, g in df.groupby("WELL", sort=True):
        g = g.sort_values("DEPTH_MD", kind="stable")
        curves = {std: _clean(g[col].to_numpy()) for std, col in _CURVE_COLUMNS.items() if col in g}
        meta = {
            "group": g["GROUP"].fillna(UNKNOWN).astype(str).to_numpy(),
            "formation": g["FORMATION"].fillna(UNKNOWN).astype(str).to_numpy(),
        }
        wells.append(Well(str(name), g["DEPTH_MD"].to_numpy(float), curves, "force2020", split, meta))
    return wells


def load_test_wells(root: Path | None = None) -> list[Well]:
    """The 20 held-out FORCE 2020 wells (leaderboard + hidden)."""
    root = root or paths.get_dataset_path("force2020")
    wells = [w for file, split in TEST_FILES for w in read_wells(root / file, split)]
    return sorted(wells, key=lambda w: w.name)


def load_official_split(root: Path | None = None) -> tuple[list[Well], list[Well]]:
    """(train, test) using the official FORCE 2020 split: 98 training wells, 20 held-out wells."""
    root = root or paths.get_dataset_path("force2020")
    wells = read_wells(root / TRAIN_FILE[0], TRAIN_FILE[1])
    wells += [w for file, split in TEST_FILES for w in read_wells(root / file, split)]
    train, test = splits.official_split(wells)
    return train, sorted(test, key=lambda w: w.name)
