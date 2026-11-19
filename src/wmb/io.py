"""Loading wells: the one in-memory format every workstream sees.

A `Well` is a depth index plus a dict of standard curves (names from
wmb.curves.CORE, NaN = missing). Loaders: FORCE 2020 CSV, LAS, and a
wells.json manifest (see viewer/manifest.schema.json and data/README.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wmb.curves import CORE, FORCE_COLUMNS, match_property

SENTINELS = (-999.25, -999.0, -9999.0, 999.25, -99999.0, 9999.0, -999.99, -32768.0)


@dataclass
class Well:
    name: str
    depth: np.ndarray                                   # metres MD, increasing
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    dataset: str = ""
    split: str | None = None                            # e.g. train / leaderboard / hidden
    meta: dict = field(default_factory=dict)

    def has(self, curve: str, min_fraction: float = 0.0) -> bool:
        x = self.curves.get(curve)
        return x is not None and np.isfinite(x).mean() > min_fraction

    def completeness(self) -> dict[str, float]:
        return {c: float(np.isfinite(self.curves[c]).mean()) if c in self.curves else 0.0 for c in CORE}

    def without(self, curve: str) -> "Well":
        """Copy with one curve removed (what a contender sees at predict time)."""
        return Well(self.name, self.depth, {k: v for k, v in self.curves.items() if k != curve},
                    self.dataset, self.split, dict(self.meta))


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    for s in SENTINELS:
        x[np.isclose(x, s)] = np.nan
    return x


def read_las(path: str | Path, curve_map: dict[str, str] | None = None, dataset: str = "") -> Well:
    """Read one LAS file. curve_map: standard name -> mnemonic; if None, mnemonics are auto-matched."""
    import lasio  # optional dependency

    las = lasio.read(str(path))
    depth = _clean(las.index)
    unit = (las.curves[0].unit or "").lower()
    if unit in ("ft", "f", "feet"):
        depth = depth * 0.3048
    curves: dict[str, np.ndarray] = {}
    if curve_map:
        for std, mn in curve_map.items():
            if mn in las.keys():
                curves[std] = _clean(las[mn])
    else:
        for c in las.curves[1:]:
            std = match_property(c.mnemonic)
            if std and std not in curves:
                curves[std] = _clean(c.data)
    name = str(las.well["WELL"].value) if "WELL" in las.well else Path(path).stem
    return Well(name or Path(path).stem, depth, curves, dataset=dataset, meta={"file": str(path)})


def read_force_csv(path: str | Path, dataset: str = "force2020", split: str | None = None) -> list[Well]:
    """Read a FORCE 2020 competition CSV (semicolon separated) into Wells."""
    import pandas as pd

    df = pd.read_csv(path, sep=";")
    wells = []
    for name, g in df.groupby("WELL", sort=True):
        g = g.sort_values("DEPTH_MD")
        curves = {std: _clean(g[col].to_numpy()) for std, col in FORCE_COLUMNS.items() if col in g}
        wells.append(Well(str(name), g["DEPTH_MD"].to_numpy(float), curves, dataset, split))
    return wells


def load_manifest(root: str | Path) -> dict:
    """Read <root>/wells.json (manifest_version 1). Light checks only; the full
    schema is viewer/manifest.schema.json."""
    p = Path(root) / "wells.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("manifest_version") != 1:
        raise ValueError(f"{p}: manifest_version must be 1")
    for key in ("dataset", "wells"):
        if key not in m:
            raise ValueError(f"{p}: missing '{key}'")
    return m


def iter_manifest_wells(root: str | Path):
    """Yield Wells for every LAS-backed well in a manifest. (CSV datasets: use read_force_csv.)"""
    root = Path(root)
    m = load_manifest(root)
    ds = m["dataset"]
    for w in m["wells"]:
        cmap = {**ds.get("curve_map", {}), **w.get("curve_map", {})}
        for f in w.get("files", ds.get("files", [])):
            if f.get("format") == "las":
                well = read_las(root / f["path"], cmap or None, dataset=ds["id"])
                well.name, well.split = w["name"], w.get("split")
                yield well
