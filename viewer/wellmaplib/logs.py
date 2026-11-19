"""Read well logs from CSV or LAS, clean sentinels, downsample, completeness."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SENTINELS = (-999.25, -999.0, -9999.0, 999.25, -999.25)

# Standard curve keys the viewer knows how to plot.
STANDARD_CURVES = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]
INTERVAL_CURVES = ["GR", "RDEP", "RHOB", "DTC"]
LITH_KEY = "LITH"

# Common LAS mnemonics tried when the exact standard name is absent.
LAS_ALIASES = {
    "GR": ("GRC", "GR_1", "GAMMA"),
    "RDEP": ("ILD", "RT", "LLD", "RD", "AT90", "RLA5"),
    "RMED": ("ILM", "LLS", "RM", "AT60"),
    "CALI": ("CAL", "CALI_1", "HCAL"),
    "RHOB": ("RHOZ", "DEN", "RHOB_1"),
    "NPHI": ("NPOR", "TNPH", "NEU", "NPHI_1"),
    "PEF": ("PEFZ",),
    "DTC": ("DT", "DTCO", "AC", "DT24"),
    "DTS": ("DTSM", "DTSH", "DT4S"),
}

PRECISION = {
    "DEPTH": 2,
    "GR": 1,
    "RDEP": 4,  # significant figures, handled separately
    "RMED": 4,
    "CALI": 2,
    "RHOB": 3,
    "NPHI": 3,
    "PEF": 2,
    "DTC": 1,
    "DTS": 1,
    "LITH": 0,
}
SIGFIG_CURVES = {"RDEP", "RMED"}


def is_valid_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    ok = x.notna()
    for v in SENTINELS:
        ok &= (x - v).abs() > 1e-6
    return ok


def clean_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    for v in SENTINELS:
        x = x.mask((x - v).abs() <= 1e-6)
    return x


def completeness_from_frame(df: pd.DataFrame, curve_cols: Iterable[str]) -> dict:
    """Coverage of standard curves over the logged interval.

    Interval: first to last depth where any of GR/RDEP/RHOB/DTC is valid.
    Overall: mean of the 9 core properties (exact fractions, 1 decimal).
    Per-curve percentages are integers 0–100.
    """
    cores = list(curve_cols)
    valid = {c: is_valid_numeric(df[c]) if c in df.columns else pd.Series(False, index=df.index) for c in cores}
    interval = pd.Series(False, index=df.index)
    for c in INTERVAL_CURVES:
        if c in valid:
            interval = interval | valid[c]
    if not interval.any():
        pcts = {c: 0 for c in cores}
        return {"overall": 0.0, "curves": pcts, "depth_from": None, "depth_to": None, "n_samples": 0}
    depths = df.loc[interval, "DEPTH"]
    d0, d1 = float(depths.min()), float(depths.max())
    in_iv = (df["DEPTH"] >= d0) & (df["DEPTH"] <= d1)
    n = int(in_iv.sum())
    pcts = {}
    exact = []
    for c in cores:
        frac = float(valid[c][in_iv].sum() / n) if n else 0.0
        exact.append(frac)
        pcts[c] = int(min(100, max(0, round(100.0 * frac))))
    overall = round(100.0 * (sum(exact) / len(cores) if cores else 0.0), 1)
    return {"overall": overall, "curves": pcts, "depth_from": round(d0, 2), "depth_to": round(d1, 2), "n_samples": n}


def downsample_indices(n: int, max_points: int = 2000) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    step = int(math.ceil(n / max_points))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _sigfig(v: float, n: int) -> float:
    if v == 0 or not math.isfinite(v):
        return v
    return round(v, n - int(math.floor(math.log10(abs(v)))) - 1)


def quantize_value(key: str, v) -> float | int | None:
    if v is None or (isinstance(v, float) and not math.isfinite(v)) or pd.isna(v):
        return None
    fv = float(v)
    if key == LITH_KEY:
        return int(round(fv))
    if key in SIGFIG_CURVES:
        return _sigfig(fv, PRECISION[key])
    nd = PRECISION.get(key, 3)
    return round(fv, nd)


def series_to_list(key: str, s: pd.Series) -> list:
    return [quantize_value(key, v) for v in s.tolist()]


def read_force_style_csv(
    path: Path,
    *,
    delimiter: str = ";",
    well_column: str = "WELL",
    depth_column: str = "DEPTH_MD",
    x_column: str | None = "X_LOC",
    y_column: str | None = "Y_LOC",
    curve_map: dict[str, str],
    lithology_column: str | None = None,
    wells: list[str] | None = None,
    usecols_extra: list[str] | None = None,
) -> pd.DataFrame:
    """Read a semicolon CSV (FORCE-style), returning standard column names.

    Standard names: WELL, DEPTH, X, Y, plus keys in *curve_map*, plus LITH.
    """
    header = pd.read_csv(path, sep=delimiter, nrows=0).columns.tolist()
    wanted = {well_column, depth_column}
    rename = {well_column: "WELL", depth_column: "DEPTH"}
    if x_column and x_column in header:
        wanted.add(x_column)
        rename[x_column] = "X"
    if y_column and y_column in header:
        wanted.add(y_column)
        rename[y_column] = "Y"
    for std, src in curve_map.items():
        if src in header:
            wanted.add(src)
            rename[src] = std
    if lithology_column and lithology_column in header:
        wanted.add(lithology_column)
        rename[lithology_column] = LITH_KEY
    if usecols_extra:
        wanted.update(c for c in usecols_extra if c in header)
    cols = [c for c in header if c in wanted]
    df = pd.read_csv(path, sep=delimiter, usecols=cols, low_memory=False)
    df = df.rename(columns=rename)
    if wells is not None:
        df = df[df["WELL"].isin(wells)]
    return df


def merge_target_lithology(df: pd.DataFrame, target_path: Path, *, delimiter: str = ";", well_column: str = "WELL", depth_column: str = "DEPTH_MD", lithology_column: str = "FORCE_2020_LITHOFACIES_LITHOLOGY") -> pd.DataFrame:
    tgt = pd.read_csv(target_path, sep=delimiter, low_memory=False)
    if lithology_column not in tgt.columns:
        return df
    tgt = tgt.rename(columns={well_column: "WELL", depth_column: "DEPTH", lithology_column: LITH_KEY})
    if LITH_KEY in df.columns:
        df = df.drop(columns=[LITH_KEY])
    return df.merge(tgt[["WELL", "DEPTH", LITH_KEY]], on=["WELL", "DEPTH"], how="left")


def try_read_las(path: Path, curve_map: dict[str, str]) -> pd.DataFrame:
    """Read one LAS file if lasio is installed; otherwise raise ImportError."""
    import lasio  # type: ignore

    las = lasio.read(str(path), ignore_header_errors=True, engine="normal")
    data = {"DEPTH": las.index}
    # index mnemonics case-insensitively
    present = {str(c.mnemonic).upper(): c.mnemonic for c in las.curves}

    def resolve(std: str, src: str) -> str | None:
        candidates = [src, *LAS_ALIASES.get(std, ())]
        for cand in candidates:
            u = str(cand).upper()
            if u in present:
                return present[u]
            # Prefixed mnemonics such as LFP_GR or WLC_DT, not e.g. VSHGR.
            tail = "_" + u
            for name, orig in present.items():
                if name.endswith(tail):
                    return orig
        return None

    for std, src in curve_map.items():
        key = resolve(std, src)
        if key is not None:
            data[std] = las[key]
    if "LITH" not in data:
        for cand in ("LITH", "LITHOLOGY", "FACIES"):
            if cand in present:
                data["LITH"] = las[present[cand]]
                break
    df = pd.DataFrame(data)
    well_name = None
    try:
        well_name = las.well.WELL.value
    except Exception:
        well_name = path.stem
    df["WELL"] = well_name or path.stem
    return df
