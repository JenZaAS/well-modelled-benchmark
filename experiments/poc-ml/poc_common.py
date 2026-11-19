"""Shared bits for the FH2026 ML proof of concept (open FORCE 2020 data only, CPU only).

Data and weights are read from the cache (FH2026_CACHE, default ~/fh2026-cache); nothing is written
inside the project folder except results/ (small JSON files)."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("FH2026_CACHE", Path.home() / "fh2026-cache")).expanduser()
DATA = CACHE / "data" / "force2020"
WORK = CACHE / "poc-ml"            # logs and scratch: outside the project

# FORCE 2020 lithology codes in the order of the official penalty matrix
ORDER = [30000, 65030, 65000, 80000, 74000, 70000, 70032, 88000, 86000, 99000, 90000, 93000]
IDX = {c: i for i, c in enumerate(ORDER)}
NAMES = ["sandstone", "sandstone/shale", "shale", "marl", "dolomite", "limestone", "chalk",
         "halite", "anhydrite", "tuff", "coal", "basement"]
DESCR = ["quartz-rich sand, low gamma ray", "interbedded sand and shale", "clay-rich mudrock, high gamma ray",
         "clay-rich carbonate", "dense carbonate", "carbonate, low gamma ray, fast sonic",
         "soft fine carbonate, high porosity", "rock salt, low density, very fast sonic",
         "evaporite, high density, very low gamma ray", "volcanic ash", "very low density, slow sonic",
         "crystalline basement"]

LITHO_CURVES = ["GR", "RHOB", "NPHI", "DTC", "RDEP"]       # fixed input set, no tuning
DENS_CURVES = ["GR", "NPHI", "DTC", "RDEP"]                # RHOB is the target
COLS = ["WELL", "DEPTH_MD", "GROUP", "FORMATION", "FORCE_2020_LITHOFACIES_LITHOLOGY", "GR", "RHOB", "NPHI", "DTC", "RDEP"]
HALF = 6                                                   # +-6 samples = about +-0.9 m window


def hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
        elif platform.system() == "Windows":
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True).stdout.strip() or cpu
    except Exception:  # noqa: BLE001
        pass
    ram = None
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        if hasattr(os, "sysconf"):
            ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": ram, "os": platform.platform()}


def load_split(n_train_wells: int = 8, n_test_wells: int = 2, seed: int = 0):
    """Training wells from train.csv, held-out wells from hidden_test.csv (different wells), both
    picked at random among wells with >=80 % RHOB and GR coverage. Fixed seed, no tuning."""
    tr = pd.read_csv(DATA / "train.csv", sep=";", usecols=COLS)
    te = pd.read_csv(DATA / "hidden_test.csv", sep=";", usecols=COLS)
    def good(df):
        cov = df.groupby("WELL")[["RHOB", "GR", "NPHI", "DTC"]].apply(lambda g: g.notna().mean().min())
        return sorted(cov[cov >= 0.8].index)
    rng = np.random.default_rng(seed)
    trw = [str(w) for w in rng.choice(good(tr), n_train_wells, replace=False)]
    tew = [str(w) for w in rng.choice(good(te), n_test_wells, replace=False)]
    return tr[tr.WELL.isin(trw)].copy(), te[te.WELL.isin(tew)].copy(), trw, tew


def add_window(df: pd.DataFrame, curves: list[str]) -> pd.DataFrame:
    """Per-well rolling min/max over a centred window (the 'depth window')."""
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    g = df.groupby("WELL")
    for c in curves:
        df[f"{c}_min"] = g[c].transform(lambda s: s.rolling(2 * HALF + 1, center=True, min_periods=1).min())
        df[f"{c}_max"] = g[c].transform(lambda s: s.rolling(2 * HALF + 1, center=True, min_periods=1).max())
    return df


def fmt(x, c: str) -> str:
    if pd.isna(x):
        return "na"
    if c in ("RHOB", "NPHI"):
        return f"{x:.2f}"
    if c in ("GR", "DTC"):
        return f"{x:.0f}"
    return f"{x:.1f}"


def feature_matrix(df: pd.DataFrame, curves: list[str], formations: list[str]) -> np.ndarray:
    """Depth, centre values and window min/max as numbers for XGBoost (formation as an integer code)."""
    code = {f: i for i, f in enumerate(formations)}
    cols = ["DEPTH_MD"] + curves + [f"{c}_min" for c in curves] + [f"{c}_max" for c in curves]
    X = df[cols].to_numpy(float)
    f = df["FORMATION"].map(lambda v: code.get(v, -1)).to_numpy(float)[:, None]
    g = df["GROUP"].astype("category").cat.codes.to_numpy(float)[:, None]
    return np.hstack([X, f, g])


def bins(n: int, lo: float = 1.5, hi: float = 3.0, kind: str = "uniform", y: np.ndarray | None = None):
    """Bin edges and centres for RHOB. 'quantile' = equal-count edges from training values."""
    if kind == "quantile":
        edges = np.quantile(np.clip(y, lo, hi), np.linspace(0, 1, n + 1))
        edges[0], edges[-1] = lo, hi
    else:
        edges = np.linspace(lo, hi, n + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    return edges, centres


def to_bin(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, np.clip(y, edges[0], edges[-1]), side="right") - 1, 0, len(edges) - 2)


def dist_metrics(y: np.ndarray, mean: np.ndarray, std: np.ndarray | None = None,
                 lo1=None, hi1=None, lo2=None, hi2=None) -> dict:
    err = mean - y
    out = {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
           "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))}
    if std is not None:
        lo1, hi1, lo2, hi2 = mean - std, mean + std, mean - 2 * std, mean + 2 * std
        out["mean_std"] = float(np.mean(std))
    if lo1 is not None:
        out["cover_1sigma"] = float(np.mean((y >= lo1) & (y <= hi1)))   # ideal ~0.68
        out["cover_2sigma"] = float(np.mean((y >= lo2) & (y <= hi2)))   # ideal ~0.95
    return {k: round(v, 4) for k, v in out.items()}


def litho_scores(y: np.ndarray, p: np.ndarray) -> dict:
    A = np.load(DATA / "penalty_matrix.npy")
    return {"accuracy": round(float(np.mean(y == p)), 4), "penalty_score": round(float(-np.mean(A[y, p])), 4)}
