#!/usr/bin/env python3
"""Per-well curve completeness for FORCE 2020 (FH2026 inventory).

Logged interval: first to last depth where ANY of GR/RDEP/RHOB/DTC is valid.
Coverage: % of samples in that interval with a valid value (non-null, not -999.25
or other common LAS sentinels -999 / -9999 / 999.25).
Overall: mean of the 9 core properties (exact fractions, reported to 1 decimal).
Percentages reported as integers 0-100.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import os as _os
_DATA = Path(_os.environ.get("FH2026_DATA", Path.home() / "fh2026-cache" / "data"))
_WORK = Path(_os.environ.get("FH2026_WORK", Path.home() / "fh2026-cache" / "work"))

BASE = Path(_os.environ.get("FH2026_FORCE", _DATA / "force2020"))
OUT = _WORK / "inventory" / "force_completeness.csv"  # aggregates, outside the repo

CORE = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]
INTERVAL_CURVES = ["GR", "RDEP", "RHOB", "DTC"]
LITHO = "FORCE_2020_LITHOFACIES_LITHOLOGY"
SENTINELS = {-999.25, -999.0, -9999.0, 999.25}


def is_valid(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    ok = x.notna()
    # exact / near-exact sentinel match
    for v in SENTINELS:
        ok &= (x - v).abs() > 1e-6
    return ok


def load_split(name: str, path: Path, litho_path: Path | None = None) -> pd.DataFrame:
    usecols = ["WELL", "DEPTH_MD"] + CORE
    cols_in_file = pd.read_csv(path, sep=";", nrows=0).columns.tolist()
    if LITHO in cols_in_file:
        usecols = usecols + [LITHO]
    df = pd.read_csv(path, sep=";", usecols=lambda c: c in usecols, low_memory=False)
    if litho_path is not None:
        tgt = pd.read_csv(litho_path, sep=";", low_memory=False)
        # drop litho if already present
        if LITHO in df.columns:
            df = df.drop(columns=[LITHO])
        df = df.merge(tgt, on=["WELL", "DEPTH_MD"], how="left")
    return df


def well_stats(g: pd.DataFrame, split: str) -> dict:
    valid_core = {c: is_valid(g[c]) if c in g.columns else pd.Series(False, index=g.index) for c in CORE}
    interval_mask = False
    for c in INTERVAL_CURVES:
        interval_mask = interval_mask | valid_core[c]
    if not interval_mask.any():
        return {
            "well": g["WELL"].iloc[0],
            "split": split,
            "depth_from": "",
            "depth_to": "",
            "n_samples": 0,
            **{c: 0 for c in CORE},
            "extras": "LITHO=0; no interval (no GR/RDEP/RHOB/DTC)",
            "overall": 0.0,
        }
    depths = g.loc[interval_mask, "DEPTH_MD"]
    d0 = float(depths.min())
    d1 = float(depths.max())
    in_interval = (g["DEPTH_MD"] >= d0) & (g["DEPTH_MD"] <= d1)
    n = int(in_interval.sum())
    pcts = {}
    exact = []
    for c in CORE:
        v = int(round(100.0 * valid_core[c][in_interval].sum() / n)) if n else 0
        # clamp after rounding
        v = min(100, max(0, v))
        pcts[c] = v
        exact.append((valid_core[c][in_interval].sum() / n) if n else 0.0)
    overall = round(100.0 * sum(exact) / len(CORE), 1)

    litho_pct = ""
    if LITHO in g.columns:
        litho_ok = is_valid(g[LITHO])
        lp = int(round(100.0 * litho_ok[in_interval].sum() / n)) if n else 0
        litho_pct = f"LITHO={lp}"
    return {
        "well": g["WELL"].iloc[0],
        "split": split,
        "depth_from": round(d0, 2),
        "depth_to": round(d1, 2),
        "n_samples": n,
        **pcts,
        "extras": litho_pct,
        "overall": overall,
    }


def main() -> None:
    splits = [
        ("train", BASE / "train.csv", None),
        ("leaderboard", BASE / "leaderboard_test_features.csv", BASE / "leaderboard_test_target.csv"),
        ("hidden", BASE / "hidden_test.csv", None),
    ]
    rows = []
    for split, path, litho in splits:
        print(f"loading {split} ...", flush=True)
        df = load_split(split, path, litho)
        print(f"  {len(df):,} rows, {df['WELL'].nunique()} wells", flush=True)
        for well, g in df.groupby("WELL", sort=False):
            rows.append(well_stats(g, split))
        del df

    out = pd.DataFrame(rows)
    out = out.sort_values(["overall", "well"], ascending=[False, True]).reset_index(drop=True)
    cols = ["well", "split", "depth_from", "depth_to", "n_samples"] + CORE + ["extras", "overall"]
    out = out[cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out)} wells)")
    print(out.to_string(index=False))
    print("\n--- summary ---")
    print(out.groupby("split").size().to_string())
    print("overall: min", out["overall"].min(), "median", out["overall"].median(), "max", out["overall"].max())
    print("most complete:")
    print(out.head(8).to_string(index=False))
    print("least complete:")
    print(out.tail(8).to_string(index=False))
    print("mean % by curve:")
    for c in CORE:
        print(f"  {c:5s}  mean={out[c].mean():5.1f}  wells>0={(out[c]>0).sum()}  wells>=90={(out[c]>=90).sum()}")
    print("15/9 wells:")
    print(out[out["well"].str.startswith("15/9")].to_string(index=False))


if __name__ == "__main__":
    main()
