#!/usr/bin/env python3
"""Per-well completeness for downloaded NLOG LAS files.

Merges all LAS in a well folder onto a 0.1 m MD grid (feet converted with 0.3048).
Logged interval: first–last grid depth where any of GR/RDEP/RHOB/DTC is valid.
Coverage: % of grid samples in that interval with a valid value nearby (within
half the source step, minimum 0.15 m). Overall = mean of 9 core fractions (1 d.p.).
"""
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
import os as _os
_DATA = Path(_os.environ.get("FH2026_DATA", Path.home() / "fh2026-cache" / "data"))
_WORK = Path(_os.environ.get("FH2026_WORK", Path.home() / "fh2026-cache" / "work"))

import lasio
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = _DATA / "nlog"
OUT_CSV = ROOT / "nlog_completeness.csv"
OUT_MAP = ROOT / "mnemonic_mapping_applied.json"
GRID = 0.1  # metres
FT = 0.3048
SENTINELS = np.array([-999.25, -999.0, -9999.0, 999.25, -999.25, -99999.0, 9999.0])
CORE = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]
INTERVAL = ["GR", "RDEP", "RHOB", "DTC"]
def is_las_filename(name: str) -> bool:
    low = name.lower()
    if low.endswith(".las"):
        return True
    # NLOG ASCII LAS extracts often use .a1, .a2, ... .a56
    stem, _, ext = low.rpartition(".")
    return ext.startswith("a") and ext[1:].isdigit()

# Ordered mnemonic lists (first listed is preferred on a coverage tie).
# Compared case-insensitively after replacing - with _ and stripping :N duplicates.
MNEMONICS: dict[str, list[str]] = {
    "GR": ["GR", "GRD", "GRR", "GRS", "GRSL", "SGRD", "ECGR", "GAM", "GR_EDTC"],
    "RDEP": ["RDEP", "RD", "LLD", "HLLD", "ILD", "RILD", "DILD", "RT", "RESD", "AT90", "AHT90",
             "AIT90", "IDPH", "RLLD", "ILM90", "AO90", "HRLD", "THRT", "M4RX", "M4R9"],
    "RMED": ["RMED", "RM", "LLS", "HLLS", "ILM", "RILM", "DILM", "AT30", "AHT30", "AIT30",
             "IMPH", "RLLS", "AO30", "HRLS", "M4R3", "M4R2", "RS"],
    "CALI": ["CALI", "HCAL", "LCAL", "CAL", "CALF", "CAL1", "C1", "CAL2", "C2", "DCAL"],
    "RHOB": ["RHOB", "RHOZ", "RHOI", "DEN", "DENS", "ZDEN", "LDEN", "RHO8", "ZDL", "RHOM"],
    "NPHI": ["NPHI", "NPHIL", "TNPH", "NPOR", "NEU", "PHIN", "CNC", "CNL", "CNPOR", "CNCF", "NPHI_LS"],
    "PEF": ["PEF", "PEFZ", "PEFI", "PE", "PEFS", "PEFL", "PEF8", "PEF_SL"],
    "DTC": ["DTCO", "DTC", "DT4C", "DT4P", "DT24", "DT24QI", "DT", "AC", "BCSL", "SON", "DTCR"],
    "DTS": ["DTSM", "DTS", "DTSH", "DT4S", "DTSQI", "DT2", "ACS", "DTST", "DTSF", "DTSS"],
}

# Explicit exclusions (even if they contain a substring)
EXCLUDE = {
    "GR": {"GRMAX", "GRMIN", "SGR", "CGR", "RSGR", "THGR", "POTA", "URAN", "THOR"},
    "RDEP": {"RXO", "RMIC", "RSHA", "MSFL", "RXOZ", "RXOI", "RXO8"},
    "RMED": {"RXO", "RMIC", "RSHA", "MSFL"},
    "CALI": {"BS", "BIT", "DCAL_BIT"},
    "RHOB": {"RHOFL", "RHOMA", "DRHO", "DRHOZ"},
    "NPHI": set(),
    "PEF": set(),
    "DTC": {"DTS", "DTSM", "DTSH", "DT4S", "DT2", "DTSF", "DTSS", "DTST", "DTSR", "DTSX", "DTST0"},
    "DTS": {"DTC", "DTCO", "DT4C", "DT4P", "DT", "DTCR", "DTCX", "DTSR", "DTSX", "DTST0"},
}


def norm_mnem(m: str) -> str:
    m = str(m).strip().upper().replace("-", "_")
    # lasio duplicate suffix :1 :2
    if ":" in m:
        m = m.split(":")[0]
    m = re.sub(r"_NO_\d+$", "", m)
    return m


def is_null_header(val) -> float | None:
    try:
        x = float(val)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        return None
    return None


def valid_mask(x: np.ndarray, header_null: float | None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if header_null is not None and math.isfinite(header_null):
        ok &= np.abs(x - header_null) > 1e-5
    for v in SENTINELS:
        ok &= np.abs(x - v) > 1e-5
    # insane sonic / density guards (don't mark as valid)
    return ok


def depth_to_metres(depth: np.ndarray, unit: str | None, td_m: float | None = None) -> np.ndarray:
    u = (unit or "M").strip().upper()
    d = np.asarray(depth, dtype=float)
    if u in {"F", "FT", "FEET", "FOOT"}:
        return d * FT
    # A few NLOG ASCII extracts label metres but store 0.1 m as integers (e.g. 25049 for 2504.9 m).
    finite = d[np.isfinite(d)]
    if len(finite):
        dmax = float(np.nanmax(np.abs(finite)))
        td_lim = (td_m * 1.5) if (td_m and td_m > 0) else None
        if dmax > 10000 and (dmax / 10.0) < 8000:
            if td_lim is None or dmax > td_lim:
                return d / 10.0
    return d


def file_step_m(las, depth_m: np.ndarray, depth_unit: str | None) -> float:
    step = None
    try:
        step = float(las.well.STEP.value)
    except Exception:
        step = None
    if step is None or abs(step) < 1e-9 or not math.isfinite(step):
        if len(depth_m) > 1:
            d = np.diff(np.sort(depth_m))
            d = d[np.isfinite(d) & (d > 1e-9)]
            step_m = float(np.median(d)) if len(d) else GRID
        else:
            step_m = GRID
    else:
        step_m = abs(step)
        u = (depth_unit or "M").strip().upper()
        if u in {"F", "FT", "FEET", "FOOT"}:
            step_m *= FT
    return max(step_m, 0.05)


def match_property(mnemonic: str) -> str | None:
    n = norm_mnem(mnemonic)
    # exact match in lists, first property that claims it
    # DTS vs DTC: exact list membership only (DT is DTC, not DTS)
    for prop, names in MNEMONICS.items():
        if n in EXCLUDE.get(prop, set()):
            continue
        if n in names:
            return prop
    return None


def iter_las_files(wdir: Path) -> list[Path]:
    out = []
    for p in sorted(wdir.iterdir()):
        if not p.is_file():
            continue
        if is_las_filename(p.name):
            if p.name.lower() in {"nlog_details.json", "nlog_logdocuments.json", "download_manifest.json"}:
                continue
            out.append(p)
    return out


def load_las_curves(path: Path, td_m: float | None = None) -> dict | None:
    try:
        las = lasio.read(str(path))
    except Exception as e:
        return {"path": path.name, "error": f"{type(e).__name__}: {e}"}
    try:
        df = las.df()
    except Exception as e:
        return {"path": path.name, "error": f"df: {e}"}
    if df is None or df.empty:
        return {"path": path.name, "error": "empty"}
    depth_unit = getattr(las, "index_unit", None) or (las.curves[0].unit if las.curves else "M")
    raw_depth = df.index.to_numpy(dtype=float)
    depth_m = depth_to_metres(raw_depth, depth_unit, td_m=td_m)
    depth_scaled = bool(len(depth_m) and abs(float(np.nanmax(np.abs(depth_m))) - float(np.nanmax(np.abs(raw_depth)))) > 1)
    header_null = None
    try:
        header_null = is_null_header(las.well.NULL.value)
    except Exception:
        header_null = -999.25
    step_m = file_step_m(las, depth_m, depth_unit)
    if depth_scaled:
        # header STEP is in the same inflated unit
        try:
            hdr_step = abs(float(las.well.STEP.value))
            if hdr_step and depth_unit and depth_unit.strip().upper() not in {"F","FT","FEET","FOOT"}:
                step_m = max(hdr_step / 10.0, 0.05)
        except Exception:
            pass
    units = {c.mnemonic: (c.unit or "") for c in las.curves}
    well_hdr = None
    try:
        well_hdr = las.well.WELL.value
    except Exception:
        well_hdr = None
    curves = {}
    for col in df.columns:
        prop = match_property(col)
        if not prop:
            continue
        arr = df[col].to_numpy(dtype=float)
        mask = valid_mask(arr, header_null)
        curves.setdefault(prop, []).append({
            "mnemonic": col,
            "norm": norm_mnem(col),
            "unit": units.get(col, ""),
            "n": int(len(arr)),
            "n_valid": int(mask.sum()),
            "depth_from_m": float(np.nanmin(depth_m[mask])) if mask.any() else None,
            "depth_to_m": float(np.nanmax(depth_m[mask])) if mask.any() else None,
        })
    return {
        "path": path.name,
        "rel": path.name,
        "well_hdr": well_hdr,
        "depth_unit": depth_unit,
        "depth_scaled_from_dm": depth_scaled,
        "step_m": step_m,
        "n": int(len(depth_m)),
        "depth_from_m": float(np.nanmin(depth_m)) if len(depth_m) else None,
        "depth_to_m": float(np.nanmax(depth_m)) if len(depth_m) else None,
        "curves": curves,
        "_depth_m": depth_m,
        "_df": df,
        "_null": header_null,
    }


def paint(grid: np.ndarray, depth_m: np.ndarray, values: np.ndarray, mask: np.ndarray, half: float) -> np.ndarray:
    """Nearest-sample paint: set grid points that have a valid sample within `half` metres."""
    out = np.full(grid.shape, np.nan, dtype=float)
    if not mask.any():
        return out
    d = depth_m[mask]
    v = values[mask]
    order = np.argsort(d)
    d, v = d[order], v[order]
    # searchsorted nearest
    idx = np.searchsorted(d, grid)
    idx0 = np.clip(idx - 1, 0, len(d) - 1)
    idx1 = np.clip(idx, 0, len(d) - 1)
    dist0 = np.abs(grid - d[idx0])
    dist1 = np.abs(grid - d[idx1])
    use1 = dist1 < dist0
    nearest_dist = np.where(use1, dist1, dist0)
    nearest_val = np.where(use1, v[idx1], v[idx0])
    ok = nearest_dist <= half
    out[ok] = nearest_val[ok]
    return out


def prefer_rank(prop: str, mnemonic: str) -> int:
    n = norm_mnem(mnemonic)
    try:
        return MNEMONICS[prop].index(n)
    except ValueError:
        return 99


def analyse_well(wdir: Path) -> dict:
    files = iter_las_files(wdir)
    loaded = []
    errors = []
    td_m = None
    det_path = wdir / "nlog_details.json"
    if det_path.is_file():
        try:
            _det = json.loads(det_path.read_text())
            if _det.get("endAhDepthInMeters"):
                td_m = float(_det["endAhDepthInMeters"])
        except Exception:
            td_m = None
    for p in files:
        rec = load_las_curves(p, td_m=td_m)
        if rec is None or "error" in rec and "_df" not in rec:
            errors.append(rec or {"path": p.name, "error": "unknown"})
            continue
        if rec.get("error") and "_df" not in rec:
            errors.append(rec)
            continue
        loaded.append(rec)

    # Collect candidates per property: paint later once grid is known
    # First find overall depth span of any matched curve
    spans = []
    for rec in loaded:
        df = rec["_df"]
        depth_m = rec["_depth_m"]
        header_null = rec["_null"]
        for col in df.columns:
            prop = match_property(col)
            if not prop:
                continue
            mask = valid_mask(df[col].to_numpy(dtype=float), header_null)
            if mask.any():
                spans.append((float(np.nanmin(depth_m[mask])), float(np.nanmax(depth_m[mask]))))
    if not spans:
        return {
            "well": wdir.name,
            "split": "nlog",
            "depth_from": "",
            "depth_to": "",
            "n_samples": 0,
            **{c: 0 for c in CORE},
            "extras": "no standard curves parsed; " + "; ".join(e.get("path","") + ":" + str(e.get("error")) for e in errors),
            "overall": 0.0,
            "curve_map": {},
            "files_ok": [r["path"] for r in loaded],
            "errors": errors,
        }

    g0 = math.floor(min(s[0] for s in spans) / GRID) * GRID
    g1 = math.ceil(max(s[1] for s in spans) / GRID) * GRID
    grid = np.arange(g0, g1 + GRID * 0.5, GRID)

    # For each property, paint every candidate and pick the one with most valid points
    chosen = {}
    painted = {}
    all_maps = {c: [] for c in CORE}
    for rec in loaded:
        df = rec["_df"]
        depth_m = rec["_depth_m"]
        header_null = rec["_null"]
        half = max(rec["step_m"] * 0.6, 0.15)
        for col in df.columns:
            prop = match_property(col)
            if not prop:
                continue
            vals = df[col].to_numpy(dtype=float)
            mask = valid_mask(vals, header_null)
            series = paint(grid, depth_m, vals, mask, half)
            n_valid = int(np.isfinite(series).sum())
            cand = {
                "file": rec["path"],
                "mnemonic": col,
                "unit": "",
                "n_valid_grid": n_valid,
                "rank": prefer_rank(prop, col),
                "series": series,
                "src_valid": int(mask.sum()),
                "src_from": float(np.nanmin(depth_m[mask])) if mask.any() else None,
                "src_to": float(np.nanmax(depth_m[mask])) if mask.any() else None,
                "step_m": rec["step_m"],
                "depth_unit": rec["depth_unit"],
            }
            # fill unit from load record
            for item in rec["curves"].get(prop, []):
                if item["mnemonic"] == col:
                    cand["unit"] = item["unit"]
                    break
            all_maps[prop].append(cand)

    for prop in CORE:
        cands = all_maps[prop]
        if not cands:
            chosen[prop] = None
            painted[prop] = np.full(grid.shape, np.nan)
            continue
        # Prefer the curve with most valid grid points; fill remaining gaps from
        # other files so a short deeper sonic (etc.) still contributes.
        cands_sorted = sorted(cands, key=lambda c: (-c["n_valid_grid"], c["rank"], c["file"]))
        series = np.full(grid.shape, np.nan)
        used_files = []
        for c in cands_sorted:
            fill = np.isnan(series) & np.isfinite(c["series"])
            if not fill.any():
                continue
            series[fill] = c["series"][fill]
            used_files.append(c)
        best = cands_sorted[0]
        rec = {k: best[k] for k in best if k != "series"}
        rec["also"] = [
            {"file": c["file"], "mnemonic": c["mnemonic"], "n_valid_grid": c["n_valid_grid"]}
            for c in used_files[1:]
        ]
        chosen[prop] = rec
        painted[prop] = series

    # Logged interval from interval curves
    interval_ok = np.zeros(grid.shape, dtype=bool)
    for prop in INTERVAL:
        interval_ok |= np.isfinite(painted[prop])
    if not interval_ok.any():
        extras = "no interval (no GR/RDEP/RHOB/DTC on grid)"
        row = {
            "well": wdir.name,
            "split": "nlog",
            "depth_from": "",
            "depth_to": "",
            "n_samples": 0,
            **{c: 0 for c in CORE},
            "extras": extras,
            "overall": 0.0,
            "curve_map": {p: chosen[p] for p in CORE},
            "files_ok": [r["path"] for r in loaded],
            "errors": errors,
        }
        return row

    d0 = float(grid[interval_ok][0])
    d1 = float(grid[interval_ok][-1])
    in_iv = (grid >= d0) & (grid <= d1)
    n = int(in_iv.sum())
    pcts = {}
    exact = []
    for prop in CORE:
        frac = float(np.isfinite(painted[prop][in_iv]).sum() / n) if n else 0.0
        exact.append(frac)
        pcts[prop] = int(min(100, max(0, round(100.0 * frac))))
    overall = round(100.0 * (sum(exact) / len(CORE)), 1)

    parts = []
    for prop in CORE:
        ch = chosen[prop]
        if ch:
            also = ch.get("also") or []
            also_s = ("; also " + ", ".join(f"{a['mnemonic']}@{a['file']}" for a in also)) if also else ""
            parts.append(f"{prop}={ch['mnemonic']}@{ch['file']}({ch.get('unit') or '?'}, {ch['n_valid_grid']} pts){also_s}")
        else:
            parts.append(f"{prop}=MISSING")
    if errors:
        parts.append("parse_errors=" + ",".join(e["path"] for e in errors))

    # official well name from details if present
    well_name = wdir.name
    det_path = wdir / "nlog_details.json"
    short = None
    if det_path.is_file():
        det = json.loads(det_path.read_text())
        well_name = det.get("boreholeName") or well_name
        short = det.get("shortName")

    extras = "; ".join(parts)
    if short:
        extras = f"short={short}; " + extras

    return {
        "well": well_name,
        "folder": wdir.name,
        "shortName": short,
        "split": "nlog",
        "depth_from": round(d0, 2),
        "depth_to": round(d1, 2),
        "n_samples": n,
        **pcts,
        "extras": extras,
        "overall": overall,
        "curve_map": {p: chosen[p] for p in CORE},
        "files_ok": [r["path"] for r in loaded],
        "errors": errors,
        "candidates": {
            p: [{k: c[k] for k in c if k != "series"} for c in all_maps[p]]
            for p in CORE
        },
    }


def main():
    wells = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    rows = []
    maps = []
    for wdir in wells:
        print(f"analyse {wdir.name} ...", flush=True)
        rec = analyse_well(wdir)
        rows.append(rec)
        dts = rec.get("DTS", 0)
        print(f"  overall={rec['overall']}  DTS={dts}%  {rec.get('depth_from')}–{rec.get('depth_to')} m  n={rec.get('n_samples')}", flush=True)

    # sort like FORCE: completeness desc, then well name
    rows_sorted = sorted(rows, key=lambda r: (-float(r.get("overall") or 0), str(r.get("well"))))
    cols = ["well", "split", "depth_from", "depth_to", "n_samples", *CORE, "extras", "overall"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows_sorted])
    df.to_csv(OUT_CSV, index=False)
    OUT_MAP.write_text(json.dumps({
        "mnemonic_rules": MNEMONICS,
        "exclusions": {k: sorted(v) for k, v in EXCLUDE.items()},
        "grid_m": GRID,
        "wells": [
            {
                "well": r["well"],
                "folder": r.get("folder"),
                "shortName": r.get("shortName"),
                "curve_map": r.get("curve_map"),
                "overall": r.get("overall"),
                "files_ok": r.get("files_ok"),
                "errors": r.get("errors"),
                "candidates": r.get("candidates"),
            }
            for r in rows_sorted
        ],
    }, indent=2))
    print(f"wrote {OUT_CSV} ({len(df)} wells)")
    print(df[["well", "overall", "DTS", "DTC", "GR", "RHOB", "NPHI", "PEF", "CALI", "RDEP", "RMED"]].to_string(index=False))


if __name__ == "__main__":
    main()
