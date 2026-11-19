#!/usr/bin/env python3
"""Shared LAS header matching, completeness, and manifest helpers for FH2026."""
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CORE = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]
INTERVAL = ["GR", "RDEP", "RHOB", "DTC"]
MIN_SUITE = ["GR", "RHOB", "NPHI", "DTC"]  # plus resistivity (RDEP or RMED)
GRID = 0.1  # metres
FT = 0.3048
SENTINELS = np.array([-999.25, -999.0, -9999.0, 999.25, -99999.0, 9999.0, -999.99, -32768.0])

# Ordered lists: first listed is preferred on a coverage tie.
MNEMONICS: dict[str, list[str]] = {
    "GR": [
        "GR", "GRD", "GRR", "GRS", "GRSL", "SGRD", "ECGR", "GR_EDTC", "GR_TMG",
        "TGR", "HSGR", "GRN", "GAMMA", "GRGC", "GRC", "GAM", "GR_SPLICE",
        "GAMMA_RAY_SPLICE", "GR_PROCESS",
    ],
    "RDEP": [
        "RDEP", "RD", "LLD", "HLLD", "ILD", "RILD", "DILD", "RT", "RESD",
        "AT90", "AHT90", "AIT90", "IDPH", "RLLD", "AO90", "HRLD", "THRT",
        "M4RX", "M4R9", "AILD", "TBIT90", "ATCO90", "AT60", "AF60", "AF90",
        "LLD_SPLICE", "LN", "64N", "64IN", "R64", "RLL", "IL",
    ],
    "RMED": [
        "RMED", "RM", "LLS", "HLLS", "ILM", "RILM", "DILM", "AT30", "AHT30",
        "AIT30", "IMPH", "RLLS", "AO30", "HRLS", "M4R3", "M4R2", "TBIT30",
        "TBIT20", "AF30", "LLM_SPLICE", "LLS_SPLICE", "SN", "16N", "16IN",
        "R16", "AILM", "RS",
    ],
    "CALI": [
        "CALI", "HCAL", "LCAL", "CAL", "CALF", "CAL1", "C1", "CAL2", "C2",
        "DCAL", "CALX", "CALY", "HDAR",
    ],
    "RHOB": [
        "RHOB", "RHOZ", "RHOI", "DEN", "DENS", "ZDEN", "LDEN", "RHO8", "ZDL",
        "RHOM", "RHOC", "RHO", "RHOB_SPLICE", "DENB",
    ],
    "NPHI": [
        "NPHI", "NPHIL", "TNPH", "NPOR", "NEU", "PHIN", "CNC", "CNL", "CNPOR",
        "CNCF", "NPHI_LS", "NPHS", "NPHL", "SNP", "CNLS", "NPLS", "NPSS",
        "NEUTRON_SPLICE", "NEUT",
    ],
    "PEF": ["PEF", "PEFZ", "PEFI", "PE", "PEFS", "PEFL", "PEF8", "PEF_SL", "LPE"],
    "DTC": [
        "DTCO", "DTC", "DT4C", "DT4P", "DT24", "DT24QI", "DT", "AC", "BCSL",
        "SON", "DTCR", "DTL", "DTLN", "DTLF", "BHC", "DTP", "DTCO_MH",
    ],
    "DTS": [
        "DTSM", "DTS", "DTSH", "DT4S", "DTSQI", "DT2", "ACS", "DTST", "DTSF",
        "DTSS", "DTRS", "DTSH_YD", "DTSH_MH", "DTSH_FAST", "DTSH_SLOW",
        "DT2R", "DTSH1", "DTS1",
    ],
}

EXCLUDE = {
    "GR": {"GRMAX", "GRMIN", "THGR", "POTA", "URAN", "THOR", "SGR", "CGR", "HFK", "HTHO", "HURA"},
    "RDEP": {"RXO", "RMIC", "RSHA", "MSFL", "RXOZ", "RXOI", "RXO8", "RMUD"},
    "RMED": {"RXO", "RMIC", "RSHA", "MSFL", "RMUD"},
    "CALI": {"BS", "BIT", "BITSIZE", "BS_RM"},
    "RHOB": {"RHOFL", "RHOMA", "DRHO", "DRHOZ", "BAD_RHOB", "HDRA", "DPHI", "RHOB_PROCESSING", "RHOL"},
    "NPHI": {"DPHI"},
    "PEF": set(),
    "DTC": {"DTS", "DTSM", "DTSH", "DT4S", "DT2", "DTSF", "DTSS", "DTST", "DTSR", "DTSX"},
    "DTS": {"DTC", "DTCO", "DT4C", "DT4P", "DT", "DTCR", "DTCX", "DTCO_MH"},
}

DEPTH_MNems = {"DEPT", "DEPTH", "MD", "TDEP", "DEPTH_MD", "MEASURERD", "MEASUREDDEPTH"}


def norm_mnem(m: str) -> str:
    m = str(m).strip().upper().replace("-", "_").replace(" ", "_")
    if ":" in m:
        m = m.split(":")[0]
    m = re.sub(r"_NO_\d+$", "", m)
    return m


def match_property(mnemonic: str) -> str | None:
    n = norm_mnem(mnemonic)
    if n in DEPTH_MNems:
        return None
    for prop, names in MNEMONICS.items():
        if n in EXCLUDE.get(prop, set()):
            continue
        if n in names:
            return prop
    return None


def has_min_suite(props: set[str]) -> bool:
    if not set(MIN_SUITE).issubset(props):
        return False
    return "RDEP" in props or "RMED" in props


def parse_latlon_token(val: str | None) -> float | None:
    if not val:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"degrees", "deg", "none", "null"}:
        return None
    hem = None
    m = re.search(r"\b([NSEW])\b", s, flags=re.I)
    if m:
        hem = m.group(1).upper()
    # decimal with optional hemisphere / "degrees"
    m = re.search(r"(-?\d+\.\d+)", s)
    # dms: 40° 15' 9.184" or 40 15 9.184
    dms = re.search(
        r"(-?\d+)\s*[°\s]\s*(\d+)\s*['’]?\s*([\d.]+)?\s*[\"”]?",
        s,
    )
    if "°" in s or "'" in s or "’" in s:
        if dms and (dms.group(2) is not None):
            deg = float(dms.group(1))
            minu = float(dms.group(2))
            sec = float(dms.group(3) or 0)
            # incomplete like "15' 9.184\" N" (no degrees) — reject
            if re.match(r"^\s*\d+\s*['’]", s):
                return None
            valf = abs(deg) + minu / 60.0 + sec / 3600.0
            if deg < 0:
                valf = -valf
        elif m:
            valf = float(m.group(1))
        else:
            return None
    elif m:
        valf = float(m.group(1))
    else:
        try:
            valf = float(s.split()[0])
        except (ValueError, IndexError):
            return None
    if hem in {"S", "W"} and valf > 0:
        valf = -valf
    if not math.isfinite(valf):
        return None
    return valf


def parse_header_bytes(data: bytes) -> dict:
    text = data.decode("latin-1", errors="replace")
    m = re.search(r"^~A", text, flags=re.I | re.M)
    if m:
        text = text[: m.start()]
    well: dict[str, str] = {}
    curves: list[tuple[str, str]] = []
    section = None
    for line in text.splitlines():
        raw = line.rstrip("\n")
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            key = s[1:].split()[0].upper() if s[1:].split() else ""
            if key.startswith("W"):
                section = "W"
            elif key.startswith("C"):
                section = "C"
            elif key.startswith("A"):
                break
            else:
                section = None
            continue
        if section not in {"W", "C"}:
            continue
        if "." in s:
            mnem = s.split(".", 1)[0].strip()
            rest = s.split(".", 1)[1]
            unit = rest.split()[0] if rest.split() else ""
        else:
            mnem = s.split()[0] if s.split() else ""
            unit = ""
        if section == "W":
            if ":" in raw:
                left = raw.split(":", 1)[0]
                after = left.split(".", 1)[1] if "." in left else left
                parts = after.split(None, 1)
                val = parts[1].strip() if len(parts) > 1 else ""
            else:
                val = ""
            well[norm_mnem(mnem)] = val
        else:
            curves.append((mnem, unit))
    props: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for mnem, unit in curves:
        if norm_mnem(mnem) in DEPTH_MNems:
            continue
        p = match_property(mnem)
        if p:
            props.setdefault(p, []).append(mnem)
        else:
            unmapped.append(mnem)
    return {"well": well, "curves": curves, "props": props, "unmapped": unmapped}


def header_null(las) -> float | None:
    try:
        x = float(las.well.NULL.value)
        if math.isfinite(x):
            return x
    except Exception:
        return -999.25
    return -999.25


def depth_to_metres(depth: np.ndarray, unit: str | None) -> np.ndarray:
    u = (unit or "M").strip().upper()
    d = np.asarray(depth, dtype=float)
    if u in {"F", "FT", "FEET", "FOOT"}:
        return d * FT
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


def valid_mask(x: np.ndarray, header_null: float | None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if header_null is not None and math.isfinite(header_null):
        ok &= np.abs(x - header_null) > 1e-5
    for v in SENTINELS:
        ok &= np.abs(x - v) > 1e-5
    return ok


def load_las_curves(path: Path) -> dict:
    try:
        las = lasio.read(str(path), ignore_header_errors=True)
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
    depth_m = depth_to_metres(raw_depth, depth_unit)
    hnull = header_null(las)
    step_m = file_step_m(las, depth_m, depth_unit)
    units = {c.mnemonic: (c.unit or "") for c in las.curves}
    well_hdr = None
    lat = lon = None
    try:
        well_hdr = las.well.WELL.value
    except Exception:
        well_hdr = None
    for key in ("LATI", "LAT", "LATITUDE"):
        try:
            lat = parse_latlon_token(str(las.well[key].value))
            if lat is not None:
                break
        except Exception:
            pass
    for key in ("LONG", "LON", "LNGI", "LONGITUDE"):
        try:
            lon = parse_latlon_token(str(las.well[key].value))
            if lon is not None:
                break
        except Exception:
            pass
    curves = {}
    unmapped = []
    for col in df.columns:
        prop = match_property(col)
        if not prop:
            unmapped.append(col)
            continue
        arr = df[col].to_numpy(dtype=float)
        ok = valid_mask(arr, hnull)
        if not ok.any():
            continue
        item = {
            "mnemonic": col,
            "unit": units.get(col, ""),
            "depth_m": depth_m[ok],
            "value": arr[ok],
            "step_m": step_m,
        }
        # keep best later; collect all
        curves.setdefault(prop, []).append(item)
    return {
        "path": path.as_posix(),
        "name": path.name,
        "well": well_hdr,
        "lat": lat,
        "lon": lon,
        "depth_unit": depth_unit,
        "curves": curves,
        "unmapped": unmapped,
        "error": None,
    }


def completeness_for_files(paths: list[Path]) -> dict:
    """Union files onto a 0.1 m MD grid. Same metric as the NLOG inventory."""
    loaded = []
    errors = []
    for p in paths:
        rec = load_las_curves(p)
        if rec.get("error"):
            errors.append(rec)
        else:
            loaded.append(rec)
    if not loaded:
        return {
            "depth_from": "",
            "depth_to": "",
            "n_samples": 0,
            **{c: 0 for c in CORE},
            "overall": 0.0,
            "curve_map": {},
            "curve_sources": {},
            "unmapped": [],
            "errors": errors,
            "lat": None,
            "lon": None,
            "well_hdr": None,
        }

    # gather depth range of any matched core curve
    all_d = []
    for rec in loaded:
        for prop, items in rec["curves"].items():
            for it in items:
                if len(it["depth_m"]):
                    all_d.append(it["depth_m"])
    if not all_d:
        return {
            "depth_from": "",
            "depth_to": "",
            "n_samples": 0,
            **{c: 0 for c in CORE},
            "overall": 0.0,
            "curve_map": {},
            "curve_sources": {},
            "unmapped": [],
            "errors": errors,
            "lat": None,
            "lon": None,
            "well_hdr": loaded[0].get("well"),
        }

    dmin = float(min(a.min() for a in all_d))
    dmax = float(max(a.max() for a in all_d))
    # first build per-property valid masks on a coarse union, then trim to interval
    start = math.floor(dmin / GRID) * GRID
    stop = math.ceil(dmax / GRID) * GRID
    ngrid = int(round((stop - start) / GRID)) + 1
    # safety: cap extremely long empty grids
    if ngrid > 2_000_000:
        # 200 km — should not happen
        GRID_LOCAL = (stop - start) / 1_000_000
        ngrid = 1_000_001
        grid = start + np.arange(ngrid) * GRID_LOCAL
        half = max(GRID_LOCAL, 0.15)
    else:
        grid = start + np.arange(ngrid) * GRID
        half = None

    # For each property, splat each contributing curve onto the grid; keep the
    # candidate with the most valid samples as primary.
    prop_valid = {p: np.zeros(ngrid, dtype=bool) for p in CORE}
    sources: dict[str, dict] = {}
    for rec in loaded:
        rel = Path(rec["name"])
        for prop, items in rec["curves"].items():
            if prop not in prop_valid:
                continue
            for it in items:
                d = it["depth_m"]
                step = it["step_m"]
                rad = half if half is not None else max(step * 0.5, 0.15)
                # assign each sample to nearest grid node within radius
                idx = np.round((d - start) / (grid[1] - grid[0])).astype(int)
                ok = (idx >= 0) & (idx < ngrid)
                # radius check
                if ok.any():
                    dist = np.abs(d[ok] - grid[idx[ok]])
                    ok2 = dist <= rad + 1e-9
                    ii = idx[ok][ok2]
                    prop_valid[prop][ii] = True
                n_valid = int(prop_valid[prop].sum())  # running; store this file's own count
                n_this = int(ok2.sum()) if ok.any() else 0
                rec_src = {
                    "mnemonic": it["mnemonic"],
                    "path": rec["name"],
                    "unit": it["unit"],
                    "n_valid": n_this,
                }
                cur = sources.get(prop)
                if cur is None:
                    sources[prop] = {**rec_src, "also": []}
                else:
                    # prefer more points, then earlier mnemonic list position
                    better = n_this > cur["n_valid"]
                    if better:
                        prev = {k: cur[k] for k in ("mnemonic", "path", "unit", "n_valid")}
                        sources[prop] = {**rec_src, "also": [prev] + cur.get("also", [])}
                    else:
                        cur.setdefault("also", []).append(rec_src)

    interval_mask = np.zeros(ngrid, dtype=bool)
    for p in INTERVAL:
        interval_mask |= prop_valid[p]
    if not interval_mask.any():
        pcts = {c: 0 for c in CORE}
        return {
            "depth_from": "",
            "depth_to": "",
            "n_samples": 0,
            **pcts,
            "overall": 0.0,
            "curve_map": {p: sources[p]["mnemonic"] for p in sources},
            "curve_sources": sources,
            "unmapped": sorted({u for rec in loaded for u in rec.get("unmapped", [])}),
            "errors": errors,
            "lat": next((r["lat"] for r in loaded if r.get("lat") is not None), None),
            "lon": next((r["lon"] for r in loaded if r.get("lon") is not None), None),
            "well_hdr": loaded[0].get("well"),
        }

    i0 = int(np.argmax(interval_mask))
    i1 = int(ngrid - 1 - np.argmax(interval_mask[::-1]))
    in_iv = slice(i0, i1 + 1)
    n = i1 - i0 + 1
    exact = []
    pcts = {}
    for c in CORE:
        frac = float(prop_valid[c][in_iv].sum() / n) if n else 0.0
        exact.append(frac)
        pcts[c] = min(100, max(0, int(round(100.0 * frac))))
    overall = round(100.0 * sum(exact) / len(CORE), 1)
    curve_map = {p: sources[p]["mnemonic"] for p in sources}
    lat = next((r["lat"] for r in loaded if r.get("lat") is not None), None)
    lon = next((r["lon"] for r in loaded if r.get("lon") is not None), None)
    return {
        "depth_from": round(float(grid[i0]), 2),
        "depth_to": round(float(grid[i1]), 2),
        "n_samples": int(n),
        **pcts,
        "overall": overall,
        "curve_map": curve_map,
        "curve_sources": sources,
        "unmapped": sorted({u for rec in loaded for u in rec.get("unmapped", [])}),
        "errors": errors,
        "lat": lat,
        "lon": lon,
        "well_hdr": loaded[0].get("well"),
    }


def validate_manifest(doc: dict, schema_path: Path) -> list[str]:
    import jsonschema
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(x) for x in e.path)}: {e.message}" for e in validator.iter_errors(doc)]


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["well", "split", "depth_from", "depth_to", "n_samples"] + CORE + ["extras", "overall"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].sort_values(["overall", "well"], ascending=[False, True]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
