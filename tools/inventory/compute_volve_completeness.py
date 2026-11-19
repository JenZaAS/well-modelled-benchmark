#!/usr/bin/env python3
"""Per-well curve completeness for the FH2026 Volve subset.

Reads LAS files with lasio (one file at a time). Input: $FH2026_VOLVE (default $FH2026_DATA/Volve), read-only.
Output: $FH2026_INVENTORY_OUT (default ~/fh2026-cache/work/inventory), never the repo.

Logged interval: first to last depth where ANY of GR/RDEP/RHOB/DTC is valid,
across all LAS files for that wellbore.

For each property, pick the (file, mnemonic) with the highest coverage over
that interval (that file's own depth samples in [d0, d1]). Ties prefer
measured/raw input over combined/LFP/CPI, and prefer non-duplicate (_no_1)
mnemonics.

Overall = mean of the 9 core properties (exact fractions, reported to 1 decimal).
Percentages reported as integers 0-100.
"""
from __future__ import annotations

import csv
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
import os as _os
_DATA = Path(_os.environ.get("FH2026_DATA", Path.home() / "fh2026-cache" / "data"))
_WORK = Path(_os.environ.get("FH2026_WORK", Path.home() / "fh2026-cache" / "work"))

import lasio
import numpy as np
import pandas as pd

VOLVE = Path(_os.environ.get("FH2026_VOLVE", _DATA / "Volve"))  # read-only input
OUTDIR = Path(_os.environ.get("FH2026_INVENTORY_OUT", _WORK / "inventory"))  # outside the repo

CORE = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]
INTERVAL_PROPS = ["GR", "RDEP", "RHOB", "DTC"]
EXTRAS_PROPS = ["PHIF", "PHIT", "VSH", "SW"]
ALL_PROPS = CORE + EXTRAS_PROPS

SENTINELS = {-999.25, -999.0, -9999.0, 999.25}

# Exact mnemonic (normalized) -> property. More specific names listed first
# in spirit; classification uses these sets plus a few guarded patterns.
EXACT: dict[str, str] = {}

def _add(prop: str, *names: str) -> None:
    for n in names:
        EXACT[n.upper()] = prop

_add("GR", "GR", "GRD", "GRR", "LFP_GR")
_add("RDEP", "RDEP", "RD", "LLD", "RT", "LFP_RT", "RDEEP", "HLLD", "AT90", "AHT90")
_add("RMED", "RMED", "RM", "LLS", "LFP_RMED", "HLLS", "AT30", "AHT30", "RS")
# RS is often shallow; we map it to RMED only as a last-resort medium/shallow
# resistivity when no RM/RMED/LLS exists (handled in pick). Listed here so it
# is a candidate, but ranked below true medium names.
_add("CALI", "CALI", "HCAL", "LCAL", "C1", "C2", "LFP_CALI", "CAL")
_add("RHOB", "RHOB", "DEN", "RHOZ", "DENS", "LFP_RHOB", "LFP_RHOB_LOG", "RHOB_LOG")
_add("NPHI", "NPHI", "NEU", "TNPH", "NPOR", "PHIN", "LFP_NPHI", "CNC", "CNPOR")
_add("PEF", "PEF", "PE", "PEFZ", "LFP_PEF", "LFP_PE", "PEFS", "PEFL")
_add("DTC", "DTC", "DT", "AC", "DTCO", "DT4P", "LFP_DT", "LFP_DT_LOG", "DT_LOG", "AC04")
_add("DTS", "DTS", "DTSM", "ACS", "DTSH", "DT4S", "LFP_DTS", "LFP_DTS_LOG", "DTS_LOG", "DT2", "DT2R")
_add("PHIF", "PHIF", "PHIE", "LFP_PHIE", "LFP_PHIEC", "LFP_PHIEWC")
_add("PHIT", "PHIT", "LFP_PHIT", "LFP_PHITC", "LFP_PHITWC", "PHITC", "PHITWC")
_add("VSH", "VSH", "LFP_VSH", "VSHGR", "LFP_VSHGR", "VSHDN", "LFP_VSHDN")
_add("SW", "SW", "SWT", "SWE", "LFP_SWT", "LFP_SWE", "LFP_SWTC", "LFP_SWEC", "LFP_SWTCNC", "LFP_SWECNC")

# Explicit non-maps (document why)
EXCLUDE = {
    "GRMAX", "GRMIN", "SGR", "CGR", "GR_N",
    "RSHA", "RSH", "LFP_RSH", "RMIC", "RXO", "MSFL", "RMF",
    "RHOFL", "RHOMA", "DRHO", "RHOF",
    "SWIRR", "SW_IRR",
    "LFP_VSHDRY", "LFP_VSHDRYC", "LFP_VSHDRYWC", "VSHDRY", "VSHDRYC", "VSHDRYWC",
    "BS", "BIT", "BITSIZE",
}

# Preferred mnemonic order inside a property (lower = better) for ties
PREFERRED_MN = {
    "GR": ["GR", "GRD", "LFP_GR", "GRR"],
    "RDEP": ["RDEP", "RD", "LLD", "RT", "LFP_RT"],
    "RMED": ["RMED", "RM", "LLS", "LFP_RMED", "RS"],
    "CALI": ["CALI", "HCAL", "LCAL", "LFP_CALI", "CAL", "C1", "C2"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "DENS", "LFP_RHOB_LOG", "LFP_RHOB"],
    "NPHI": ["NPHI", "TNPH", "NEU", "NPOR", "LFP_NPHI"],
    "PEF": ["PEF", "PEFZ", "PE", "LFP_PEF", "LFP_PE"],
    "DTC": ["DTC", "DT", "DTCO", "AC", "LFP_DT_LOG", "LFP_DT"],
    "DTS": ["DTS", "DTSM", "DTSH", "DT4S", "LFP_DTS_LOG", "LFP_DTS", "ACS"],
    "PHIF": ["PHIF", "PHIE", "LFP_PHIE", "LFP_PHIEC", "LFP_PHIEWC"],
    "PHIT": ["PHIT", "LFP_PHIT", "PHITC", "LFP_PHITC", "PHITWC", "LFP_PHITWC"],
    "VSH": ["VSH", "LFP_VSH", "VSHGR", "LFP_VSHGR", "VSHDN", "LFP_VSHDN"],
    "SW": ["SW", "SWT", "LFP_SWT", "SWE", "LFP_SWE"],
}

ACS_FLAG_WELLS = set()  # filled if ACS is chosen (or is the only DTS)


def normalize_mn(mn: str) -> tuple[str, bool]:
    s = (mn or "").strip().upper().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    is_dup = bool(re.search(r"_NO_?\d+$", s))
    if is_dup:
        s = re.sub(r"_NO_?\d+$", "", s)
    return s, is_dup


def classify(mn: str) -> str | None:
    """Return property name or None. mn is raw LAS mnemonic."""
    s, _ = normalize_mn(mn)
    if not s or s in {"DEPT", "DEPTH", "MD", "DEPTH_MD", "TVD", "TVDSS", "TVD_SS"}:
        return None
    if s in EXCLUDE:
        return None
    if s in EXACT:
        return EXACT[s]
    # CALI_* family, but not CALIBRATION etc.
    if re.match(r"^(LFP_)?CALI(_.+)?$", s) or re.match(r"^HCAL(_.+)?$", s):
        return "CALI"
    return None


def file_kind(path: Path) -> dict:
    name = path.name
    low = name.lower()
    kind = {
        "edited": "_edited" in low,
        "original": ".original" in low or low.endswith(".las.original"),
        "combined": "appendntr" in low,
        "lfp": bool(re.search(r"(^|[^a-z])lfp([^a-z]|$)", low)) or "lfp" in low,
        "cpi": "cpi" in low,
        "petro_out": "petro_computed_output" in low,
        "petro_in": "petro_computed_input" in low,
        "composite": "composite" in low,
        "klogh_only": "klogh" in low and "petro" not in low and "cpi" not in low and "lfp" not in low and "appendntr" not in low,
    }
    # tags for notes
    tags = []
    if kind["original"]:
        tags.append("original")
    if kind["edited"]:
        tags.append("edited")
    if kind["combined"]:
        tags.append("combined")
    if kind["lfp"]:
        tags.append("LFP")
    if kind["cpi"]:
        tags.append("CPI")
    if kind["petro_out"]:
        tags.append("PETRO_OUT")
    if kind["petro_in"]:
        tags.append("PETRO_IN")
    if kind["composite"]:
        tags.append("composite")
    kind["tags"] = tags
    kind["label"] = "+".join(tags) if tags else "raw"
    return kind


def source_penalty(path: Path, mn_norm: str, is_dup: bool, prop: str) -> int:
    """Lower is better. Applied only when coverage ties, and as a mild
    secondary sort so a 1-point coverage gain from an LFP fill does not
    automatically beat a measured curve? Actually the task says highest
    coverage wins. Penalty is tie-break only."""
    kind = file_kind(path)
    pen = 0
    if is_dup:
        pen += 80
    pref = PREFERRED_MN.get(prop, [])
    if mn_norm in pref:
        pen += pref.index(mn_norm)
    else:
        pen += 20
    # Prefer measured / input logs over computed / combined
    if kind["klogh_only"]:
        pen += 70
    if kind["lfp"]:
        pen += 25
    if kind["cpi"]:
        pen += 20
    if kind["petro_out"]:
        pen += 20
    if kind["combined"]:
        pen += 15
    if kind["original"]:
        pen += 8
    if kind["edited"]:
        pen += 4
    if kind["petro_in"]:
        pen += 3
    if kind["composite"]:
        pen += 2
    # Prefer *_LOG (often the measured log in LFP files) slightly
    if mn_norm.endswith("_LOG"):
        pen -= 1
    return pen


def pretty_well(name: str) -> str:
    return canonicalize_well(name)


def wellbore_from(path: Path, las_well: str | None) -> str:
    """Identify wellbore from filename + folder + LAS header.

    BT2 / T2 wellbores share a folder with the parent, so the filename
    (or a specific LAS WELL value) must win over the folder name.
    """
    stem = path.name
    for ext in (".las.original", ".LAS.original", ".las", ".LAS"):
        if stem.lower().endswith(ext.lower()):
            stem = stem[: -len(ext)]
            break
    blob = f"{path.parent.name}/{stem}"
    if las_well:
        blob = f"{blob} {las_well}"
    patterns = [
        r"15[\s/_-]*9[\s/_-]*19[\s/_-]*BT[\s/_-]*2",
        r"15[\s/_-]*9[\s/_-]*19[\s/_-]*SR",
        r"15[\s/_-]*9[\s/_-]*19[\s/_-]*A\b",
        r"15[\s/_-]*9[\s/_-]*19[\s/_-]*B(?!T)",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*11[\s/_-]*T[\s/_-]*2",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*11[\s/_-]*A\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*15[\s/_-]*C\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*1[\s/_-]*A\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*1[\s/_-]*B\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*10\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*14\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*11\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*4\b",
        r"15[\s/_-]*9[\s/_-]*F[\s/_-]*1(?!\d)",
        r"15[\s/_-]*9[\s/_-]*19\b",
    ]
    for pat in patterns:
        m = re.search(pat, blob, flags=re.I)
        if m:
            return canonicalize_well(m.group(0))
    if las_well and re.search(r"\d", las_well):
        return canonicalize_well(las_well)
    return canonicalize_well(path.parent.name)


def pretty_from_token(tok: str) -> str:
    t = tok.strip().upper()
    t = t.replace("_", "/").replace(" ", "")
    # 15/9/19BT2 etc after slashify of 15_9-19 BT2 wait
    return pretty_well(tok)


def normalize_well_key(name: str) -> str:
    s = (name or "").upper()
    return re.sub(r"[^A-Z0-9]", "", s)


def is_valid_array(x: np.ndarray, null_value: float | None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if null_value is not None and np.isfinite(null_value):
        ok &= np.abs(x - float(null_value)) > 1e-6
    for v in SENTINELS:
        ok &= np.abs(x - v) > 1e-6
    return ok


def las_well_name(las) -> str | None:
    for key in ("WELL", "Well", "WN", "WELL_NAME"):
        try:
            item = las.well[key]
            val = getattr(item, "value", item)
            if val is not None and str(val).strip() and str(val).strip() != "":
                return str(val).strip()
        except Exception:
            continue
    return None


def depth_and_null(las):
    # depth: first curve, or a known mnemonic
    depth = None
    for c in las.curves:
        mn, _ = normalize_mn(c.mnemonic)
        if mn in {"DEPT", "DEPTH", "MD", "DEPTH_MD"}:
            depth = np.asarray(c.data, dtype=float)
            break
    if depth is None:
        depth = np.asarray(las.index, dtype=float)
    null_value = None
    try:
        null_value = float(las.well["NULL"].value)
    except Exception:
        try:
            null_value = float(las.null)
        except Exception:
            null_value = -999.25
    return depth, null_value


def iter_las_files(root: Path):
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith(".las") or low.endswith(".las.original"):
            files.append(p)
    return files


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = iter_las_files(VOLVE)
    print(f"Found {len(files)} LAS files under {VOLVE}", flush=True)

    # well -> list of file results
    # each file result: well, path, kind, depth, n, and for each prop a list of
    # {mn, valid_mask, is_dup, is_acs}
    by_well: dict[str, list] = defaultdict(list)
    unmapped: dict[str, set] = defaultdict(set)
    mapped_seen: dict[str, set] = defaultdict(set)  # prop -> {mn (raw)}
    file_errors: list[str] = []
    file_meta = []

    for i, path in enumerate(files, 1):
        rel = path.relative_to(VOLVE)
        print(f"[{i}/{len(files)}] {rel} ...", flush=True)
        try:
            las = lasio.read(str(path), ignore_data=False, mnemonic_case="preserve")
        except Exception as e:
            msg = f"FAILED {rel}: {e}"
            print("  " + msg, flush=True)
            traceback.print_exc()
            file_errors.append(msg)
            continue
        try:
            lw = las_well_name(las)
            well = wellbore_from(path, lw)
            # Repair pretty 15/9 tokens
            well = pretty_well(well.replace("_", " "))
            # More normalization: 159-19A style already handled
            well = canonicalize_well(well)

            depth, null_value = depth_and_null(las)
            if depth is None or len(depth) == 0:
                print("  skip: no depth", flush=True)
                continue

            kind = file_kind(path)
            curves_found = []
            prop_arrays = {p: [] for p in ALL_PROPS}
            for c in las.curves:
                raw = c.mnemonic
                mn, is_dup = normalize_mn(raw)
                if mn in {"DEPT", "DEPTH", "MD", "DEPTH_MD", "TVD", "TVDSS", "TVD_SS"}:
                    continue
                prop = classify(raw)
                if prop is None:
                    unmapped[mn].add(str(rel))
                    continue
                data = np.asarray(c.data, dtype=float)
                if data.shape != depth.shape:
                    # try to skip mismatched
                    print(f"  skip curve {raw}: shape {data.shape} != depth {depth.shape}", flush=True)
                    continue
                valid = is_valid_array(data, null_value)
                prop_arrays[prop].append({
                    "mnemonic": raw,
                    "mn_norm": mn,
                    "is_dup": is_dup,
                    "is_acs": mn == "ACS",
                    "valid": valid,
                    "n_valid": int(valid.sum()),
                })
                mapped_seen[prop].add(f"{mn}")
                curves_found.append(f"{prop}:{raw}")

            rec = {
                "well": well,
                "las_well": lw,
                "path": path,
                "rel": str(rel),
                "kind": kind,
                "depth": depth,
                "null_value": null_value,
                "props": prop_arrays,
                "n": int(len(depth)),
                "dmin": float(np.nanmin(depth)) if np.isfinite(depth).any() else None,
                "dmax": float(np.nanmax(depth)) if np.isfinite(depth).any() else None,
            }
            by_well[well].append(rec)
            file_meta.append({
                "well": well,
                "las_well": lw,
                "file": str(rel),
                "kind": kind["label"],
                "n": rec["n"],
                "dmin": rec["dmin"],
                "dmax": rec["dmax"],
                "curves": ",".join(curves_found),
            })
            n_mapped = sum(len(v) for v in prop_arrays.values())
            print(f"  well={well} las_well={lw!r} n={rec['n']} range={rec['dmin']:.2f}-{rec['dmax']:.2f} mapped={n_mapped} kind={kind['label']}", flush=True)
        finally:
            del las

    print(f"\nWellbores found ({len(by_well)}): {sorted(by_well)}", flush=True)

    rows = []
    source_rows = []
    notes_global = []

    for well in sorted(by_well):
        recs = by_well[well]
        # Logged interval from any valid GR/RDEP/RHOB/DTC across files
        d0, d1 = None, None
        for rec in recs:
            depth = rec["depth"]
            mask = np.zeros(len(depth), dtype=bool)
            for prop in INTERVAL_PROPS:
                for c in rec["props"][prop]:
                    mask |= c["valid"]
            if mask.any():
                ds = depth[mask]
                lo, hi = float(np.nanmin(ds)), float(np.nanmax(ds))
                d0 = lo if d0 is None else min(d0, lo)
                d1 = hi if d1 is None else max(d1, hi)
        if d0 is None or d1 is None:
            notes_global.append(f"{well}: no GR/RDEP/RHOB/DTC interval")
            d0, d1 = 0.0, 0.0

        # Reference n_samples: file with the most depth samples in [d0,d1]
        # Prefer a non-original measured/combined file on a standard grid.
        ref = None
        ref_score = None
        for rec in recs:
            in_int = (rec["depth"] >= d0) & (rec["depth"] <= d1) & np.isfinite(rec["depth"])
            n_in = int(in_int.sum())
            kind = rec["kind"]
            # higher n_in is better; penalize original, klogh, cpi-only short files
            score = (n_in, -source_penalty(rec["path"], "GR", False, "GR"))
            if ref is None or score > ref_score:
                ref, ref_score = rec, score
        in_ref = (ref["depth"] >= d0) & (ref["depth"] <= d1) & np.isfinite(ref["depth"]) if ref else np.array([])
        n_samples = int(in_ref.sum()) if ref is not None else 0

        picked = {}
        for prop in ALL_PROPS:
            candidates = []
            for rec in recs:
                in_int = (rec["depth"] >= d0) & (rec["depth"] <= d1) & np.isfinite(rec["depth"])
                n_in = int(in_int.sum())
                if n_in == 0:
                    continue
                for c in rec["props"][prop]:
                    n_valid = int((c["valid"] & in_int).sum())
                    cov = n_valid / n_in
                    pen = source_penalty(rec["path"], c["mn_norm"], c["is_dup"], prop)
                    candidates.append({
                        "well": well,
                        "prop": prop,
                        "file": rec["rel"],
                        "kind": rec["kind"]["label"],
                        "mnemonic": c["mnemonic"],
                        "mn_norm": c["mn_norm"],
                        "is_dup": c["is_dup"],
                        "is_acs": c["is_acs"],
                        "n_in": n_in,
                        "n_valid": n_valid,
                        "cov": cov,
                        "pen": pen,
                        "tags": rec["kind"]["tags"],
                    })
            if not candidates:
                picked[prop] = None
                continue
            candidates.sort(key=lambda r: (-r["cov"], r["pen"], -r["n_valid"]))
            best = candidates[0]
            best["alternatives"] = [
                c for c in candidates[1:]
                if c["file"] != best["file"] or c["mnemonic"] != best["mnemonic"]
            ]
            picked[prop] = best
            if best["is_acs"]:
                ACS_FLAG_WELLS.add(well)

        pcts = {}
        exacts = []
        for prop in CORE:
            b = picked[prop]
            if b is None:
                pcts[prop] = 0
                exacts.append(0.0)
            else:
                p = int(round(100.0 * b["cov"]))
                pcts[prop] = min(100, max(0, p))
                exacts.append(b["cov"])
        overall = round(100.0 * (sum(exacts) / len(CORE)), 1) if exacts else 0.0

        extra_bits = []
        for prop in EXTRAS_PROPS:
            b = picked[prop]
            if b is None:
                extra_bits.append(f"{prop}=0")
            else:
                extra_bits.append(f"{prop}={min(100, max(0, int(round(100.0 * b['cov']))))}")

        notes = []
        only_special = []
        for prop in ALL_PROPS:
            b = picked[prop]
            if b is None:
                if prop in CORE:
                    notes.append(f"{prop} missing")
                continue
            # is this property present in any raw/non-edited/non-combined file?
            alts = [b] + b.get("alternatives", [])
            non_special = [
                a for a in alts
                if not any(t in a["tags"] for t in ("edited", "combined", "LFP", "CPI", "PETRO_OUT", "original"))
                and a["cov"] > 0
            ]
            # "only exists in edited or combined"
            computed_tags = {"edited", "combined", "LFP", "CPI", "PETRO_OUT", "original"}
            has_raw = any(
                a["cov"] > 0 and not any(t in a["tags"] for t in computed_tags)
                for a in alts
            )
            if not has_raw and b["cov"] > 0:
                tags = ",".join(b["tags"]) if b["tags"] else b["kind"]
                only_special.append(f"{prop} only in {tags}:{b['file']}:{b['mnemonic']}")
            if b["is_acs"]:
                notes.append(f"DTS from ACS in {b['file']} (shear-sonic assumption, unverified)")

        extras = "; ".join(extra_bits)
        if only_special:
            extras += "; " + "; ".join(only_special)
        if notes:
            extras += "; " + "; ".join(notes)
        extras += f"; ref={ref['rel']}" if ref else "; ref=NONE"

        rows.append({
            "well": well,
            "split": "volve",
            "depth_from": round(d0, 2),
            "depth_to": round(d1, 2),
            "n_samples": n_samples,
            **pcts,
            "extras": extras,
            "overall": overall,
        })
        for prop in ALL_PROPS:
            b = picked[prop]
            if b is None:
                source_rows.append({
                    "well": well, "property": prop, "pct": 0,
                    "file": "", "mnemonic": "", "kind": "",
                    "n_valid": 0, "n_in": 0, "cov": 0.0,
                })
            else:
                source_rows.append({
                    "well": well, "property": prop,
                    "pct": min(100, max(0, int(round(100.0 * b["cov"])))),
                    "file": b["file"], "mnemonic": b["mnemonic"],
                    "kind": b["kind"], "n_valid": b["n_valid"],
                    "n_in": b["n_in"], "cov": round(b["cov"], 4),
                })

    out = pd.DataFrame(rows)
    out = out.sort_values(["overall", "well"], ascending=[False, True]).reset_index(drop=True)
    cols = ["well", "split", "depth_from", "depth_to", "n_samples"] + CORE + ["extras", "overall"]
    out = out[cols]
    out_path = OUTDIR / "volve_completeness.csv"
    out.to_csv(out_path, index=False)

    src_path = OUTDIR / "volve_sources.csv"
    pd.DataFrame(source_rows).to_csv(src_path, index=False)

    meta_path = OUTDIR / "volve_file_meta.csv"
    pd.DataFrame(file_meta).to_csv(meta_path, index=False)

    # mnemonic mapping document
    mapping_md = build_mapping_md(mapped_seen, unmapped, source_rows, file_errors, notes_global)
    map_path = OUTDIR / "mnemonic_mapping.md"
    map_path.write_text(mapping_md, encoding="utf-8")

    print("\n======= volve_completeness.csv =======", flush=True)
    print(out.to_csv(index=False), flush=True)
    print("======= end csv =======", flush=True)
    print(f"Wrote {out_path} ({len(out)} wells)", flush=True)
    print(f"Wrote {src_path}, {meta_path}, {map_path}", flush=True)
    if file_errors:
        print("FILE ERRORS:", file_errors, flush=True)
    print("most complete:", out.head(5).to_string(index=False), flush=True)
    print("least complete:", out.tail(5).to_string(index=False), flush=True)


def canonicalize_well(well: str) -> str:
    """Normalise Volve wellbore names to '15/9-19 A' / '15/9-F-11 T2' form."""
    s = (well or "").strip()
    if not s:
        return s
    s = s.replace("\\", "/")
    s = re.sub(r"^(?i:NO)[\s_/-]+", "", s)
    # 15 9 / 15_9 / 15-9 / 159, optional F, into a stable prefix
    s = re.sub(
        r"(?i)^15[\s/_-]*9[\s/_-]*(F)?[\s/_-]*",
        lambda m: "15/9-F-" if m.group(1) else "15/9-",
        s,
    )
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?i)^(15/9-19)[\s\-]*BT\s*2\b", r"\1 BT2", s)
    s = re.sub(r"(?i)^(15/9-19)[\s\-]*SR\b", r"\1 SR", s)
    s = re.sub(r"(?i)^(15/9-19)[\s\-]*A\b", r"\1 A", s)
    s = re.sub(r"(?i)^(15/9-19)[\s\-]*B\b(?!T)", r"\1 B", s)
    s = re.sub(r"(?i)^(15/9-F-11)[\s\-]*T\s*2\b", r"\1 T2", s)
    s = re.sub(r"(?i)^(15/9-F-11)[\s\-]*A\b", r"\1 A", s)
    s = re.sub(r"(?i)^(15/9-F-15)[\s\-]*C\b", r"\1 C", s)
    s = re.sub(r"(?i)^(15/9-F-1)(?!\d)[\s\-]*A\b", r"\1 A", s)
    s = re.sub(r"(?i)^(15/9-F-1)(?!\d)[\s\-]*B\b", r"\1 B", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_mapping_md(mapped_seen, unmapped, source_rows, file_errors, notes_global) -> str:
    lines = []
    lines.append("# Volve mnemonic mapping (FH2026 well inventory)")
    lines.append("")
    lines.append("Coverage is computed from the LAS files themselves, not from `curves_inventory.csv` (that file lists mnemonics and depth ranges, but not valid-sample counts).")
    lines.append("")
    lines.append("For each well and property, the chosen curve is the one with the highest coverage over the well's logged interval (first to last depth where any of GR, RDEP, RHOB or DTC is valid). Ties prefer measured/raw or PETRO input files over combined (`appendNTR`), LFP and CPI files, and ignore `_no_1` duplicate curves unless they are the only option.")
    lines.append("")
    lines.append("## Property ← mnemonic rules")
    lines.append("")
    lines.append("Mnemonics are compared case-insensitively, with `-` treated as `_`. A trailing `_no_1` / `_NO_1` is stripped for matching and the curve is deprioritised.")
    lines.append("")
    lines.append("| Property | Mnemonics accepted | Notes |")
    lines.append("|---|---|---|")
    lines.append("| GR | GR, GRD, GRR, LFP_GR | Not GRMAX, GRMIN, SGR |")
    lines.append("| RDEP (deep / true res) | RDEP, RD, LLD, RT, LFP_RT, HLLD, AT90, AHT90 | Not RXO, RMIC |")
    lines.append("| RMED (medium res) | RMED, RM, LLS, LFP_RMED, HLLS, AT30, AHT30, RS | RS is used only as a fallback (often shallow). Not RSHA, LFP_RSH, RMIC |")
    lines.append("| CALI | CALI, HCAL, LCAL, CAL, C1, C2, LFP_CALI, CALI_* | Not BS (bit size) |")
    lines.append("| RHOB | RHOB, DEN, RHOZ, DENS, LFP_RHOB, LFP_RHOB_LOG | Not RHOFL, RHOMA, DRHO |")
    lines.append("| NPHI | NPHI, NEU, TNPH, NPOR, PHIN, LFP_NPHI, CNC, CNPOR | |")
    lines.append("| PEF | PEF, PE, PEFZ, PEFS, PEFL, LFP_PEF, LFP_PE | |")
    lines.append("| DTC (P slowness) | DTC, DT, AC, DTCO, DT4P, AC04, LFP_DT, LFP_DT_LOG | DT is not matched as DTS |")
    lines.append("| DTS (S slowness) | DTS, DTSM, ACS, DTSH, DT4S, DT2, LFP_DTS, LFP_DTS_LOG | ACS is mapped as DTS but flagged; this is the open F-10 question |")
    lines.append("| PHIF | PHIF, PHIE, LFP_PHIE, LFP_PHIEC, LFP_PHIEWC | Preferred order: PHIF, PHIE, then LFP_* |")
    lines.append("| PHIT | PHIT, LFP_PHIT, PHITC, LFP_PHITC, PHITWC, LFP_PHITWC | |")
    lines.append("| VSH | VSH, LFP_VSH, VSHGR, LFP_VSHGR, VSHDN, LFP_VSHDN | Not dry-clay variants (VSHDRY*) |")
    lines.append("| SW | SW, SWT, SWE, LFP_SWT, LFP_SWE, LFP_SWTC, LFP_SWEC | Not SWIRR |")
    lines.append("")
    lines.append("LFP / CPI / PETRO computed curves are included as candidates (an open question is whether LFP should later be excluded from the *target* set). If a property is only present in an `*_edited.las`, an `appendNTR` combined file, or an LFP/CPI/PETRO output file, that is noted in the `extras` column.")
    lines.append("")
    lines.append("## Mnemonics actually used (best pick per well)")
    lines.append("")
    lines.append("| Property | Mnemonics chosen |")
    lines.append("|---|---|")
    by_prop = defaultdict(set)
    for r in source_rows:
        if r["mnemonic"]:
            by_prop[r["property"]].add(f"{r['mnemonic']}")
    for prop in ALL_PROPS:
        used = ", ".join(sorted(by_prop.get(prop, []))) or "(none found)"
        lines.append(f"| {prop} | {used} |")
    lines.append("")
    lines.append("## All matched mnemonics seen in the subset")
    lines.append("")
    for prop in ALL_PROPS:
        seen = ", ".join(sorted(mapped_seen.get(prop, []))) or "(none)"
        lines.append(f"- **{prop}**: {seen}")
    lines.append("")
    lines.append("## Unmapped mnemonics (first 80, alphabetical)")
    lines.append("")
    lines.append("These were present in at least one LAS `~C` section and were not assigned to a property. Resistivity, sonic or density names in this list are worth a second look.")
    lines.append("")
    interesting = []
    boring_prefixes = ("LFP_AI", "LFP_API", "LFP_BAD", "LFP_BV", "LFP_K", "LFP_M", "LFP_PR", "LFP_VP", "LFP_VS", "LFP_YOUNG", "LFP_POIS", "LFP_SHMIN", "CARB", "COAL", "SAND", "PERF", "DELTA", "TEMP", "RW", "J", "M", "N", "PORD", "KLHC", "KLOGH", "KLOG", "BVW")
    for mn in sorted(unmapped):
        if mn in {"DEPT", "DEPTH", "MD"}:
            continue
        interesting.append(mn)
    lines.append(", ".join(interesting[:80]))
    if len(interesting) > 80:
        lines.append(f"\n… and {len(interesting) - 80} more.")
    # Highlight maybe-missed log curves
    maybe = [mn for mn in interesting if re.search(r"(GR|RES|RT|RD|RM|RHO|DEN|NPHI|NEU|PEF|PE[^A-Z]|DT|DTS|AC|CAL|SW|PHI|VSH)", mn) and not mn.startswith("LFP_AI")]
    if maybe:
        lines.append("\nPossibly relevant unmapped names: " + ", ".join(maybe[:60]))
    lines.append("")
    if ACS_FLAG_WELLS:
        lines.append("## ACS mapped as DTS")
        lines.append("")
        lines.append("Wells where the chosen DTS curve is mnemonic `ACS`: " + ", ".join(sorted(ACS_FLAG_WELLS)) + ".")
        lines.append("This follows the requested mapping (DTS/DTSM/ACS) but it is still an open question whether F-10 ACS is shear sonic.")
        lines.append("")
    if notes_global:
        lines.append("## Other notes")
        lines.append("")
        for n in notes_global:
            lines.append(f"- {n}")
        lines.append("")
    if file_errors:
        lines.append("## Files that failed to parse")
        lines.append("")
        for n in file_errors:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("## Wellbore splitting")
    lines.append("")
    lines.append("Files are grouped by wellbore, not only by folder. So `15/9-19 B` and `15/9-19 BT2` (same folder) are separate rows, and `15/9-F-11` and `15/9-F-11 T2` are separate rows if both appear in filenames or LAS headers.")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
