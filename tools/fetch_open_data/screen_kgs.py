#!/usr/bin/env python3
"""Screen KGS yearly LAS zips; keep only reasonably complete wells."""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fh_logs import has_min_suite, parse_header_bytes, CORE

import os as _os
DL = Path(_os.environ.get("FH2026_DL", Path.home() / "fh2026-cache" / "downloads"))
ZIP_DIR = DL / "kgs"
INDEX = ZIP_DIR / "ks_las_files.txt"
OUT_JSON = ZIP_DIR / "kgs_header_scan.json"
YEARS = [
    "2026", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018",
    "2017", "2016", "2015", "2014", "2013", "2012", "2006_2011", "2001_2005", "1999",
]
HEADER_BYTES = 120_000


def load_index() -> dict[str, dict]:
    rows = list(csv.DictReader(open(INDEX, newline="")))
    by_fn = {}
    for r in rows:
        fn = Path(r["URL"].split("?")[0]).name.lower()
        by_fn[fn] = r
    return by_fn


def read_header_from_member(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, bytes, int] | None:
    """Return (las_filename, header_bytes, uncompressed_las_size) or None."""
    name = info.filename
    low = name.lower()
    if low.endswith(".las"):
        with z.open(info) as f:
            data = f.read(HEADER_BYTES)
        return Path(name).name, data, info.file_size
    if low.endswith(".zip"):
        import io
        raw = z.read(info)
        try:
            inner = zipfile.ZipFile(io.BytesIO(raw))
        except Exception:
            return None
        las_members = [i for i in inner.infolist() if i.filename.lower().endswith(".las") and not i.is_dir()]
        if not las_members:
            return None
        # almost always one LAS named like the outer stem
        chosen = las_members[0]
        with inner.open(chosen) as f:
            data = f.read(HEADER_BYTES)
        return Path(chosen.filename).name, data, chosen.file_size
    return None


def scan_zip(zpath: Path, index: dict) -> list[dict]:
    recs = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            got = read_header_from_member(z, info)
            if not got:
                continue
            fn, data, usize = got
            hdr = parse_header_bytes(data)
            props = {k: v for k, v in hdr["props"].items()}
            idx = index.get(fn.lower())
            recs.append({
                "zip": zpath.name,
                "zip_member": info.filename,
                "filename": fn,
                "uncompressed": usize,
                "props": {k: v for k, v in props.items()},
                "prop_set": sorted(props),
                "well_hdr": (hdr["well"] or {}).get("WELL"),
                "comp": (hdr["well"] or {}).get("COMP") or (hdr["well"] or {}).get("COMPANY"),
                "api_hdr": (hdr["well"] or {}).get("API") or (hdr["well"] or {}).get("APIN") or (hdr["well"] or {}).get("UWI"),
                "strt": (hdr["well"] or {}).get("STRT"),
                "stop": (hdr["well"] or {}).get("STOP"),
                "kgs_id": (idx or {}).get("KGS_ID"),
                "lat": (idx or {}).get("Latitude"),
                "lon": (idx or {}).get("Longitude"),
                "location": (idx or {}).get("Location"),
                "operator": (idx or {}).get("Operator"),
                "lease": (idx or {}).get("Lease"),
                "api": (idx or {}).get("API") or (idx or {}).get("API_NUM_NODASH"),
                "api_nodash": (idx or {}).get("API_NUM_NODASH"),
                "elev": (idx or {}).get("Elevation"),
                "depth_start": (idx or {}).get("Depth_start"),
                "depth_stop": (idx or {}).get("Depth_stop"),
                "url": (idx or {}).get("URL"),
                "in_index": bool(idx),
                "unmapped_sample": (hdr.get("unmapped") or [])[:12],
            })
    return recs


def well_key(rec: dict) -> str:
    if rec.get("kgs_id"):
        return rec["kgs_id"]
    if rec.get("api_nodash"):
        return rec["api_nodash"]
    return rec["filename"].rsplit(".", 1)[0]


def score(union: set[str]) -> tuple:
    n = sum(1 for c in CORE if c in union)
    dts = 1 if "DTS" in union else 0
    pef = 1 if "PEF" in union else 0
    cali = 1 if "CALI" in union else 0
    both_res = 1 if ("RDEP" in union and "RMED" in union) else 0
    return (dts, n, pef, cali, both_res)


def main() -> None:
    index = load_index()
    print(f"index files: {len(index)}", flush=True)
    all_recs = []
    for year in YEARS:
        zpath = ZIP_DIR / f"{year}.zip"
        if not zpath.is_file():
            print(f"MISSING {zpath}", flush=True)
            continue
        print(f"scanning {zpath.name} ({zpath.stat().st_size/1e6:.0f} MB) ...", flush=True)
        recs = scan_zip(zpath, index)
        n_min = sum(1 for r in recs if has_min_suite(set(r["prop_set"])))
        n_dts = sum(1 for r in recs if "DTS" in r["prop_set"])
        n_pef = sum(1 for r in recs if "PEF" in r["prop_set"])
        print(f"  {len(recs)} LAS  min_suite={n_min}  DTS={n_dts}  PEF={n_pef}", flush=True)
        all_recs.extend(recs)

    wells: dict[str, dict] = {}
    for r in all_recs:
        k = well_key(r)
        w = wells.setdefault(k, {
            "well_key": k,
            "kgs_id": r.get("kgs_id"),
            "api": r.get("api"),
            "api_nodash": r.get("api_nodash"),
            "lease": r.get("lease"),
            "operator": r.get("operator"),
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "location": r.get("location"),
            "files": [],
            "prop_set": set(),
        })
        w["files"].append(r)
        w["prop_set"].update(r["prop_set"])
        for field in ("api", "api_nodash", "lease", "operator", "lat", "lon", "location", "kgs_id"):
            if not w.get(field) and r.get(field):
                w[field] = r[field]

    eligible = []
    dts_even_partial = []
    for k, w in wells.items():
        ps = set(w["prop_set"])
        size = sum(f["uncompressed"] for f in w["files"])
        interval = 0.0
        try:
            # prefer index depths (feet)
            a = min(float(f["depth_start"]) for f in w["files"] if f.get("depth_start"))
            b = max(float(f["depth_stop"]) for f in w["files"] if f.get("depth_stop"))
            interval = max(0.0, b - a)
        except Exception:
            interval = 0.0
        item = {
            "well_key": k,
            "kgs_id": w.get("kgs_id"),
            "api": w.get("api"),
            "api_nodash": w.get("api_nodash"),
            "lease": w.get("lease"),
            "operator": w.get("operator"),
            "lat": w.get("lat"),
            "lon": w.get("lon"),
            "location": w.get("location"),
            "prop_set": sorted(ps),
            "n_core": sum(1 for c in CORE if c in ps),
            "has_min": has_min_suite(ps),
            "has_dts": "DTS" in ps,
            "has_pef": "PEF" in ps,
            "uncompressed": size,
            "n_files": len(w["files"]),
            "interval_ft": interval,
            "score": list(score(ps)),
            "files": [{
                "zip": f["zip"],
                "zip_member": f["zip_member"],
                "filename": f["filename"],
                "uncompressed": f["uncompressed"],
                "prop_set": f["prop_set"],
                "url": f.get("url"),
            } for f in w["files"]],
        }
        if item["has_min"]:
            eligible.append(item)
        if item["has_dts"]:
            dts_even_partial.append(item)

    eligible.sort(key=lambda x: (x["score"], x["interval_ft"], x["uncompressed"]), reverse=True)
    print(f"\nWells total={len(wells)}  min_suite={len(eligible)}  any_DTS={len(dts_even_partial)}", flush=True)
    n_dts_min = sum(1 for w in eligible if w["has_dts"])
    n_pef_min = sum(1 for w in eligible if w["has_pef"])
    n8 = sum(1 for w in eligible if w["n_core"] >= 8)
    n9 = sum(1 for w in eligible if w["n_core"] >= 9)
    sz_all = sum(w["uncompressed"] for w in eligible)
    sz_dts = sum(w["uncompressed"] for w in eligible if w["has_dts"])
    sz8 = sum(w["uncompressed"] for w in eligible if w["n_core"] >= 8)
    print(f"  eligible with DTS={n_dts_min} ({sz_dts/1e6:.1f} MB)", flush=True)
    print(f"  eligible 9-curve={n9}  8+curve={n8} ({sz8/1e6:.1f} MB)  all min ({sz_all/1e6:.1f} MB)", flush=True)
    print(f"  eligible with PEF={n_pef_min}", flush=True)

    BUDGET = 5_000_000_000  # 5.0 GB uncompressed kept LAS
    kept = []
    used = 0
    # Always keep every min-suite well that has DTS
    for w in eligible:
        if w["has_dts"]:
            kept.append(w)
            used += w["uncompressed"]
    # Then 8+ core (typically missing only DTS), then the rest, until budget
    rest = [w for w in eligible if not w["has_dts"]]
    for w in rest:
        if used + w["uncompressed"] <= BUDGET:
            kept.append(w)
            used += w["uncompressed"]
        else:
            break
    print(f"KEPT {len(kept)} wells, {sum(w['n_files'] for w in kept)} files, {used/1e6:.1f} MB uncompressed", flush=True)
    print(f"  kept DTS={sum(1 for w in kept if w['has_dts'])}  PEF={sum(1 for w in kept if w['has_pef'])}  8+={sum(1 for w in kept if w['n_core']>=8)}  9={sum(1 for w in kept if w['n_core']>=9)}", flush=True)

    # mark kept
    kept_keys = {w["well_key"] for w in kept}
    for w in eligible:
        w["kept"] = w["well_key"] in kept_keys

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # do not dump every rejected well's file list (too big); keep eligible + dts partial
    payload = {
        "n_files_scanned": len(all_recs),
        "n_wells": len(wells),
        "n_eligible": len(eligible),
        "n_kept": len(kept),
        "kept_uncompressed": used,
        "budget": BUDGET,
        "kept_wells": kept,
        "eligible_not_kept_counts": {
            "n": sum(1 for w in eligible if not w["kept"]),
            "uncompressed": sum(w["uncompressed"] for w in eligible if not w["kept"]),
        },
        "dts_not_min_suite": [w for w in dts_even_partial if not w["has_min"]],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)

    # top 15
    print("\nTop 15 kept (score, interval, size):")
    for w in kept[:15]:
        print(f"  {w['well_key']} {w.get('lease')} core={w['n_core']} DTS={w['has_dts']} PEF={w['has_pef']} "
              f"{w['interval_ft']:.0f} ft {w['uncompressed']/1e6:.2f} MB {w['prop_set']}")


if __name__ == "__main__":
    main()
