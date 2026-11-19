#!/usr/bin/env python3
"""Extract kept open-licence wells and write completeness, manifests, READMEs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pyproj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fh_logs import (
    CORE, completeness_for_files, has_min_suite, parse_header_bytes,
    parse_latlon_token, validate_manifest, write_csv,
)

DATA = Path(os.environ.get("FH2026_DATA", Path.home() / "fh2026-cache" / "data"))
DL = Path(os.environ.get("FH2026_DL", Path.home() / "fh2026-cache" / "downloads"))
SCHEMA = Path(__file__).resolve().parents[2] / "viewer" / "manifest.schema.json"
KGS_ZIP = DL / "kgs"
SCAN = KGS_ZIP / "kgs_header_scan.json"
FORGE_DL = DL / "forge"
FORGE78 = DL / "forge78b"
GLADE_LAS = DL / "glade" / "las"
RETRIEVAL = "19 November 2026"
WGS84 = pyproj.Transformer.from_crs(4267, 4326, always_xy=True)  # NAD27 -> WGS84
UTM12 = pyproj.Transformer.from_crs("EPSG:26912", 4326, always_xy=True)

KGS_CITATION = (
    "The source of this material is the Kansas Geological Survey website "
    "at http://www.kgs.ku.edu/. All Rights Reserved."
)


def safe(s: str, n: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (s or "").strip())
    return s.strip("_")[:n] or "well"


def file_md5(path: Path, n: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def nad27_to_wgs84(lon, lat):
    try:
        lon, lat = float(lon), float(lat)
    except (TypeError, ValueError):
        return None, None
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None, None
    x, y = WGS84.transform(lon, lat)
    return y, x  # lat, lon


def extract_member_las(z: zipfile.ZipFile, member: str, dest: Path) -> list[Path]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = []
    low = member.lower()
    if low.endswith(".las"):
        with z.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 1024 * 1024)
        written.append(dest)
    elif low.endswith(".zip"):
        inner = zipfile.ZipFile(io.BytesIO(z.read(member)))
        for info in inner.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".las"):
                continue
            outp = dest.parent / Path(info.filename).name
            if outp.exists() and outp != dest:
                outp = dest.parent / (dest.stem + "_" + Path(info.filename).name)
            with inner.open(info) as src, open(outp, "wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
            written.append(outp)
    return written


def kgs_folder(well: dict) -> str:
    api = (well.get("api") or "").strip()
    kid = well.get("kgs_id") or well["well_key"]
    lease = safe(well.get("lease") or "")
    if api:
        base = api.replace(" ", "")
    else:
        base = f"K{kid}"
    return f"{base}_{lease}" if lease else base


def extract_kansas() -> dict:
    scan = json.loads(SCAN.read_text())
    kept = {w["well_key"]: w for w in scan["kept_wells"]}
    # Always keep every DTS well, even if slightly under the min suite.
    for w in scan.get("dts_not_min_suite") or []:
        kept.setdefault(w["well_key"], w)
    wells = list(kept.values())
    root = DATA / "kansas-kgs"
    root.mkdir(parents=True, exist_ok=True)
    # copy official index
    idx = KGS_ZIP / "ks_las_files.txt"
    if idx.exists():
        shutil.copy2(idx, root / "ks_las_files.txt")

    by_zip: dict[str, list[tuple[dict, dict]]] = {}
    for w in wells:
        for f in w["files"]:
            by_zip.setdefault(f["zip"], []).append((w, f))

    extracted = {}
    for zipname, items in sorted(by_zip.items()):
        zpath = KGS_ZIP / zipname
        print(f"extract {zipname} ({len(items)} files) ...", flush=True)
        with zipfile.ZipFile(zpath) as z:
            for w, f in items:
                folder = root / kgs_folder(w)
                dest = folder / f["filename"]
                paths = extract_member_las(z, f["zip_member"], dest)
                extracted.setdefault(w["well_key"], {"well": w, "paths": []})
                extracted[w["well_key"]]["paths"].extend(paths)
        # delete year zip once extracted
        try:
            zpath.unlink()
            print(f"  deleted {zpath.name}", flush=True)
        except Exception as e:
            print(f"  could not delete {zpath}: {e}", flush=True)

    print(f"Kansas extracted {len(extracted)} wells", flush=True)
    return extracted


def organize_glade() -> dict:
    root = DATA / "glade"
    east = root / "GLADE_East"
    west = root / "GLADE_West"
    east.mkdir(parents=True, exist_ok=True)
    west.mkdir(parents=True, exist_ok=True)
    # skip image-only / MAC waveform / exact duplicate TCOM copies
    # Prefer the GW_PROD / GE_ named copies; skip bare 0512352915 duplicates,
    # image-only FMI/SRES, and MAC-60 waveform files. Keep QCOMBO_SPECTRAL_IMAGE
    # (it also carries triple-combo and shear sonic).
    skip_names = {
        "0512352915_Glade West_TCOM_R3D1_Main_V1.las",
        "0512352915_Glade West_TCOM_R3D2_Main.las",
    }
    wells = {
        "GLADE_East": {"paths": [], "api": "05-123-52916"},
        "GLADE_West": {"paths": [], "api": "05-123-52915"},
    }
    for p in sorted(GLADE_LAS.glob("*.las")):
        name = p.name
        if name in skip_names:
            continue
        low = name.lower()
        if "uncal_image" in low or "tbfmi" in low or "mac60" in low:
            continue
        if re.search(r"(^|_)image(_|$)", low) and "qcombo" not in low and "tcom" not in low and "sonic" not in low:
            continue
        dest_dir = east if re.search(r"east|_east|GE_", name, re.I) and "west" not in low else west
        if name.lower().startswith("ge_") or "east" in low:
            dest_dir = east
        elif name.lower().startswith("gw_") or "west" in low:
            dest_dir = west
        dest = dest_dir / name.replace(" ", "_")
        shutil.copy2(p, dest)
        key = "GLADE_East" if dest_dir is east else "GLADE_West"
        wells[key]["paths"].append(dest)
    print("GLADE files:", {k: [p.name for p in v["paths"]] for k, v in wells.items()}, flush=True)
    # delete bulky zip
    z = DL / "glade" / "Wireline_Logs.zip"
    if z.exists():
        z.unlink()
        print("deleted GLADE Wireline_Logs.zip", flush=True)
    return wells


def organize_forge() -> dict:
    root = DATA / "utah-forge"
    root.mkdir(parents=True, exist_ok=True)
    wells: dict[str, dict] = {}

    def add(well, src: Path, dest_name: str | None = None):
        d = root / well
        d.mkdir(parents=True, exist_ok=True)
        dest = d / (dest_name or src.name)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        wells.setdefault(well, {"paths": []})
        wells[well]["paths"].append(dest)

    # 58-32: main + one sonic (1111 and 1006 sonic are the same DSI LAS)
    add("58-32", FORGE_DL / "58-32_main.las")
    add("58-32", FORGE_DL / "58-32_sonic.las")
    # keep 1006 file only if clearly different
    a = FORGE_DL / "58-32_sonic.las"
    b = FORGE_DL / "58-32_DSI_Sonic_1006.las"
    if a.exists() and b.exists() and file_md5(a) != file_md5(b):
        add("58-32", b, "58-32_DSI_Sonic_from_GDR1006.las")
    add("Acord-1", FORGE_DL / "Acord_1-26.las", "Acord_1-26.las")
    if (FORGE_DL / "16A_Spectral.las").exists():
        add("16A-78-32", FORGE_DL / "16A_Spectral.las", "16A78-32_Spectral.las")

    # 78B and 56-32 from zips (LAS only)
    def extract_las_from_zip(zpath: Path, well: str):
        if not zpath.exists():
            return
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".las"):
                    continue
                if Path(info.filename).name.startswith("._"):
                    continue
                dest = root / well / Path(info.filename).name.replace(" ", "_")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
                wells.setdefault(well, {"paths": []})
                wells[well]["paths"].append(dest)
                print(f"  {well}: {dest.name} {dest.stat().st_size/1e6:.2f} MB", flush=True)

    extract_las_from_zip(FORGE78 / "QSLT_sonic.zip", "78B-32")
    extract_las_from_zip(FORGE78 / "wireline_5inch.zip", "78B-32")
    extract_las_from_zip(FORGE78 / "56-32_caliper.zip", "56-32")

    # aux files
    for fn in ("well_data_readme.txt", "well_location_from_earth_model.csv"):
        src = FORGE_DL / fn
        if src.exists():
            shutil.copy2(src, root / fn)

    for zp in FORGE78.glob("*.zip"):
        try:
            zp.unlink()
        except Exception:
            pass
    print("FORGE wells:", {k: [p.name for p in v["paths"]] for k, v in wells.items()}, flush=True)
    return wells


def organize_cwldb() -> dict:
    root = DATA / "usgs-cwldb"
    wells = {}
    # unique wells from table
    table = list(csv.DictReader(open(root / "CWLDB_ver4_WellDataTable.csv", newline="")))
    meta = {}
    for row in table:
        api = row["API Number"].strip()
        meta.setdefault(api, row)
    for p in sorted(root.glob("*.las")):
        # API is the 14-digit prefix of the filename
        m = re.match(r"(\d{14})", p.name.replace(" ", ""))
        api = m.group(1) if m else p.stem
        # also try first 14 chars of name
        if api == p.stem:
            api = p.name.split("_")[0]
        row = meta.get(api) or next((r for a, r in meta.items() if a in p.name or p.name.startswith(a)), {})
        lease = (row.get("Lease Name") or p.stem).strip()
        wnum = (row.get("Well Number") or "").strip()
        folder = safe(f"{api}_{lease}_{wnum}".strip("_"))
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        dest = d / p.name.replace(" ", "_")
        if dest.resolve() != p.resolve():
            shutil.move(str(p), str(dest))
        wells[folder] = {"paths": [dest], "row": row, "api": api}
    print("CWLDB wells:", list(wells), flush=True)
    return wells


def _one_well_completeness(payload: dict) -> dict:
    """Worker: compute completeness for one well."""
    sys.path.insert(0, str(Path(__file__).resolve().parent) )
    from fh_logs import completeness_for_files
    paths = [Path(p) for p in payload["paths"]]
    rec = completeness_for_files(paths)
    rec["well_key"] = payload["well_key"]
    rec["paths"] = payload["paths"]
    rec["meta"] = payload.get("meta") or {}
    return rec


def run_completeness(well_payloads: list[dict], workers: int = 8) -> list[dict]:
    rows = []
    n = len(well_payloads)
    print(f"completeness for {n} wells, {workers} workers ...", flush=True)
    if n == 0:
        return rows
    if n == 1 or workers <= 1:
        for i, p in enumerate(well_payloads, 1):
            rec = _one_well_completeness(p)
            rows.append(rec)
            print(f"  {i}/{n} {p['well_key']} overall={rec['overall']}", flush=True)
        return rows
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_well_completeness, p): p for p in well_payloads}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            rows.append(rec)
            done += 1
            if done % 25 == 0 or done == n:
                print(f"  {done}/{n} last={rec['well_key']} overall={rec['overall']}", flush=True)
    return rows


def extras_from(rec: dict, extra_bits: list[str] | None = None) -> str:
    bits = list(extra_bits or [])
    bits.append("depth_m (0.1 m grid; feet converted × 0.3048)")
    cmap = rec.get("curve_sources") or {}
    parts = []
    for prop in CORE:
        src = cmap.get(prop)
        if not src:
            parts.append(f"{prop}=MISSING")
            continue
        also = src.get("also") or []
        s = f"{prop}={src.get('mnemonic')}@{src.get('path')}({src.get('unit')}, {src.get('n_valid')} pts)"
        if also:
            s += "; also " + ", ".join(f"{a.get('mnemonic')}@{a.get('path')}" for a in also[:4])
        parts.append(s)
    bits.extend(parts)
    if rec.get("errors"):
        bits.append("read_errors=" + "; ".join(e.get("error", str(e)) for e in rec["errors"][:3]))
    return "; ".join(bits)


def csv_row(name: str, split: str, rec: dict, extra_bits=None) -> dict:
    return {
        "well": name,
        "split": split,
        "depth_from": rec.get("depth_from", ""),
        "depth_to": rec.get("depth_to", ""),
        "n_samples": rec.get("n_samples", 0),
        **{c: rec.get(c, 0) for c in CORE},
        "extras": extras_from(rec, extra_bits),
        "overall": rec.get("overall", 0.0),
    }


def file_refs(root: Path, paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        rel = p.relative_to(root).as_posix()
        out.append({"path": rel, "format": "las"})
    return out


def write_mapping(path: Path, dataset: str, recs: list[dict]) -> None:
    doc = {
        "dataset": dataset,
        "retrieval_date": RETRIEVAL,
        "standard_curves": CORE,
        "wells": [],
    }
    for rec in recs:
        doc["wells"].append({
            "name": rec["well_key"],
            "curve_map": rec.get("curve_map") or {},
            "curve_sources": rec.get("curve_sources") or {},
            "unmapped_sample": (rec.get("unmapped") or [])[:40],
            "errors": rec.get("errors") or [],
        })
    path.write_text(json.dumps(doc, indent=1, default=str))


def manifest_well(name, dataset, rec, root, paths, lat, lon, coord_source, depth_units="ft", **kw):
    w = {
        "name": name,
        "dataset": dataset,
        "split": dataset,
        "lat": lat,
        "lon": lon,
        "crs": "EPSG:4326" if lat is not None else None,
        "licence": kw.get("licence"),
        "credit": kw.get("credit"),
        "depth_units": depth_units,
        "curve_map": rec.get("curve_map") or {},
        "curve_sources": rec.get("curve_sources") or {},
        "files": file_refs(root, paths),
        "completeness": {
            "overall": rec.get("overall", 0.0),
            "depth_from": rec.get("depth_from") if rec.get("depth_from") != "" else None,
            "depth_to": rec.get("depth_to") if rec.get("depth_to") != "" else None,
            "n_samples": rec.get("n_samples", 0),
            "curves": {c: rec.get(c, 0) for c in CORE},
            "depth_unit_note": "completeness depths are metres on a 0.1 m MD grid",
        },
    }
    if coord_source:
        w["coord_source"] = coord_source
    for k, v in kw.items():
        if k not in w and v is not None:
            w[k] = v
    return {k: v for k, v in w.items() if v is not None}


def build_kansas(extracted: dict) -> None:
    root = DATA / "kansas-kgs"
    payloads = []
    for key, pack in extracted.items():
        payloads.append({
            "well_key": key,
            "paths": [str(p) for p in pack["paths"]],
            "meta": pack["well"],
        })
    recs = run_completeness(payloads, workers=8)
    rec_by = {r["well_key"]: r for r in recs}
    csv_rows = []
    mapping_recs = []
    manifest_wells = []
    for key, pack in sorted(extracted.items(), key=lambda kv: kv[1]["well"].get("lease") or kv[0]):
        rec = rec_by[key]
        meta = pack["well"]
        name = kgs_folder(meta)
        rec["well_key"] = name
        lat, lon, csrc, cnote = None, None, None, None
        if meta.get("lat") and meta.get("lon"):
            lat, lon = nad27_to_wgs84(meta["lon"], meta["lat"])
            csrc = "KGS ks_las_files.txt Latitude/Longitude, converted NAD27 (EPSG:4267) to WGS84 (EPSG:4326). KGS oil-and-gas well spots are conventionally NAD27."
            cnote = f"Published KGS values {meta['lat']}, {meta['lon']} (treated as NAD27)."
        bits = [
            f"kgs_id={meta.get('kgs_id')}",
            f"api={meta.get('api')}",
            f"lease={meta.get('lease')}",
            f"operator={meta.get('operator')}",
        ]
        csv_rows.append(csv_row(name, "kansas-kgs", rec, bits))
        mapping_recs.append(rec)
        manifest_wells.append(manifest_well(
            name, "kansas-kgs", rec, root, pack["paths"], lat, lon, csrc,
            credit="Kansas Geological Survey.",
            licence="KGS terms of use: redistribution allowed without fee with the required citation, including 'All Rights Reserved'. Data are provided as is.",
            kgs_id=meta.get("kgs_id"),
            api=meta.get("api"),
            lease=meta.get("lease"),
            operator=meta.get("operator"),
            location=meta.get("location"),
            coord_note=cnote,
        ))
    write_csv(csv_rows, root / "kansas-kgs_completeness.csv")
    write_mapping(root / "mnemonic_mapping_applied.json", "kansas-kgs", mapping_recs)
    doc = {
        "manifest_version": 1,
        "dataset": {
            "id": "kansas-kgs",
            "name": "Kansas Geological Survey LAS library (FH2026 complete-suite subset)",
            "licence": (
                "KGS terms of use (https://www.kgs.ku.edu/General/copyright.html). "
                "Information on the KGS website has been developed for public use and may be distributed or copied. "
                "The user is granted the right, without any fee or cost, to access, link to and use, publish, distribute, "
                "disseminate, transfer, or in any manner alter, modify, revise, copy, edit, or digitize this information. "
                "It is unlawful to falsely claim copyright or other rights in KGS material. The user must include an "
                f"appropriate citation crediting the source as follows: \"{KGS_CITATION}\" "
                "Keep the copyright notice intact. Do not use the KGS name in advertising without written permission. "
                "Data are provided as is, without warranty."
            ),
            "source_url": "https://www.kgs.ku.edu/PRS/Scans/Log_Summary/index.html",
            "credit": "Kansas Geological Survey.",
            "citation": KGS_CITATION,
            "depth_units": "ft",
            "crs": "EPSG:4326",
            "retrieval_date": "2026-11-19",
        },
        "wells": manifest_wells,
    }
    (root / "wells.json").write_text(json.dumps(doc, indent=1))
    errs = validate_manifest(doc, SCHEMA)
    print("kansas-kgs schema errors:", errs[:8], "n=", len(errs), flush=True)
    write_kansas_readme(root, csv_rows, scan_kept=len(extracted))
    return csv_rows


def write_kansas_readme(root: Path, csv_rows: list[dict], scan_kept: int) -> None:
    scan = json.loads(SCAN.read_text())
    n = len(csv_rows)
    dts = sum(1 for r in csv_rows if r["DTS"] > 0)
    pef = sum(1 for r in csv_rows if r["PEF"] > 0)
    full9 = sum(1 for r in csv_rows if all(r[c] > 0 for c in CORE))
    size = sum(p.stat().st_size for p in root.rglob("*.las"))
    top = sorted(csv_rows, key=lambda r: (-r["overall"], r["well"]))[:15]
    lines = [
        "# Kansas Geological Survey LAS library (FH2026 complete-suite subset)",
        "",
        "Open Kansas well logs screened for FH2026 missing-log / DTS testing. "
        "**These files are not part of any git repository** and should stay out of the public MIT project repo.",
        "",
        "- **Source:** [Kansas Geological Survey archives of LAS files](https://www.kgs.ku.edu/PRS/Scans/Log_Summary/index.html), "
        "with the well-to-file index [ks_las_files.zip](https://www.kgs.ku.edu/PRS/Ora_Archive/ks_las_files.zip) / Magellan LAS search "
        "https://www.kgs.ku.edu/Magellan/Logs/",
        f"- **Retrieval date:** {RETRIEVAL} (Europe/Oslo).",
        "- **Licence / terms:** [KGS terms of use](https://www.kgs.ku.edu/General/copyright.html). "
        "Quoted wording from that page:",
        "",
        "> Information presented on this website has been developed for public use and may be distributed or copied. "
        "The user is granted the right, without any fee or cost, to access, link to and use, publish, distribute, "
        "disseminate, transfer, or in any manner alter, modify, revise, copy, edit, or digitize this information. "
        "It is unlawful to falsely claim copyright or other rights in KGS material. The user must include an "
        "appropriate citation crediting the source of the information or citation as follows:",
        ">",
        f"> \"{KGS_CITATION}\"",
        "",
        "Also from the same page: keep the copyright notice intact; do not use the KGS name in advertising without "
        "written permission; the digital data are distributed on an \"as is\" basis, without warranty.",
        f"- **Credit / cite:** {KGS_CITATION}",
        "",
        "No KGS account was required. Yearly archive zips for 1999–2026 (and the combined 2001–2005 and 2006–2011 archives) "
        "were downloaded from the page above. Years 2020 and earlier store each LAS inside an inner zip; those were opened "
        "only to screen headers and to extract the kept wells. Curve-poor wells were not kept. Screened-out yearly zips "
        "were deleted after the kept files were extracted.",
        "",
        "## Selection",
        "",
        "The full KGS library lists about 21,780 digital LAS wells (31 December 2024 count on the Magellan page) and "
        f"{scan['n_files_scanned']:,} LAS files in the yearly archives scanned here ({scan['n_wells']:,} unique KGS well IDs).",
        "",
        "A well was eligible if the **union of all its LAS files** contained at least **GR + resistivity (deep or medium) + RHOB + NPHI + DTC**, "
        "matched with the same mnemonic rules as the NLOG / Volve inventories (plus common US names such as ILD, ILM, SN, TNPH, DTCO, DTSM). "
        "Wells with **DTS / DTSM / DTSH** were always kept, including a few DTS wells that miss only resistivity or density. "
        "Among eligible wells, preference was given to those with DTS, then to more of the 9 core curves, then PEF, caliper, both resistivities, and a longer logged interval. "
        f"All {scan['n_eligible']:,} minimum-suite wells would have been {(scan['kept_uncompressed'] + scan['eligible_not_kept_counts']['uncompressed'])/1e9:.1f} GB uncompressed, over the ~5 GB cap, "
        f"so {scan['n_kept']:,} wells were kept from that ranking (plus every DTS well). Kept LAS total just under 5.0 GB uncompressed at screening.",
        "",
        f"## Contents (this folder)",
        "",
        f"- **{n} wells**, **{sum(1 for _ in root.rglob('*.las'))} LAS files**, **{size/1e6:.1f} MB** on disk.",
        f"- Wells with any valid DTS samples: **{dts}**. Wells with any PEF: **{pef}**. Wells with all 9 core curves present: **{full9}**.",
        "- Depths in the source LAS are in feet. Completeness depths in the CSV / manifest are **metres** on a 0.1 m MD grid (feet × 0.3048).",
        "- Coordinates: KGS index Latitude/Longitude, treated as NAD27 and converted to WGS84 (EPSG:4326).",
        "- `ks_las_files.txt` is the official KGS well-to-file index (copied, not altered).",
        "",
        "### Top wells by overall completeness",
        "",
        "| Well | Overall % | DTS % | PEF % | GR | RDEP | RHOB | NPHI | DTC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in top:
        lines.append(
            f"| {r['well']} | {r['overall']} | {r['DTS']} | {r['PEF']} | {r['GR']} | {r['RDEP']} | {r['RHOB']} | {r['NPHI']} | {r['DTC']} |"
        )
    lines += [
        "",
        "## Completeness metric",
        "",
        "Same definition as the NLOG / FORCE / Volve inventories: the logged interval is the first to last depth where any of GR, RDEP, RHOB or DTC is valid. "
        "Coverage is the percentage of 0.1 m grid samples in that interval with a valid value. Overall is the mean of the nine core fractions (one decimal).",
        "",
        "## Licence caveat",
        "",
        "This is **not** a Creative Commons licence. Redistribution is allowed without a fee, but only with the exact KGS citation above, including the words **All Rights Reserved**. "
        "Do not replace these terms with a third-party CC BY-NC copy, and do not put the raw LAS files in the public MIT repo.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")


def build_glade(wells: dict) -> list[dict]:
    root = DATA / "glade"
    payloads = [{"well_key": k, "paths": [str(p) for p in v["paths"]], "meta": v} for k, v in wells.items() if v["paths"]]
    recs = run_completeness(payloads, workers=2)
    csv_rows = []
    manifest_wells = []
    for rec in recs:
        meta = rec["meta"]
        paths = [Path(p) for p in rec["paths"]]
        # coords from LAS, with a Weld County sanity check
        lat, lon = rec.get("lat"), rec.get("lon")
        cnote = "From LAS header LATI/LONG."
        if lat is None or lon is None or not (39.5 <= lat <= 41.2 and -105.5 <= lon <= -103.5):
            cnote = (
                f"LAS header lat/lon rejected or missing for a DJ Basin / Weld County well "
                f"(parsed {lat}, {lon}). Coordinates left empty rather than inventing a location. "
                f"API {meta.get('api')} (Colorado, Weld County 05-123)."
            )
            lat, lon = None, None
        csv_rows.append(csv_row(rec["well_key"], "glade", rec, [f"api={meta.get('api')}"]))
        manifest_wells.append(manifest_well(
            rec["well_key"], "glade", rec, root, paths, lat, lon,
            "LAS well-header LATI/LONG (WGS84) where the values fall in the DJ Basin / Weld County area.",
            credit="Rhodes, R., OXY USA Inc., via the DOE Geothermal Data Repository.",
            licence="Creative Commons Attribution 4.0 International (CC BY 4.0), GDR default unless otherwise noted.",
            api=meta.get("api"),
            coord_note=cnote,
        ))
    write_csv(csv_rows, root / "glade_completeness.csv")
    write_mapping(root / "mnemonic_mapping_applied.json", "glade", recs)
    licence = (
        "Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. "
        "GDR footer: “Content is available under Creative Commons Attribution 4.0 unless otherwise noted.”"
    )
    doc = {
        "manifest_version": 1,
        "dataset": {
            "id": "glade",
            "name": "GLADE East and West wireline logs (Colorado, DJ Basin)",
            "licence": licence,
            "source_url": "https://gdr.openei.org/submissions/1857",
            "credit": "Rhodes, R. (2026), OXY USA Inc. GLADE Project well logs, DOE Geothermal Data Repository submission 1857.",
            "citation": "Rhodes, R. (2026). GLADE Project: Well Log Files for GLADE East and GLADE West wells. [Data set]. Geothermal Data Repository. OXY USA Inc. https://gdr.openei.org/submissions/1857",
            "depth_units": "ft",
            "crs": "EPSG:4326",
            "retrieval_date": "2026-11-19",
        },
        "wells": manifest_wells,
    }
    (root / "wells.json").write_text(json.dumps(doc, indent=1))
    print("glade schema errors:", validate_manifest(doc, SCHEMA), flush=True)
    size = sum(p.stat().st_size for p in root.rglob("*.las"))
    nlas = sum(1 for _ in root.rglob("*.las"))
    dts = sum(1 for r in csv_rows if r["DTS"] > 0)
    full9 = sum(1 for r in csv_rows if all(r[c] > 0 for c in CORE))
    (root / "README.md").write_text(f"""# GLADE East and West wireline logs (FH2026)

Open Colorado (DJ Basin) well logs from the Geothermal Limitless Approach to Drilling Efficiencies (GLADE) project. **These files are not part of any git repository.**

- **Source:** [GDR submission 1857](https://gdr.openei.org/submissions/1857), host DOE Geothermal Data Repository (OpenEI / National Laboratory of the Rockies). Publisher: OXY USA Inc. Authors: Ryan Rhodes. DOE project EE0010444.
- **Retrieval date:** {RETRIEVAL} (Europe/Oslo).
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. The GDR record is marked publicly accessible. Site footer: “Content is available under Creative Commons Attribution 4.0 unless otherwise noted.” No account was required.
- **Credit / cite:** Rhodes, R. (2026). GLADE Project: Well Log Files for GLADE East and GLADE West wells. [Data set]. Geothermal Data Repository. OXY USA Inc. https://gdr.openei.org/submissions/1857

## What was kept

The public “Wireline Logs.zip” (402.34 MB on the GDR page; 421.89 MB downloaded) was opened and **only LAS files useful for the 9-curve suite** were kept (triple-combo, spectral/sonic, ThruBit P-and-S). Image-only FMI/SRES LAS, MAC-60 waveform/cement files, PDFs, and duplicate TCOM copies (the bare `0512352915_…` files that match the `GW_PROD_TCOM_…` copies) were not kept. The zip was deleted after extraction.

- **2 wells**, **{nlas} LAS files**, **{size/1e6:.1f} MB**.
- Wells with DTS: **{dts}**. Wells with the full 9-curve suite present: **{full9}**.
- APIs from filenames: GLADE West 05-123-52915, GLADE East 05-123-52916 (Colorado, Weld County).
- GLADE East header coordinates fall in Weld County and were kept. GLADE West LAS headers give 41.25°N, 101.74°W, which is not in the DJ Basin / Weld County; those values were **not** written as map coordinates.

Cement-bond, mud-log and temperature/pressure zips on the same GDR record were not downloaded (temperature log is not shear sonic).

## Completeness metric

Same definition as the NLOG inventory (0.1 m MD grid, feet converted to metres). Overall is the mean of the nine core curves GR, RDEP, RMED, CALI, RHOB, NPHI, PEF, DTC, DTS.

Through-bit induction mnemonics TBIT90 / TBIT30 were mapped as RDEP / RMED; GR_TMG as GR; DTSH_* / DTSM as DTS; TNPH as NPHI.
""")
    return csv_rows


def build_forge(wells: dict) -> list[dict]:
    root = DATA / "utah-forge"
    payloads = [{"well_key": k, "paths": [str(p) for p in v["paths"]], "meta": v} for k, v in wells.items() if v.get("paths")]
    recs = run_completeness(payloads, workers=2)
    # Acord-1 UTM fallback
    acord_lat = acord_lon = None
    loc = root / "well_location_from_earth_model.csv"
    if loc.exists():
        for row in csv.DictReader(open(loc, newline="")):
            if "acord" in (row.get("Well ID") or "").lower():
                x, y = float(row["x"]), float(row["y"])
                lon, lat = UTM12.transform(x, y)
                acord_lat, acord_lon = lat, lon
    csv_rows = []
    manifest_wells = []
    for rec in recs:
        name = rec["well_key"]
        paths = [Path(p) for p in rec["paths"]]
        lat, lon = rec.get("lat"), rec.get("lon")
        csrc = "LAS well-header LATI/LONG."
        cnote = None
        if name == "Acord-1" and (lat is None or lon is None) and acord_lat:
            lat, lon = acord_lat, acord_lon
            csrc = "well_location_from_earth_model.csv x,y converted from NAD83 / UTM zone 12N (EPSG:26912) to WGS84. (The CSV does not name a CRS; zone 12N is the zone that places 58-32 on its published LAS coordinates near Milford, Utah. Some FORGE documents label the same eastings as zone 11, which is incorrect.)"
        csv_rows.append(csv_row(name, "utah-forge", rec))
        manifest_wells.append(manifest_well(
            name, "utah-forge", rec, root, paths, lat, lon, csrc,
            credit="Utah FORGE / Energy and Geoscience Institute at the University of Utah, via the DOE Geothermal Data Repository.",
            licence="Creative Commons Attribution 4.0 International (CC BY 4.0), GDR default unless otherwise noted.",
            coord_note=cnote,
        ))
    write_csv(csv_rows, root / "utah-forge_completeness.csv")
    write_mapping(root / "mnemonic_mapping_applied.json", "utah-forge", recs)
    licence = (
        "Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. "
        "GDR footer: “Content is available under Creative Commons Attribution 4.0 unless otherwise noted.”"
    )
    doc = {
        "manifest_version": 1,
        "dataset": {
            "id": "utah-forge",
            "name": "Utah FORGE wireline LAS (FH2026)",
            "licence": licence,
            "source_url": "https://gdr.openei.org/",
            "credit": "Utah FORGE project, Energy and Geoscience Institute at the University of Utah, and Idaho National Laboratory (student-competition compilation), via the DOE Geothermal Data Repository.",
            "citation": (
                "Utah FORGE well logs from the DOE Geothermal Data Repository, retrieved 19 November 2026: "
                "Podgorney et al., Utah FORGE: Well Data for Student Competition, GDR 1111, https://doi.org/10.15121/1495406; "
                "Nash and Moore, Utah FORGE: Logs and Data from Deep Well 58-32 (MU-ESW1), GDR 1006, https://doi.org/10.15121/1452735; "
                "Gilmour et al., Utah FORGE: Well 16A(78)-32 Logs, GDR 1292, https://doi.org/10.15121/1777912; "
                "May and Jones, Utah FORGE: Well 16B(78)-32 Logs from Schlumberger Technologies, GDR 1531 (processed sonic / 5.875-inch LAS only), https://doi.org/10.15121/2001059; "
                "and 56-32 / 78B-32 GDR records 1295 and 1330."
            ),
            "depth_units": "ft",
            "crs": "EPSG:4326",
            "retrieval_date": "2026-11-19",
        },
        "wells": manifest_wells,
    }
    (root / "wells.json").write_text(json.dumps(doc, indent=1))
    print("utah-forge schema errors:", validate_manifest(doc, SCHEMA), flush=True)
    size = sum(p.stat().st_size for p in root.rglob("*.las"))
    nlas = sum(1 for _ in root.rglob("*.las"))
    dts = sum(1 for r in csv_rows if r["DTS"] > 0)
    full9 = sum(1 for r in csv_rows if all(r[c] > 0 for c in CORE))
    (root / "README.md").write_text(f"""# Utah FORGE wireline LAS (FH2026)

Open Utah FORGE (Milford, Beaver County) well logs from the DOE Geothermal Data Repository. **These files are not part of any git repository.**

- **Retrieval date:** {RETRIEVAL} (Europe/Oslo).
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. GDR footer: “Content is available under Creative Commons Attribution 4.0 unless otherwise noted.” Individual records used here did not state a tighter licence. No account was required.
- **Credit / cite:** Utah FORGE / Energy and Geoscience Institute at the University of Utah (and Idaho National Laboratory for the student-competition compilation of 58-32 and Acord-1), via GDR.

## Records used

| Well | GDR record | What was kept |
| --- | --- | --- |
| 58-32 (MU-ESW1) | [1111](https://gdr.openei.org/submissions/1111) (DOI [10.15121/1495406](https://doi.org/10.15121/1495406)); sonic also on [1006](https://gdr.openei.org/submissions/1006) | Main geophysical LAS + DSI sonic LAS. The 1.7 GB `58-32_logs.zip` and 450 MB dipole-waveform zip were **not** downloaded. |
| Acord-1 (Accord 1-26) | 1111 | Digitized historic geophysical LAS (no DTS). |
| 16A(78)-32 | [1292](https://gdr.openei.org/submissions/1292) | Spectral GR LAS only (5.3 MB). The 6.34 GB Schlumberger zip and 1.1–1.5 GB anisotropy / Sanvean zips were **not** downloaded (over the data cap; they likely also contain DLIS/waveforms). |
| 78B-32 | [1330](https://gdr.openei.org/submissions/1330) | QSLT P-and-S sonic LAS and the 5.875-inch hole generic LAS (QAIT / QSLT / neutron / GR). DLIS in those zips was **not** kept (and was not converted). The 4.7 GB 10.625-inch field-data zip and 6.1 GB FMI/UBI zip were not downloaded. |
| 56-32 | [1295](https://gdr.openei.org/submissions/1295) | ThruBit caliper LAS only (1.5 MB). The 8.9 GB Schlumberger zip was **not** downloaded. This well is curve-poor in this folder. |

## Contents

- **{len(csv_rows)} wells**, **{nlas} LAS files**, **{size/1e6:.1f} MB**.
- Wells with DTS: **{dts}**. Wells with all 9 core curves present (union of files): **{full9}**.
- 58-32 has the main triple-combo / array-induction / density-neutron-PEF LAS plus a separate DSI file with DTCO / DTSM, so the **union** is the full 9-curve suite.
- Depths in feet in the LAS; completeness depths are metres on a 0.1 m grid.

## Completeness metric

Same definition as the NLOG inventory.
""")
    return csv_rows


def build_cwldb(wells: dict) -> list[dict]:
    root = DATA / "usgs-cwldb"
    payloads = []
    for name, pack in wells.items():
        payloads.append({"well_key": name, "paths": [str(p) for p in pack["paths"]], "meta": pack})
    recs = run_completeness(payloads, workers=2)
    csv_rows = []
    manifest_wells = []
    for rec in recs:
        pack = rec["meta"]
        row = pack.get("row") or {}
        lat = lon = None
        csrc = None
        if row.get("Latitude (NAD27)") and row.get("Longitude (NAD27)"):
            lat, lon = nad27_to_wgs84(row["Longitude (NAD27)"], row["Latitude (NAD27)"])
            csrc = "CWLDB_ver4_WellDataTable.csv Latitude/Longitude (NAD27), converted to WGS84 (EPSG:4326)."
        paths = [Path(p) for p in rec["paths"]]
        bits = [
            f"api={pack.get('api')}",
            f"state={row.get('State Name')}",
            f"county={row.get('County or Parish Name')}",
            f"field={row.get('Field Name')}",
            f"td_ft={row.get('Total Depth (feet)')}",
            f"log_types={row.get('Geophysical Log Types')}",
        ]
        csv_rows.append(csv_row(rec["well_key"], "usgs-cwldb", rec, bits))
        manifest_wells.append(manifest_well(
            rec["well_key"], "usgs-cwldb", rec, root, paths, lat, lon, csrc,
            credit="U.S. Geological Survey, Geology, Energy & Minerals Science Center.",
            licence="CC0 1.0 Universal / US public domain (https://creativecommons.org/publicdomain/zero/1.0/; https://www.usa.gov/publicdomain/label/1.0/).",
            api=pack.get("api"),
            state=row.get("State Name"),
            county=row.get("County or Parish Name"),
            field=row.get("Field Name"),
            lease=row.get("Lease Name"),
            td_ft=row.get("Total Depth (feet)"),
        ))
    write_csv(csv_rows, root / "usgs-cwldb_completeness.csv")
    write_mapping(root / "mnemonic_mapping_applied.json", "usgs-cwldb", recs)
    licence = (
        "CC0 1.0 Universal. The USGS product page states: “This work is marked with CC0 1.0 Universal.” "
        "Science Data Catalog licence field: http://www.usa.gov/publicdomain/label/1.0/."
    )
    doc = {
        "manifest_version": 1,
        "dataset": {
            "id": "usgs-cwldb",
            "name": "USGS Gulf Coast Basin Curated Wells and Logs Database v4.0",
            "licence": licence,
            "source_url": "https://www.sciencebase.gov/catalog/item/63e25e99d34e9fa19a9b84fc",
            "credit": "Buursink, M.L., Self-Trail, J.M., Craddock, W.H., Lohr, C.D., Doolan, C.A., and Counts, J.W., 2023, U.S. Geological Survey.",
            "citation": "Buursink, M.L., Self-Trail, J.M., Craddock, W.H., Lohr, C.D., Doolan, C.A., and Counts, J.W., 2023, United States Gulf Coast Basin Curated Wells and Logs Database (ver. 4.0, September 2025): U.S. Geological Survey data release, https://doi.org/10.5066/P95BNWT3.",
            "depth_units": "ft",
            "crs": "EPSG:4326",
            "retrieval_date": "2026-11-19",
        },
        "wells": manifest_wells,
    }
    (root / "wells.json").write_text(json.dumps(doc, indent=1))
    print("usgs-cwldb schema errors:", validate_manifest(doc, SCHEMA), flush=True)
    size = sum(p.stat().st_size for p in root.rglob("*.las"))
    nlas = sum(1 for _ in root.rglob("*.las"))
    dts = sum(1 for r in csv_rows if r["DTS"] > 0)
    full9 = sum(1 for r in csv_rows if all(r[c] > 0 for c in CORE))
    (root / "README.md").write_text(f"""# USGS Gulf Coast Curated Wells and Logs Database v4.0 (FH2026)

Open onshore US Gulf Coast petroleum well logs from the USGS. **These files are not part of any git repository.**

- **Source:** [USGS product page](https://www.usgs.gov/data/united-states-gulf-coast-basin-curated-wells-and-logs-database-ver-40-september-2025); [ScienceBase item](https://www.sciencebase.gov/catalog/item/63e25e99d34e9fa19a9b84fc); DOI [10.5066/P95BNWT3](https://doi.org/10.5066/P95BNWT3).
- **Retrieval date:** {RETRIEVAL} (Europe/Oslo).
- **Licence:** CC0 1.0 Universal / US public domain. The USGS product page states: “This work is marked with CC0 1.0 Universal.” Catalogue licence: http://www.usa.gov/publicdomain/label/1.0/. This is the most permissive licence in this batch. No login was required.
- **Credit / cite:** Buursink, M.L., Self-Trail, J.M., Craddock, W.H., Lohr, C.D., Doolan, C.A., and Counts, J.W., 2023, United States Gulf Coast Basin Curated Wells and Logs Database (ver. 4.0, September 2025): U.S. Geological Survey data release, https://doi.org/10.5066/P95BNWT3.

## What was downloaded

All **9 LAS files** (about 35.7 MB) plus the well table, FGDC metadata, and revision-history text. Multi-gigabyte cuttings/core photograph zips and nannofossil / geochemistry spreadsheets were **not** downloaded.

- **{len(csv_rows)} wells**, **{nlas} LAS files**, **{size/1e6:.1f} MB**.
- Wells with DTS: **{dts}**. Wells with all 9 core curves present: **{full9}**.
- Three wells are in Texas (Foerster 1, Nueces A-1, Nicholson Ranch 1), four in Louisiana, two in Mississippi.
- These are deep petroleum wells (often TD 10,000 ft or more) but the published LAS suites are relatively **curve-poor** for this benchmark: typically GR, SP, induction/resistivity and sonic, sometimes caliper and neutron. **No DTS and almost no bulk density / PEF** were present in the LAS headers or completeness pass. Older neutron curves named NEUT (API units, not neutron porosity) were mapped as NPHI with that caveat in the mapping file.
- Coordinates in the well table are NAD27; they were converted to WGS84 for the manifest.

## Completeness metric

Same definition as the NLOG inventory (0.1 m MD grid, feet converted to metres).
""")
    return csv_rows


def main():
    print("=== organize small datasets ===", flush=True)
    glade = organize_glade()
    forge = organize_forge()
    cwldb = organize_cwldb()
    print("=== extract Kansas ===", flush=True)
    kgs = extract_kansas()
    print("disk after extract:", flush=True)
    os.system("df -h / | tail -1")
    print("=== completeness + manifests ===", flush=True)
    build_cwldb(cwldb)
    build_glade(glade)
    build_forge(forge)
    build_kansas(kgs)
    print("ALL DONE", flush=True)
    os.system(f"du -sh {DATA}/*")


if __name__ == "__main__":
    main()
