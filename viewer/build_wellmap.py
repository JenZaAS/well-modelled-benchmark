#!/usr/bin/env python3
"""Build a self-contained interactive well map from one or more data roots.

Each data root is a folder containing a ``wells.json`` manifest (see
``manifest.schema.json`` and README.md). The viewer does not hard-code
FORCE, Volve, or any other dataset: add a folder, write a manifest, and
pass ``--root``. Generated HTML embeds downsampled curves and should not
be committed with the code (see .gitignore).

Example::

    python build_wellmap.py \\
        --root $FH2026_DATA/force2020 \\
        --output wellmap.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wellmaplib.coords import utm_ed50_to_wgs84
from wellmaplib.lithology import scheme_payload
from wellmaplib.logs import (
    LITH_KEY,
    STANDARD_CURVES,
    clean_numeric,
    completeness_from_frame,
    downsample_indices,
    merge_target_lithology,
    quantize_value,
    read_force_style_csv,
    series_to_list,
    try_read_las,
)
from wellmaplib.manifest import (
    discover_manifests,
    load_manifest,
    resolve_curve_map,
    resolve_files,
    well_has_dts,
)

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "viewer_template.html"
VENDOR = HERE / "vendor"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_place_from_xy(well: dict) -> tuple[float, float] | None:
    """If lat/lon missing but x/y + a known CRS is present, convert."""
    if well.get("lat") is not None and well.get("lon") is not None:
        return float(well["lat"]), float(well["lon"])
    x, y = well.get("x"), well.get("y")
    if x is None or y is None:
        return None
    crs = (well.get("xy_crs") or "").upper().replace(" ", "")
    # Known fallback used by FORCE / NCS wellheads.
    if crs in {"EPSG:23031", "EPSG:23031N", "ED50/UTM31N", "ED50UTM31N"} or crs.endswith("23031"):
        lat, lon = utm_ed50_to_wgs84(float(x), float(y), zone=31, northern=True)
        well["lat"] = round(lat, 6)
        well["lon"] = round(lon, 6)
        well["crs"] = "EPSG:4326"
        well.setdefault("coord_source", "xy_crs")
        well.setdefault("coord_transform", "EPSG:1613")
        return well["lat"], well["lon"]
    return None


def _group_key(file_rec: dict) -> str:
    return file_rec["abs_path"] + "|" + file_rec.get("format", "csv")


def load_csv_groups(manifest: dict) -> dict[str, pd.DataFrame]:
    """Read each unique CSV once, merge lithology targets, return path->frame."""
    data_root = Path(manifest["_data_root"])
    dataset = manifest["dataset"]
    default_map = resolve_curve_map(dataset, {})
    needed: dict[str, dict] = {}
    targets_for: dict[str, list[dict]] = defaultdict(list)
    wells_for: dict[str, set[str]] = defaultdict(set)

    for well in manifest["wells"]:
        files = resolve_files(dataset, well, data_root)
        curve_map = resolve_curve_map(dataset, well)
        csvs = [f for f in files if f.get("format") == "csv"]
        tgts = [f for f in files if f.get("format") == "csv-target"]
        for f in csvs:
            key = _group_key(f)
            if key not in needed:
                needed[key] = {**f, "curve_map": dict(curve_map)}
            else:
                needed[key]["curve_map"].update(curve_map)
            wells_for[key].add(well["name"])
            for t in tgts:
                targets_for[key].append(t)

    frames: dict[str, pd.DataFrame] = {}
    for key, f in needed.items():
        path = Path(f["abs_path"])
        if not path.is_file():
            print(f"warning: missing file {path}", file=sys.stderr)
            continue
        df = read_force_style_csv(
            path,
            delimiter=f.get("delimiter") or ";",
            well_column=f.get("well_column") or "WELL",
            depth_column=f.get("depth_column") or "DEPTH_MD",
            x_column=f.get("x_column") or "X_LOC",
            y_column=f.get("y_column") or "Y_LOC",
            curve_map=f["curve_map"],
            lithology_column=dataset.get("lithology_column"),
            wells=sorted(wells_for[key]),
        )
        seen = set()
        for t in targets_for[key]:
            tpath = Path(t["abs_path"])
            if not tpath.is_file() or tpath in seen:
                continue
            seen.add(tpath)
            df = merge_target_lithology(
                df,
                tpath,
                delimiter=t.get("delimiter") or ";",
                well_column=t.get("well_column") or "WELL",
                depth_column=t.get("depth_column") or "DEPTH_MD",
                lithology_column=t.get("lithology_column") or dataset.get("lithology_column") or LITHO_COL_DEFAULT,
            )
        frames[key] = df
    return frames


LITHO_COL_DEFAULT = "FORCE_2020_LITHOFACIES_LITHOLOGY"


def load_well_frame(well: dict, dataset: dict, data_root: Path, csv_frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    files = resolve_files(dataset, well, data_root)
    curve_map = resolve_curve_map(dataset, well)
    parts: list[pd.DataFrame] = []
    for f in files:
        fmt = (f.get("format") or "").lower()
        if fmt == "csv-target":
            continue
        if fmt == "csv":
            df = csv_frames.get(_group_key(f))
            if df is None:
                continue
            part = df[df["WELL"] == well["name"]].copy()
            parts.append(part)
        elif fmt == "las":
            path = Path(f["abs_path"])
            if not path.is_file():
                print(f"warning: missing LAS {path}", file=sys.stderr)
                continue
            try:
                part = try_read_las(path, curve_map)
            except ImportError:
                print("warning: lasio is not installed; skipping LAS files", file=sys.stderr)
                continue
            except Exception as exc:
                print(f"warning: could not read {path}: {exc}", file=sys.stderr)
                continue
            parts.append(part)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    if "DEPTH" in out.columns:
        out = out.sort_values("DEPTH")
    return out


def pack_well(well: dict, dataset: dict, frame: pd.DataFrame | None, max_points: int) -> dict:
    ds_id = dataset["id"]
    packed: dict[str, Any] = {
        "id": f"{ds_id}::{well['name']}",
        "name": well["name"],
        "dataset": ds_id,
        "split": well.get("split") or "",
        "lat": well.get("lat"),
        "lon": well.get("lon"),
        "coord_source": well.get("coord_source"),
        "depth_units": well.get("depth_units") or dataset.get("depth_units") or "m",
        "licence": well.get("licence") or dataset.get("licence") or "",
        "credit": well.get("credit") or dataset.get("credit") or "",
        "lithology_scheme": dataset.get("lithology_scheme") or "force2020",
        "overall": None,
        "curves": {},
        "has_dts": False,
        "d": [],
        "c": {},
        "lith": None,
    }

    if frame is None or frame.empty or "DEPTH" not in frame.columns:
        comp = well.get("completeness") or {}
        packed["overall"] = comp.get("overall")
        packed["curves"] = dict(comp.get("curves") or {})
        packed["has_dts"] = well_has_dts(well)
        return packed

    depth = clean_numeric(frame["DEPTH"])
    work = pd.DataFrame({"DEPTH": depth})
    for k in STANDARD_CURVES:
        if k in frame.columns:
            work[k] = clean_numeric(frame[k])
        else:
            work[k] = pd.NA
    if LITH_KEY in frame.columns:
        work[LITH_KEY] = clean_numeric(frame[LITH_KEY])

    work = work[work["DEPTH"].notna()].reset_index(drop=True)

    computed = completeness_from_frame(work, STANDARD_CURVES)
    given = well.get("completeness") or {}
    packed["overall"] = given.get("overall", computed["overall"])
    packed["curves"] = dict(given.get("curves") or computed["curves"])
    packed["has_dts"] = bool(packed["curves"].get("DTS", 0) and packed["curves"]["DTS"] > 0)

    idx = downsample_indices(len(work), max_points=max_points)
    slim = work.iloc[idx]
    packed["d"] = series_to_list("DEPTH", slim["DEPTH"])
    curves = {}
    for k in STANDARD_CURVES:
        arr = series_to_list(k, slim[k])
        if any(v is not None for v in arr):
            curves[k] = arr
    packed["c"] = curves
    if LITH_KEY in slim.columns:
        lith = series_to_list(LITH_KEY, slim[LITH_KEY])
        if any(v is not None for v in lith):
            packed["lith"] = lith
    return packed


def collect_payload(manifest_paths: list[Path], max_points: int) -> tuple[dict, dict]:
    coast = _load_json(VENDOR / "ncs_coast.geojson")
    datasets_out = []
    wells_out = []
    lithology = {}
    unplaced = []
    report = {
        "manifests": [],
        "placed": 0,
        "unplaced": [],
        "coord_sources": {},
        "issues": [],
    }

    for mpath in manifest_paths:
        manifest = load_manifest(mpath)
        dataset = manifest["dataset"]
        data_root = Path(manifest["_data_root"])
        report["manifests"].append(str(mpath))
        datasets_out.append(
            {
                "id": dataset["id"],
                "name": dataset.get("name"),
                "licence": dataset.get("licence"),
                "source_url": dataset.get("source_url"),
                "credit": dataset.get("credit"),
                "citation": dataset.get("citation"),
                "depth_units": dataset.get("depth_units") or "m",
            }
        )
        scheme = dataset.get("lithology_scheme") or "force2020"
        lithology[scheme] = scheme_payload(scheme)

        print(f"Reading logs for {dataset['id']} ({len(manifest['wells'])} wells)…")
        csv_frames = load_csv_groups(manifest)

        for well in manifest["wells"]:
            maybe_place_from_xy(well)
            if well.get("lat") is None or well.get("lon") is None:
                unplaced.append(
                    {
                        "name": well["name"],
                        "dataset": dataset["id"],
                        "reason": well.get("coord_note") or "no lat/lon or convertible x/y",
                    }
                )
                report["unplaced"].append(f"{dataset['id']}::{well['name']}")
            else:
                report["placed"] += 1
                src = well.get("coord_source") or "unknown"
                report["coord_sources"][src] = report["coord_sources"].get(src, 0) + 1

            frame = load_well_frame(well, dataset, data_root, csv_frames)
            if frame is None:
                report["issues"].append(f"no curve data for {dataset['id']}::{well['name']}")
            wells_out.append(pack_well(well, dataset, frame, max_points))

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_points": max_points,
        "datasets": datasets_out,
        "lithology": lithology,
        "wells": wells_out,
        "unplaced": unplaced,
        "coast": coast,
        "coord_note": (
            "Map positions are WGS84 (EPSG:4326). FORCE wells use Sodir FactPages "
            "ED50 lat/lon (EPSG:4230) converted with EPSG:1613, or FORCE X_LOC/Y_LOC "
            "assumed as ED50 / UTM zone 31N (EPSG:23031) then EPSG:1613."
        ),
    }
    return payload, report


def build_html(payload: dict, title: str) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    leaflet_css = (VENDOR / "leaflet.css").read_text(encoding="utf-8")
    leaflet_js = (VENDOR / "leaflet.js").read_text(encoding="utf-8")
    data_js = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    html = (
        template.replace("__TITLE__", title)
        .replace("__LEAFLET_CSS__", leaflet_css)
        .replace("__LEAFLET_JS__", leaflet_js)
        .replace("__WELLMAP_DATA__", data_js)
    )
    return html


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="Data root containing wells.json (repeatable). Also accepts a path to the manifest itself.",
    )
    p.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Explicit wells.json path (repeatable).",
    )
    p.add_argument("--output", type=Path, default=HERE / "wellmap.html")
    p.add_argument("--max-points", type=int, default=2000, help="Max downsampled points per well (default 2000).")
    p.add_argument("--title", default="Well completeness map")
    args = p.parse_args()

    roots = list(args.root) + list(args.manifest)
    if not roots:
        p.error("pass at least one --root or --manifest")
    manifests = discover_manifests(roots)
    # if user passed an explicit file, discover already includes it
    if not manifests:
        raise SystemExit(f"No wells.json found in: {', '.join(str(r) for r in roots)}")

    payload, report = collect_payload(manifests, max_points=args.max_points)
    html = build_html(payload, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size = args.output.stat().st_size
    print(f"Wrote {args.output} ({size / (1024*1024):.2f} MB, {len(html)} chars)")
    print(f"  manifests: {len(report['manifests'])}")
    print(f"  wells placed: {report['placed']}")
    print(f"  wells unplaced: {len(report['unplaced'])} {report['unplaced']}")
    print(f"  coordinate sources: {report['coord_sources']}")
    if report["issues"]:
        print(f"  issues ({len(report['issues'])}):")
        for issue in report["issues"][:20]:
            print(f"    - {issue}")
        if len(report["issues"]) > 20:
            print(f"    … {len(report['issues'])-20} more")
    report_path = args.output.with_suffix(".build-report.json")
    report["output"] = str(args.output)
    report["size_bytes"] = size
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
