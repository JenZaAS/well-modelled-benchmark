#!/usr/bin/env python3
"""Build a wells.json manifest for the FORCE 2020 CSV dataset.

The manifest is written next to the data (default:
``<force-dir>/wells.json``), not into the code folder. Coordinates prefer
Sodir FactPages lat/lon (ED50) matched on a normalised wellbore name,
converted to WGS84 with EPSG:1613. Unmatched wells fall back to FORCE
X_LOC/Y_LOC, assumed ED50 UTM zone 31N (EPSG:23031).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os as _os
_DATA = Path(_os.environ.get("FH2026_DATA", Path.home() / "fh2026-cache" / "data"))
_WORK = Path(_os.environ.get("FH2026_WORK", Path.home() / "fh2026-cache" / "work"))

import pandas as pd

# Allow running as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wellmaplib.coords import ed50_ll_to_wgs84, utm_ed50_to_wgs84
from wellmaplib.logs import STANDARD_CURVES
from wellmaplib.names import name_aliases, normalize_well_name, parent_well_name

FORCE_CURVE_MAP = {c: c for c in STANDARD_CURVES}
LITHO_COL = "FORCE_2020_LITHOFACIES_LITHOLOGY"

SPLIT_FILES = {
    "train": {
        "csv": "train.csv",
        "target": None,
    },
    "leaderboard": {
        "csv": "leaderboard_test_features.csv",
        "target": "leaderboard_test_target.csv",
    },
    "hidden": {
        "csv": "hidden_test.csv",
        "target": None,
    },
}

LICENCE = (
    "Well log curves: Norwegian Licence for Open Government Data (NLOD 2.0). "
    "Lithofacies labels: Creative Commons Attribution 4.0 International (CC-BY-4.0)."
)
CREDIT = (
    "Lithofacies data was provided by the FORCE Machine Learning competition "
    "with well logs and seismic 2020. Well log data from the Norwegian "
    "Petroleum Directorate (now Sodir)."
)
CITATION = (
    "Bormann P., Aursand P., Dilib F., Dischington P., Manral S. (2020). "
    "2020 FORCE Machine Learning Contest."
)
SOURCE_URL = "https://zenodo.org/records/4351156"


def load_completeness(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "well" not in df.columns:
        raise SystemExit(f"Completeness CSV is missing a 'well' column: {path}")
    return df


def load_sodir_index(sodir_dir: Path) -> dict[str, dict]:
    tables = [
        "wellbore_exploration_all.csv",
        "wellbore_development_all.csv",
        "wellbore_other_all.csv",
    ]
    index: dict[str, dict] = {}
    loaded = 0
    for name in tables:
        path = sodir_dir / name
        if not path.is_file():
            continue
        df = pd.read_csv(path, low_memory=False)
        loaded += 1
        src = name.replace(".csv", "")
        for _, r in df.iterrows():
            raw = r.get("wlbWellboreName")
            if pd.isna(raw):
                continue
            rec = {
                "sodir_name": str(raw),
                "source_table": src,
                "lat_ed50": None if pd.isna(r.get("wlbNsDecDeg")) else float(r["wlbNsDecDeg"]),
                "lon_ed50": None if pd.isna(r.get("wlbEwDecDeg")) else float(r["wlbEwDecDeg"]),
                "easting": None if pd.isna(r.get("wlbEwUtm")) else float(r["wlbEwUtm"]),
                "northing": None if pd.isna(r.get("wlbNsUtm")) else float(r["wlbNsUtm"]),
                "utm_zone": None if pd.isna(r.get("wlbUtmZone")) else int(r["wlbUtmZone"]),
                "datum": None if pd.isna(r.get("wlbGeodeticDatum")) else str(r["wlbGeodeticDatum"]),
            }
            for alias in name_aliases(str(raw)):
                index.setdefault(alias, rec)
    if not loaded:
        raise SystemExit(f"No Sodir wellbore CSVs found in {sodir_dir}")
    return index


def lookup_sodir(well_name: str, index: dict[str, dict]) -> dict | None:
    for alias in name_aliases(well_name):
        if alias in index:
            return index[alias]
    return None


def median_xy(path: Path, wells: list[str], delimiter: str = ";") -> dict[str, tuple[float, float]]:
    """Median non-null X_LOC/Y_LOC per requested well."""
    usecols = ["WELL", "X_LOC", "Y_LOC"]
    df = pd.read_csv(path, sep=delimiter, usecols=usecols, low_memory=False)
    df = df[df["WELL"].isin(wells)]
    out: dict[str, tuple[float, float]] = {}
    for well, g in df.groupby("WELL"):
        x = pd.to_numeric(g["X_LOC"], errors="coerce")
        y = pd.to_numeric(g["Y_LOC"], errors="coerce")
        ok = x.notna() & y.notna()
        if not ok.any():
            continue
        out[str(well)] = (float(x[ok].median()), float(y[ok].median()))
    return out


def well_files(split: str) -> list[dict]:
    info = SPLIT_FILES[split]
    files = [
        {
            "path": info["csv"],
            "format": "csv",
            "delimiter": ";",
            "well_column": "WELL",
            "depth_column": "DEPTH_MD",
            "x_column": "X_LOC",
            "y_column": "Y_LOC",
        }
    ]
    if info["target"]:
        files.append(
            {
                "path": info["target"],
                "format": "csv-target",
                "delimiter": ";",
                "well_column": "WELL",
                "depth_column": "DEPTH_MD",
                "lithology_column": LITHO_COL,
            }
        )
    return files


def completeness_payload(row: pd.Series) -> dict:
    curves = {c: int(row[c]) for c in STANDARD_CURVES if c in row.index}
    return {
        "overall": float(row["overall"]),
        "depth_from": None if pd.isna(row.get("depth_from")) else float(row["depth_from"]),
        "depth_to": None if pd.isna(row.get("depth_to")) else float(row["depth_to"]),
        "n_samples": None if pd.isna(row.get("n_samples")) else int(row["n_samples"]),
        "curves": curves,
    }


def sodir_placement(hit: dict) -> tuple[dict, str | None]:
    """Placement fields for a Sodir hit (ED50 lat/lon -> WGS84 via EPSG:1613)."""
    note = None
    if hit["datum"] and "ED50" not in hit["datum"].upper():
        note = f"Sodir datum is {hit['datum']!r}, treated as ED50"
    lat, lon = ed50_ll_to_wgs84(hit["lat_ed50"], hit["lon_ed50"])
    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "crs": "EPSG:4326",
        "x": hit["easting"],
        "y": hit["northing"],
        "xy_crs": f"EPSG:230{hit['utm_zone']}" if hit["utm_zone"] else "EPSG:23031",
        "coord_source": "sodir_factpages",
        "coord_source_name": hit["sodir_name"],
        "coord_source_table": hit["source_table"],
        "coord_source_crs": "EPSG:4230",
        "coord_transform": "EPSG:1613",
    }, note


def place_well(
    name: str, xy: tuple[float, float] | None, sodir: dict[str, dict]
) -> tuple[dict, str, str | None]:
    """Coordinate fields, report bucket and optional note for one FORCE well.

    Sodir name match first, then FORCE X_LOC/Y_LOC as ED50 / UTM 31N.
    """
    hit = lookup_sodir(name, sodir)
    if hit and hit["lat_ed50"] is not None and hit["lon_ed50"] is not None:
        fields, note = sodir_placement(hit)
        return fields, "placed_sodir", note
    if xy is not None:
        e, n = xy
        lat, lon = utm_ed50_to_wgs84(e, n, zone=31, northern=True)
        parent = parent_well_name(name)
        parent_hit = lookup_sodir(parent, sodir) if parent else None
        note = (
            "No Sodir wellbore name match. Placed from median FORCE X_LOC/Y_LOC "
            "assuming ED50 / UTM zone 31N (EPSG:23031), then EPSG:1613 to WGS84."
        )
        if parent_hit:
            note += f" Closest Sodir wellbore (not used for coordinates): {parent_hit['sodir_name']}."
        return {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "crs": "EPSG:4326",
            "x": e,
            "y": n,
            "xy_crs": "EPSG:23031",
            "coord_source": "force_xy_utm",
            "coord_source_crs": "EPSG:23031",
            "coord_transform": "EPSG:1613",
            "coord_note": note,
        }, "placed_utm_fallback", note
    return {
        "lat": None,
        "lon": None,
        "crs": "EPSG:4326",
        "coord_source": None,
        "coord_note": "No Sodir name match and no FORCE X_LOC/Y_LOC.",
    }, "unplaced", None

def build_manifest(
    *,
    force_dir: Path,
    sodir_dir: Path,
    completeness_path: Path,
) -> tuple[dict, dict]:
    inv = load_completeness(completeness_path)
    sodir = load_sodir_index(sodir_dir)

    # Preload FORCE XY only for wells we may need (all is cheap enough vs logs).
    xy: dict[str, tuple[float, float]] = {}
    for split, info in SPLIT_FILES.items():
        wells = inv.loc[inv["split"] == split, "well"].astype(str).tolist()
        path = force_dir / info["csv"]
        if not path.is_file():
            raise SystemExit(f"Missing FORCE file: {path}")
        xy.update(median_xy(path, wells))

    wells_out = []
    report = {
        "placed_sodir": [],
        "placed_utm_fallback": [],
        "unplaced": [],
        "notes": [],
    }

    for _, row in inv.iterrows():
        name = str(row["well"])
        split = str(row["split"])
        rec: dict = {
            "name": name,
            "dataset": "force2020",
            "split": split,
            "licence": LICENCE,
            "credit": CREDIT,
            "depth_units": "m",
            "curve_map": dict(FORCE_CURVE_MAP),
            "files": well_files(split),
            "completeness": completeness_payload(row),
        }

        fields, kind, note = place_well(name, xy.get(name), sodir)
        rec.update(fields)
        report[kind].append(name)
        if note:
            report["notes"].append(f"{name}: {note}")
        wells_out.append(rec)

    manifest = {
        "manifest_version": 1,
        "dataset": {
            "id": "force2020",
            "name": "FORCE 2020 Well Logs and Lithofacies",
            "licence": LICENCE,
            "source_url": SOURCE_URL,
            "credit": CREDIT,
            "citation": CITATION,
            "depth_units": "m",
            "crs": "EPSG:4326",
            "xy_crs": "EPSG:23031",
            "curve_map": dict(FORCE_CURVE_MAP),
            "lithology_column": LITHO_COL,
            "lithology_scheme": "force2020",
        },
        "wells": wells_out,
    }
    return manifest, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force-dir",
        type=Path,
        default=_DATA / "force2020",
        help="Folder containing train.csv, leaderboard_test_features.csv, hidden_test.csv",
    )
    p.add_argument(
        "--sodir-dir",
        type=Path,
        default=_DATA / "sodir_factpages",
        help="Folder containing Sodir wellbore_*_all.csv tables",
    )
    p.add_argument(
        "--completeness",
        type=Path,
        default=_WORK / "inventory" / "force_completeness.csv",
        help="Per-well completeness CSV (from tools/inventory/compute_force_completeness.py)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output wells.json (default: <force-dir>/wells.json)",
    )
    args = p.parse_args()
    out = args.output or (args.force_dir / "wells.json")
    manifest, report = build_manifest(
        force_dir=args.force_dir,
        sodir_dir=args.sodir_dir,
        completeness_path=args.completeness,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(manifest['wells'])} wells)")
    print(f"  placed via Sodir name match: {len(report['placed_sodir'])}")
    print(f"  placed via FORCE X/Y UTM fallback: {len(report['placed_utm_fallback'])}")
    print(f"  unplaced: {len(report['unplaced'])} {report['unplaced']}")
    for note in report["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
