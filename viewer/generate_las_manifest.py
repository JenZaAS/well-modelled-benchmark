#!/usr/bin/env python3
"""Stub: scan a folder of LAS files and write a skeleton wells.json.

This does not parse curve data or coordinates (an agent or later pass should
fill lat/lon or x/y+xy_crs, and adjust curve_map). The builder can then
read the LAS files with lasio once those fields are present.

Example::

    python generate_las_manifest.py --las-dir /data/volve/las \\
        --dataset-id volve --dataset-name "Volve field wells" \\
        --licence "..." --source-url "..." --credit "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wellmaplib.logs import STANDARD_CURVES

# Common mnemonic guesses; an agent should confirm per dataset.
DEFAULT_CURVE_GUESSES = {
    "GR": "GR",
    "RDEP": "RDEP",
    "RMED": "RMED",
    "CALI": "CALI",
    "RHOB": "RHOB",
    "NPHI": "NPHI",
    "PEF": "PEF",
    "DTC": "DTC",
    "DTS": "DTS",
}

LAS_SUFFIXES = {".las", ".LAS", ".las.md"}


def guess_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("  ", " ").strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--las-dir", type=Path, required=True, help="Folder of .las files (the data root)")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--licence", default="")
    p.add_argument("--source-url", default="")
    p.add_argument("--credit", default="")
    p.add_argument("--citation", default="")
    p.add_argument("--depth-units", default="m", choices=["m", "ft"])
    p.add_argument("--split", default="", help="Optional default split label for every well")
    p.add_argument("--output", type=Path, default=None, help="Default: <las-dir>/wells.json")
    args = p.parse_args()

    root = args.las_dir
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix in {".las", ".LAS"})
    if not files:
        raise SystemExit(f"No .las files under {root}")

    wells = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        well = {
            "name": guess_name(f),
            "dataset": args.dataset_id,
            "split": args.split or None,
            "lat": None,
            "lon": None,
            "crs": "EPSG:4326",
            "x": None,
            "y": None,
            "xy_crs": None,
            "coord_source": None,
            "coord_note": "TODO: set lat/lon (EPSG:4326) or x/y + xy_crs.",
            "depth_units": args.depth_units,
            "curve_map": dict(DEFAULT_CURVE_GUESSES),
            "files": [{"path": rel, "format": "las"}],
        }
        if not args.split:
            well.pop("split")
        wells.append(well)

    manifest = {
        "manifest_version": 1,
        "dataset": {
            "id": args.dataset_id,
            "name": args.dataset_name,
            "licence": args.licence,
            "source_url": args.source_url,
            "credit": args.credit,
            "citation": args.citation,
            "depth_units": args.depth_units,
            "crs": "EPSG:4326",
            "curve_map": dict(DEFAULT_CURVE_GUESSES),
        },
        "wells": wells,
    }
    out = args.output or (root / "wells.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote skeleton {out} ({len(wells)} LAS files).")
    print("Fill lat/lon (or x/y + xy_crs) and confirm curve_map before building the viewer.")


if __name__ == "__main__":
    main()
