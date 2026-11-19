"""Load and lightly validate wells.json manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .logs import STANDARD_CURVES  # re-exported

MANIFEST_FILENAMES = ("wells.json", "wells.manifest.json")
SCHEMA_RELATIVE = Path(__file__).resolve().parent.parent / "manifest.schema.json"


def discover_manifests(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root).resolve()
        candidates = []
        if root.is_file():
            candidates.append(root)
        else:
            for name in MANIFEST_FILENAMES:
                candidates.append(root / name)
            # also accept a manifest one level down in common dataset folders
            if root.is_dir():
                for child in sorted(root.iterdir()):
                    if child.is_dir():
                        for name in MANIFEST_FILENAMES:
                            candidates.append(child / name)
        for c in candidates:
            if c.is_file():
                r = c.resolve()
                if r not in seen:
                    seen.add(r)
                    found.append(r)
    return found


def load_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_manifest(data)
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"Invalid manifest {path}: {joined}")
    data["_manifest_path"] = str(path)
    data["_data_root"] = str(path.parent)
    return data


def validate_manifest(data: Any) -> list[str]:
    """Lightweight required-field checks (the JSON Schema is the full contract)."""
    err: list[str] = []
    if not isinstance(data, dict):
        return ["manifest must be a JSON object"]
    if data.get("manifest_version") != 1:
        err.append("manifest_version must be 1")
    ds = data.get("dataset")
    if not isinstance(ds, dict):
        err.append("dataset must be an object")
    else:
        if not ds.get("id"):
            err.append("dataset.id is required")
        if not ds.get("name"):
            err.append("dataset.name is required")
    wells = data.get("wells")
    if not isinstance(wells, list) or not wells:
        err.append("wells must be a non-empty array")
        return err
    for i, w in enumerate(wells):
        prefix = f"wells[{i}]"
        if not isinstance(w, dict):
            err.append(f"{prefix} must be an object")
            continue
        if not w.get("name"):
            err.append(f"{prefix}.name is required")
        has_ll = w.get("lat") is not None and w.get("lon") is not None
        has_xy = w.get("x") is not None and w.get("y") is not None
        files = w.get("files") or (ds.get("files") if isinstance(ds, dict) else None)
        if not files:
            err.append(f"{prefix} has no files (and dataset.files is empty)")
        # coordinates may be filled later from x/y, so only warn if neither present
        if not has_ll and not has_xy:
            # allowed: builder will report as unplaced
            pass
    return err


def resolve_curve_map(dataset: dict, well: dict) -> dict[str, str]:
    merged = {}
    for src in (dataset.get("curve_map") or {}, well.get("curve_map") or {}):
        for k, v in src.items():
            if v:
                merged[str(k)] = str(v)
    # identity defaults for any missing standard keys
    for k in STANDARD_CURVES:
        merged.setdefault(k, k)
    return merged


def resolve_files(dataset: dict, well: dict, data_root: Path) -> list[dict]:
    raw = list(well.get("files") or dataset.get("files") or [])
    out = []
    for f in raw:
        item = dict(f)
        p = Path(item["path"])
        item["abs_path"] = str(p if p.is_absolute() else (data_root / p).resolve())
        out.append(item)
    return out


def well_has_dts(well: dict) -> bool:
    comp = well.get("completeness") or {}
    curves = comp.get("curves") or comp
    dts = curves.get("DTS") if isinstance(curves, dict) else None
    if dts is None:
        dts = comp.get("DTS")
    try:
        return float(dts) > 0
    except (TypeError, ValueError):
        return False
