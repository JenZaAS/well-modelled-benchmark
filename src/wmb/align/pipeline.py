"""Read raw LAS files, map/merge onto a standard depth grid, write aligned wells.

No interpolation across gaps: a source sample is assigned to the nearest grid
node only when it falls within half a step. Overlapping files fill remaining
NaNs from lower-priority sources; values are never averaged or invented.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from wmb.align.mapping import (
    DEPTH_OUT,
    DEV_MNEMONICS,
    EXTRA_CURVES,
    NULL_OUT,
    TVD_MNEMONICS,
    Candidate,
    aliases_for,
    assess_candidate,
    clean_values,
    convert_depth,
    curve_spec,
    depth_flags,
    excluded,
    infer_and_convert,
    is_deprioritized_copy,
    load_align_spec,
    norm_mnem,
    pick_best,
    range_flags,
    standard_curves,
)
from wmb.curves import CORE
from wmb.io import Well

LAS_NAME = re.compile(r"\.(las|LAS)$")
FRAME_NAME = re.compile(r"\.a\d+$", re.I)
COMMON_STEPS = (0.1, 0.1524, 0.15, 0.05, 0.3048, 0.2, 0.125, 0.0762)


def discover_las(well_dir: Path) -> list[Path]:
    """LAS and LAS-like Schlumberger frame extracts (``.a1``, ``.a2``, …) in a well folder."""
    out = []
    for p in sorted(well_dir.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if LAS_NAME.search(p.name) or FRAME_NAME.search(p.name):
            if _looks_like_las(p):
                out.append(p)
    return out


def _looks_like_las(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    if not head or b"\x00" in head[:256]:
        return False
    try:
        text = head.decode("latin-1", errors="ignore")
    except Exception:
        return False
    return bool(re.search(r"^~[VvCcWwAa]", text, re.M) or "~CURVE" in text.upper() or "~VERSION" in text.upper())


_WINDOWS_NANS = (b"-1.#IND", b"+1.#IND", b"1.#IND", b"-1.#INF", b"+1.#INF", b"1.#INF",
                 b"-1.#QNAN", b"+1.#QNAN", b"1.#QNAN")


def _las_bytes_without_windows_nans(data: bytes) -> bytes | None:
    """Replace MSVC 1.#INF / 1.#IND tokens so lasio can parse the file. None if clean."""
    if not any(tok in data for tok in _WINDOWS_NANS):
        return None
    text = data.decode("latin-1")
    # Longer tokens first so "-1.#INF" is not left as a leading minus plus "1.#INF".
    for tok in ("-1.#QNAN", "+1.#QNAN", "1.#QNAN", "-1.#IND", "+1.#IND", "1.#IND",
                "-1.#INF", "+1.#INF", "1.#INF"):
        text = text.replace(tok, "-999.25")
    return text.encode("latin-1")


def _read_las(path: Path):
    import lasio
    import tempfile

    # engine='normal' is required for wrapped LAS (common in NLOG). The default
    # engine is faster but cannot read those files.
    kwargs = dict(encoding="latin-1", engine="normal")
    data = Path(path).read_bytes()
    cleaned = _las_bytes_without_windows_nans(data)
    tmp = None
    read_path = path
    try:
        if cleaned is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".las", delete=False)
            tmp.write(cleaned)
            tmp.close()
            read_path = Path(tmp.name)
        try:
            return lasio.read(str(read_path), null_policy="strict", **kwargs)
        except Exception:
            return lasio.read(str(read_path), **kwargs)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


def _curve_unit(curve) -> str:
    return (getattr(curve, "unit", None) or "") or ""


def _curve_desc(curve) -> str:
    return (getattr(curve, "descr", None) or "") or ""


def read_las_curves(path: Path) -> dict:
    """Load one LAS-like file: depth in metres plus raw curves and header bits."""
    las = _read_las(path)
    if not las.curves:
        raise ValueError(f"{path}: no curves")
    index = las.curves[0]
    depth_raw = clean_values(np.asarray(las.index, dtype=float))
    depth_m, dflags, dunit = convert_depth(depth_raw, _curve_unit(index))
    finite = depth_m[np.isfinite(depth_m)]
    if finite.size and float(np.nanmax(np.abs(finite))) > 8000:
        raise ValueError(
            f"depth index looks implausible after conversion to metres "
            f"(min={float(np.nanmin(finite)):.1f}, max={float(np.nanmax(finite)):.1f} m; "
            f"unit was {_curve_unit(index)!r})"
        )
    curves = []
    for c in las.curves[1:]:
        curves.append({
            "mnemonic": c.mnemonic,
            "unit": _curve_unit(c),
            "descr": _curve_desc(c),
            "values": clean_values(np.asarray(c.data, dtype=float)),
        })
    well_name = ""
    try:
        if "WELL" in las.well and las.well["WELL"].value:
            well_name = str(las.well["WELL"].value).strip()
    except Exception:
        well_name = ""
    return {
        "path": path,
        "well_name": well_name,
        "depth_m": depth_m,
        "depth_unit_in": _curve_unit(index),
        "depth_unit_out": dunit,
        "depth_flags": dflags + depth_flags(depth_m),
        "curves": curves,
        "n": int(depth_m.size),
    }


def choose_step(depths: list[np.ndarray]) -> float:
    """Median native spacing in metres, snapped to a common logging step when close."""
    samples = []
    for d in depths:
        x = np.asarray(d, float)
        x = np.unique(x[np.isfinite(x)])
        if x.size < 3:
            continue
        dd = np.diff(np.sort(x))
        dd = dd[dd > 1e-6]
        if dd.size:
            samples.append(float(np.median(dd)))
    if not samples:
        return 0.1
    raw = float(np.median(samples))
    raw = min(max(raw, 0.01), 0.5)
    for s in COMMON_STEPS:
        if abs(raw - s) / s <= 0.08:
            return s
    return round(raw, 4)


def build_grid(depth_min: float, depth_max: float, step: float) -> np.ndarray:
    lo = math.floor(depth_min / step) * step
    hi = math.ceil(depth_max / step) * step
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 6)


def assign_to_grid(depth_m: np.ndarray, values: np.ndarray, grid: np.ndarray, step: float) -> np.ndarray:
    """Nearest-node assignment; no interpolation. Points farther than half a step stay NaN."""
    out = np.full(grid.shape, np.nan, dtype=float)
    dist = np.full(grid.shape, np.inf, dtype=float)
    d = np.asarray(depth_m, float)
    v = np.asarray(values, float)
    mask = np.isfinite(d) & np.isfinite(v)
    if not mask.any():
        return out
    idx = np.rint((d[mask] - grid[0]) / step).astype(int)
    ok = (idx >= 0) & (idx < grid.size)
    nodes = idx[ok]
    src_d = d[mask][ok]
    src_v = v[mask][ok]
    half = 0.5 * step + 1e-9
    offset = np.abs(src_d - grid[nodes])
    keep = offset <= half
    nodes, src_v, offset = nodes[keep], src_v[keep], offset[keep]
    # Closest sample wins a bin; later equal distances keep the first.
    order = np.argsort(offset)[::-1]  # farthest first, then closer overwrites
    for i in order:
        j = int(nodes[i])
        if offset[i] <= dist[j]:
            out[j] = src_v[i]
            dist[j] = offset[i]
    return out


def _alias_index(aliases: list[str], mnemonic: str) -> int:
    n = norm_mnem(mnemonic)
    for i, a in enumerate(aliases):
        if norm_mnem(a) == n:
            return i
    return len(aliases) + 1


def map_file_curves(file_rec: dict, spec: dict, rel_path: str) -> list[Candidate]:
    """Turn every usable curve in one file into assessed candidates for standard names."""
    depth = file_rec["depth_m"]
    out: list[Candidate] = []
    wanted = standard_curves(spec)
    for std in wanted:
        aliases = aliases_for(spec, std)
        alias_norms = {norm_mnem(a): i for i, a in enumerate(aliases)}
        excl = excluded(spec, std)
        dcal_ok = std == "CALI"  # considered only as a fallback later
        for c in file_rec["curves"]:
            n = norm_mnem(c["mnemonic"])
            if n in excl:
                continue
            if n not in alias_norms and not (dcal_ok and n == "DCAL"):
                continue
            values, flags, unit_out = infer_and_convert(c["values"], spec, std, c["unit"], c["descr"])
            if values.shape[0] != depth.shape[0]:
                nmin = min(values.shape[0], depth.shape[0])
                values, depth_used = values[:nmin], depth[:nmin]
            else:
                depth_used = depth
            rank = alias_norms.get(n, 10_000)
            cand = Candidate(
                standard=std,
                mnemonic=c["mnemonic"],
                unit_in=c["unit"],
                unit_out=unit_out,
                description=c["descr"],
                path=rel_path,
                values=values,
                depth_m=depth_used,
                flags=list(flags) + list(file_rec.get("depth_flags") or []),
                alias_rank=rank,
                deprioritized=is_deprioritized_copy(c["mnemonic"]),
            )
            out.append(assess_candidate(cand, spec))
    return out


def _pass_through_depth_ref(file_rec: dict, rel_path: str) -> dict:
    """Record TVD / deviation curves if present. Values are not computed."""
    found = {"tvd": [], "deviation": []}
    tvd_set = {norm_mnem(m) for m in TVD_MNEMONICS}
    dev_set = {norm_mnem(m) for m in DEV_MNEMONICS}
    depth = file_rec["depth_m"]
    for c in file_rec["curves"]:
        n = norm_mnem(c["mnemonic"])
        bucket = None
        if n in tvd_set:
            bucket = "tvd"
        elif n in dev_set:
            bucket = "deviation"
        elif n == "DEV" and any(w in (c["descr"] or "").lower() for w in ("deviation", "inclination", "hole")):
            bucket = "deviation"
        if not bucket:
            continue
        values = clean_values(c["values"])
        unit = c["unit"]
        flags: list[str] = []
        # Convert a TVD depth from feet when the unit (or the values) say so; never invent TVD.
        if bucket == "tvd":
            try:
                values, flags, unit = convert_depth(values, c["unit"])
            except ValueError:
                continue
        n_valid = int(np.isfinite(values).sum())
        if n_valid == 0:
            continue
        found[bucket].append({
            "mnemonic": c["mnemonic"],
            "path": rel_path,
            "unit_in": c["unit"],
            "unit_out": unit if bucket == "tvd" else (c["unit"] or ""),
            "n_valid": n_valid,
            "flags": flags,
            "depth_m": depth,
            "values": values,
        })
    return found


def merge_well(cands: list[Candidate], files: list[dict], spec: dict,
               tvd_bits: list[dict] | None = None) -> dict:
    """Pick one primary source per standard curve and fill remaining gaps from others."""
    by_std: dict[str, list[Candidate]] = {}
    for c in cands:
        by_std.setdefault(c.standard, []).append(c)
    depths = [f["depth_m"] for f in files if f["depth_m"].size]
    for c in cands:
        if c.n_valid:
            depths.append(c.depth_m[np.isfinite(c.values)])
    finite = [d[np.isfinite(d)] for d in depths if np.isfinite(d).any()]
    if not finite:
        raise ValueError("no valid depths")
    dmin = min(float(d.min()) for d in finite)
    dmax = max(float(d.max()) for d in finite)
    step = choose_step(finite)
    grid = build_grid(dmin, dmax, step)
    # Drop leading/trailing all-NaN later, after assignment.

    curves: dict[str, np.ndarray] = {}
    provenance: dict[str, dict] = {}
    rejected = []
    for std, group in by_std.items():
        for c in group:
            if c.rejected:
                rejected.append({"curve": std, "mnemonic": c.mnemonic, "path": c.path, "reason": c.rejected})
        primary = pick_best(group)
        if primary is None:
            continue
        others = [c for c in group if c is not primary and c.rejected is None]
        others = sorted(others, key=lambda c: (c.alias_rank, -c.span_m, -c.n_valid, int(c.computed_file)))
        series = assign_to_grid(primary.depth_m, primary.values, grid, step)
        also = []
        for extra in others:
            fill = assign_to_grid(extra.depth_m, extra.values, grid, step)
            gap = ~np.isfinite(series) & np.isfinite(fill)
            if gap.any():
                series[gap] = fill[gap]
                also.append({"mnemonic": extra.mnemonic, "path": extra.path, "unit": extra.unit_in,
                             "n_filled": int(gap.sum())})
        curves[std] = series
        flags = list(dict.fromkeys(primary.flags))
        range_meta = range_flags(series, spec, std)
        if range_meta.get("median_outside_plausible_range"):
            flags.append("median_outside_plausible_range")
        provenance[std] = {
            "mnemonic": primary.mnemonic,
            "path": primary.path,
            "unit_in": primary.unit_in,
            "unit_out": primary.unit_out,
            "n_valid": int(np.isfinite(series).sum()),
            "flags": flags,
            "range": {k: v for k, v in range_meta.items() if k != "median_outside_plausible_range"},
            "also": also,
        }

    extras: dict[str, np.ndarray] = {}
    extras_prov = {}
    if tvd_bits:
        for rec in tvd_bits:
            key = norm_mnem(rec["mnemonic"])
            # Prefer TVD, then TVDSS; skip duplicates of the same name.
            if key in extras:
                continue
            series = assign_to_grid(rec["depth_m"], rec["values"], grid, step)
            if not np.isfinite(series).any():
                continue
            extras[key] = series
            extras_prov[key] = {k: rec[k] for k in ("mnemonic", "path", "unit_in", "unit_out", "n_valid", "flags")}

    # Trim grid to the interval that has any core curve (or any curve if none).
    core_stack = [curves[c] for c in CORE if c in curves]
    if core_stack:
        any_valid = np.any(np.isfinite(np.vstack(core_stack)), axis=0)
    else:
        cols = list(curves.values()) + list(extras.values())
        any_valid = np.any(np.isfinite(np.vstack(cols)), axis=0) if cols else np.zeros(grid.shape, bool)
    if any_valid.any():
        i0, i1 = int(np.argmax(any_valid)), int(len(any_valid) - 1 - np.argmax(any_valid[::-1]))
        sl = slice(i0, i1 + 1)
        grid = grid[sl]
        curves = {k: v[sl] for k, v in curves.items()}
        extras = {k: v[sl] for k, v in extras.items()}

    return {
        "depth_m": grid,
        "step_m": step,
        "curves": curves,
        "extras": extras,
        "provenance": provenance,
        "extras_provenance": extras_prov,
        "rejected": rejected,
        "depth_from": float(grid[0]) if grid.size else None,
        "depth_to": float(grid[-1]) if grid.size else None,
        "n_samples": int(grid.size),
    }


def completeness_row(merged: dict) -> dict:
    depth = merged["depth_m"]
    curves = merged["curves"]
    # Logged interval: first to last depth where any of GR, RDEP, RHOB or DTC is valid.
    keys = [k for k in ("GR", "RDEP", "RHOB", "DTC") if k in curves]
    if keys:
        mask = np.any(np.vstack([np.isfinite(curves[k]) for k in keys]), axis=0)
    else:
        cols = list(curves.values())
        mask = np.any(np.isfinite(np.vstack(cols)), axis=0) if cols else np.zeros(depth.shape, bool)
    n = int(mask.sum()) or int(depth.size) or 1
    pct = {}
    for c in CORE:
        x = curves.get(c)
        if x is None or not mask.any():
            pct[c] = 0
        else:
            pct[c] = int(round(100 * float(np.isfinite(x[mask]).mean())))
    overall = round(float(np.mean([pct[c] for c in CORE])), 1) if CORE else 0.0
    return {
        "overall": overall,
        "depth_from": float(depth[mask][0]) if mask.any() else merged.get("depth_from"),
        "depth_to": float(depth[mask][-1]) if mask.any() else merged.get("depth_to"),
        "n_samples": n,
        "curves": pct,
        "present": {c: bool(pct[c] > 0) for c in CORE},
        "depth_unit_note": "completeness depths are metres on the aligned MD grid",
    }


def align_well(well_dir: Path, spec: dict, *, name: str | None = None,
               input_root: Path | None = None, meta: dict | None = None) -> dict:
    """Align every LAS-like file in ``well_dir`` into one standard well."""
    well_dir = Path(well_dir)
    input_root = Path(input_root) if input_root else well_dir.parent
    files = discover_las(well_dir)
    if not files:
        raise FileNotFoundError(f"{well_dir}: no LAS files")
    file_recs = []
    file_errors = []
    cands: list[Candidate] = []
    tvd_all: list[dict] = []
    dev_meta: list[dict] = []
    depth_flags_all: list[str] = []
    for path in files:
        rel = str(path.relative_to(input_root))
        try:
            rec = read_las_curves(path)
        except Exception as exc:
            file_errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rec["rel"] = rel
        file_recs.append(rec)
        depth_flags_all.extend(rec["depth_flags"])
        cands.extend(map_file_curves(rec, spec, rel))
        bits = _pass_through_depth_ref(rec, rel)
        tvd_all.extend(bits["tvd"])
        for d in bits["deviation"]:
            dev_meta.append({k: d[k] for k in ("mnemonic", "path", "unit_in", "unit_out", "n_valid", "flags")})
    if not file_recs:
        raise ValueError(f"{well_dir}: no readable LAS files ({file_errors[:3]})")
    merged = merge_well(cands, file_recs, spec, tvd_bits=tvd_all)
    well_name = name or (meta or {}).get("name") or well_dir.name
    comp = completeness_row(merged)
    curve_map = {std: p["mnemonic"] for std, p in merged["provenance"].items()}
    tvd_manifest = [{k: r[k] for k in ("mnemonic", "path", "unit_in", "unit_out", "n_valid", "flags")} for r in tvd_all]
    well_flags = list(dict.fromkeys(depth_flags_all))
    # Metadata TD can hint at deviation; we never compute a TVD curve from it.
    td_md = (meta or {}).get("td_md_m")
    td_tvd = (meta or {}).get("td_tvd_m")
    apparently = None
    if isinstance(td_md, (int, float)) and isinstance(td_tvd, (int, float)):
        apparently = abs(float(td_md) - float(td_tvd)) > 10
        if apparently:
            well_flags.append("apparently_deviated_from_td")
    depth_ref = {
        "reference": "MD",
        "tvd_available": bool(tvd_manifest),
        "tvd_curves": tvd_manifest,
        "deviation_curves": dev_meta,
        "td_md_m": td_md,
        "td_tvd_m": td_tvd,
        "apparently_deviated": apparently,
    }
    return {
        "name": well_name,
        "folder": well_dir.name,
        "depth_m": merged["depth_m"],
        "step_m": merged["step_m"],
        "curves": merged["curves"],
        "extras": merged["extras"],
        "provenance": merged["provenance"],
        "extras_provenance": merged["extras_provenance"],
        "curve_map": curve_map,
        "completeness": comp,
        "rejected": merged["rejected"],
        "file_errors": file_errors,
        "flags": well_flags,
        "depth_reference": depth_ref,
        "n_source_files": len(files),
        "n_source_files_read": len(file_recs),
        "meta": dict(meta or {}),
    }


def _write_las(path: Path, well: dict) -> None:
    """Write a LAS 2.0 file with FORCE names, depth in metres, null -999.25."""
    depth = np.asarray(well["depth_m"], float)
    units = {c: (curve_spec({"canonical": {"curves": {}}, "alignment": {}}, c) or {}).get("standard_unit", "")
             for c in list(well["curves"]) + list(well["extras"])}
    # Prefer official units from the canonical file via a tiny helper
    from wmb.curves import UNITS
    from wmb.align.mapping import EXTRA_CURVES as EX

    def unit_for(name: str) -> str:
        if name in UNITS:
            return UNITS[name]
        if name in EX:
            return EX[name]["standard_unit"]
        if name in {"TVD", "TVDSS", "TVD_M", "TVDEPT", "DVER", "SSDEPT"}:
            return "m"
        return units.get(name, "")

    columns = [(DEPTH_OUT, "m", depth)]
    for name in [c for c in CORE if c in well["curves"]]:
        columns.append((name, unit_for(name), well["curves"][name]))
    for name in [c for c in well["curves"] if c not in CORE]:
        columns.append((name, unit_for(name), well["curves"][name]))
    for name, arr in well["extras"].items():
        columns.append((name, unit_for(name), arr))

    def fmt(v: float) -> str:
        if v is None or not np.isfinite(v):
            return f"{NULL_OUT:.2f}"
        return f"{float(v):.6g}"

    start = float(depth[0]) if depth.size else 0.0
    stop = float(depth[-1]) if depth.size else 0.0
    step = float(well.get("step_m") or 0)
    lines = [
        "~Version information",
        "VERS.                          2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0",
        "WRAP.                           NO : One line per depth step",
        "~Well information",
        f"STRT.M                  {start:.4f} : First depth",
        f"STOP.M                  {stop:.4f} : Last depth",
        f"STEP.M                  {step:.4f} : Depth step",
        f"NULL.                   {NULL_OUT:.2f} : Missing value",
        f"WELL.  {well['name']} : Well name",
        "COMP.   :",
        "FLD.    :",
        "LOC.    :",
        "SRVC.   :",
        "DATE.   :",
        "~Curve information",
    ]
    for name, unit, _ in columns:
        descr = "Measured depth" if name == DEPTH_OUT else name
        lines.append(f"{name}.{unit}   : {descr}")
    lines.append("~Ascii")
    for i in range(depth.size):
        lines.append(" ".join(fmt(col[i]) for _, _, col in columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def well_to_wmb(well: dict, dataset: str = "", split: str | None = None) -> Well:
    curves = {k: np.asarray(v, float) for k, v in well["curves"].items() if k in CORE}
    meta = {
        "flags": well.get("flags") or [],
        "curve_map": well.get("curve_map") or {},
        "step_m": well.get("step_m"),
        "depth_reference": {k: v for k, v in (well.get("depth_reference") or {}).items()
                            if k not in {"tvd_curves", "deviation_curves"} or True},
    }
    # depth_reference curves already have no arrays (stripped in manifest form)
    return Well(well["name"], np.asarray(well["depth_m"], float), curves, dataset, split, meta)


def _source_wells_index(manifest: dict | None) -> dict[str, dict]:
    if not manifest:
        return {}
    return {w["name"]: w for w in manifest.get("wells") or [] if "name" in w}


def write_dataset(wells: list[dict], output: Path, *, dataset_id: str, pack: dict,
                  source_manifest: dict | None = None, alignment_summary: dict | None = None) -> dict:
    """Write one aligned LAS per well, wells.json, completeness table and README. Data stays outside git."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ds_in = (source_manifest or {}).get("dataset") or {}
    pack_ds = pack.get("dataset") or {}
    licence = pack_ds.get("licence") or {}
    credit = licence.get("credit") if isinstance(licence, dict) else ""
    licence_name = licence.get("name") if isinstance(licence, dict) else (licence or "")
    dataset_block = {
        "id": pack_ds.get("id") or dataset_id,
        "name": pack_ds.get("name") or ds_in.get("name") or dataset_id,
        "licence": ds_in.get("licence") or licence_name,
        "source_url": (pack_ds.get("source") or {}).get("url") or ds_in.get("source_url") or "",
        "credit": ds_in.get("credit") or credit,
        "citation": ds_in.get("citation") or licence.get("citation") if isinstance(licence, dict) else ds_in.get("citation"),
        "depth_units": "m",
        "depth_reference": "MD",
        "crs": ds_in.get("crs") or "EPSG:4326",
        "note": (
            "Aligned to the FORCE / canonical curve standard (names, units, metres MD). "
            "One LAS file per well. Original mnemonic and file are in curve_map / curve_sources. "
            "Tests use measured depth; TVD/deviation is recorded in depth_reference when present and is not computed."
        ),
    }
    if ds_in.get("disclaimer_url"):
        dataset_block["disclaimer_url"] = ds_in["disclaimer_url"]
    if pack_ds.get("retrieval_date") or ds_in.get("retrieval_date"):
        dataset_block["retrieval_date"] = pack_ds.get("retrieval_date") or ds_in.get("retrieval_date")

    manifest_wells = []
    comp_rows = []
    for w in wells:
        rel = f"{w['folder']}/aligned.las"
        _write_las(output / rel, w)
        src = w.get("meta") or {}
        entry = {
            "name": w["name"],
            "dataset": dataset_block["id"],
            "split": src.get("split") or dataset_id,
            "depth_units": "m",
            "depth_reference": w.get("depth_reference") or {"reference": "MD"},
            # curve_map is mnemonic-in-the-aligned-file (FORCE names).
            # Original raw names stay in source_curve_map and curve_sources.
            "curve_map": {std: std for std in (w.get("curve_map") or {})},
            "source_curve_map": w.get("curve_map") or {},
            "curve_sources": w.get("provenance") or {},
            "files": [{"path": rel, "format": "las",
                       "curves": [c for c in CORE if (w.get("completeness") or {}).get("present", {}).get(c)]}],
            "completeness": w.get("completeness") or {},
            "flags": w.get("flags") or [],
            "step_m": w.get("step_m"),
            "alignment": {
                "n_source_files": w.get("n_source_files"),
                "n_source_files_read": w.get("n_source_files_read"),
                "rejected": w.get("rejected") or [],
                "file_errors": w.get("file_errors") or [],
            },
        }
        # Pass through useful public metadata; skip huge original file lists.
        for key in ("lat", "lon", "crs", "x", "y", "xy_crs", "kgs_id", "api", "lease", "location",
                    "short_name", "nlog_id", "nlog_url", "field", "onshore_offshore",
                    "spud_date", "spud_year", "td_md_m", "td_tvd_m", "coord_source", "coord_note"):
            if src.get(key) is not None:
                entry[key] = src[key]
        manifest_wells.append(entry)
        present = (w.get("completeness") or {}).get("present") or {}
        pct = (w.get("completeness") or {}).get("curves") or {}
        row = {"well": w["name"], "overall_pct": (w.get("completeness") or {}).get("overall"),
               "depth_from_m": (w.get("completeness") or {}).get("depth_from"),
               "depth_to_m": (w.get("completeness") or {}).get("depth_to"),
               "step_m": w.get("step_m")}
        for c in CORE:
            row[c] = "yes" if present.get(c) else "no"
            row[f"{c}_pct"] = pct.get(c, 0)
        flags = []
        for p in (w.get("provenance") or {}).values():
            flags.extend(p.get("flags") or [])
        flags.extend(w.get("flags") or [])
        row["flags"] = "|".join(dict.fromkeys(flags))
        comp_rows.append(row)

    manifest = {
        "manifest_version": 1,
        "dataset": dataset_block,
        "alignment": alignment_summary or {},
        "wells": manifest_wells,
    }
    (output / "wells.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    fieldnames = ["well", "overall_pct", "depth_from_m", "depth_to_m", "step_m"]
    for c in CORE:
        fieldnames += [c, f"{c}_pct"]
    fieldnames.append("flags")
    with (output / "completeness.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comp_rows)

    present_counts = {c: sum(1 for w in wells if (w.get("completeness") or {}).get("present", {}).get(c)) for c in CORE}
    n = len(wells)
    lines = [
        f"# {dataset_block['name']}",
        "",
        "Aligned local copy for FH2026 testing. **Do not put these files in the public MIT repo.**",
        "",
        f"- **Source:** {dataset_block.get('source_url') or '(see credit)'}",
        f"- **Licence:** {dataset_block.get('licence') or licence_name}",
        f"- **Required credit:** {dataset_block.get('credit') or credit}",
    ]
    if dataset_block.get("citation") and dataset_block.get("citation") != dataset_block.get("credit"):
        lines += [f"- **Citation:** {dataset_block['citation']}"]
    if dataset_block.get("disclaimer_url"):
        lines += [f"- **Disclaimer:** {dataset_block['disclaimer_url']}"]
    lines += [
        "",
        "This folder is a *standardised* copy of the public source files: curve names and units follow "
        "the FORCE / canonical standard, depths are metres measured depth (MD), nulls are NaN / -999.25, "
        "and each well is one LAS file (`<well>/aligned.las`). Original mnemonics and files are recorded "
        "in `wells.json` (`curve_map`, `curve_sources`). Out-of-range samples are flagged, not dropped. "
        "Gaps are left as missing; they are not interpolated. TVD or deviation is recorded in "
        "`depth_reference` when it is already in the source; it is never computed here.",
        "",
        f"**Wells:** {n}. Completeness (at least one valid sample):",
        "",
        "| Curve | Wells |",
        "|---|---:|",
    ]
    for c in CORE:
        lines.append(f"| {c} | {present_counts[c]} / {n} |")
    lines += [
        "",
        "See `completeness.csv` for the well × curve yes/no table (and coverage percentages).",
        "The alignment procedure is in the project docs: `docs/data-alignment.md`.",
        "",
        "## Layout",
        "",
        "```",
        f"{output.name}/",
        "  README.md",
        "  wells.json",
        "  completeness.csv",
        "  <well>/aligned.las",
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def iter_well_dirs(input_root: Path, names: list[str] | None = None) -> list[Path]:
    input_root = Path(input_root)
    skip = {"__pycache__", ".git"}
    dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir() and p.name not in skip and not p.name.startswith(".")]
    if names:
        want = set(names)
        dirs = [p for p in dirs if p.name in want]
        missing = want - {p.name for p in dirs}
        if missing:
            raise FileNotFoundError(f"well folders not found under {input_root}: {sorted(missing)[:8]}")
    return dirs


def align_dataset(input_root: Path, output: Path, dataset_id: str, *,
                  alias_set: str | None = None, well_names: list[str] | None = None,
                  source_manifest: Path | dict | None = None) -> dict:
    """Align every well folder under ``input_root`` and write the output folder."""
    spec = load_align_spec(dataset_id, alias_set=alias_set)
    if source_manifest is None and (Path(input_root) / "wells.json").is_file():
        source_manifest = Path(input_root) / "wells.json"
    if isinstance(source_manifest, Path):
        source_manifest = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    index = _source_wells_index(source_manifest)
    dirs = iter_well_dirs(input_root, well_names)
    wells = []
    errors = []
    for i, d in enumerate(dirs, 1):
        meta = index.get(d.name) or {}
        print(f"[{i}/{len(dirs)}] {d.name}", flush=True)
        try:
            wells.append(align_well(d, spec, name=meta.get("name") or d.name,
                                    input_root=input_root, meta=meta))
        except Exception as exc:
            errors.append({"well": d.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR {d.name}: {type(exc).__name__}: {exc}", flush=True)
    # Mapping summary
    raw_names: dict[str, set[str]] = {c: set() for c in CORE}
    flags: dict[str, int] = {}
    for w in wells:
        for std, p in (w.get("provenance") or {}).items():
            if std in raw_names:
                raw_names[std].add(p.get("mnemonic") or "")
            for f in p.get("flags") or []:
                flags[f] = flags.get(f, 0) + 1
        for f in w.get("flags") or []:
            flags[f] = flags.get(f, 0) + 1
    summary = {
        "dataset": dataset_id,
        "alias_set": spec["alignment"].get("alias_set"),
        "n_wells_in": len(dirs),
        "n_wells_aligned": len(wells),
        "n_well_errors": len(errors),
        "well_errors": errors,
        "raw_mnemonics_used": {k: sorted(v - {""}) for k, v in raw_names.items()},
        "n_raw_mnemonics": {k: len(v - {""}) for k, v in raw_names.items()},
        "flag_counts": dict(sorted(flags.items())),
    }
    write_dataset(wells, output, dataset_id=dataset_id, pack=spec.get("pack") or {},
                  source_manifest=source_manifest, alignment_summary=summary)
    (Path(output) / "alignment_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return {"wells": wells, "summary": summary, "output": Path(output)}
