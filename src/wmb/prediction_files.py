"""Per-run prediction files for the viewer's Benchmark tab (not part of scoring).

    results/<model>/<dataset>/<target>/<scenario>/predictions.json   format "json" (default)
    results/<model>/<dataset>/<target>/<scenario>/predictions.npz    format "npz"

Both hold the same information: per test well the depth, the measured target, p50, the 90% band (p05/p95, or std
when there is no band), the hidden intervals and a few context logs, all rounded to 4 decimals. "json" is one
human-readable file. "npz" stores the same rounded values as compressed fixed-point integers: about 3x smaller,
about 3x faster to write, and the viewer reads one well without parsing the whole run. The viewer serves either
as the same JSON, so switching formats changes nothing in the browser.

Choose the format with `python -m benchmark.run --predictions json|npz`, the environment variable
WMB_PREDICTIONS_FORMAT, or `predictions_format=` on benchmark.run.run_scenario / run_model / run_tests.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

FORMATS = ("json", "npz")
DEFAULT_FORMAT = "json"
ENV_VAR = "WMB_PREDICTIONS_FORMAT"
FILE_NAMES = {"json": "predictions.json", "npz": "predictions.npz"}
DECIMALS = 4
CONTEXT_CURVES = ("GR", "CALI", "RDEP")
NPZ_VERSION = 1
_ARRAYS = ("depth", "true", "p50", "p05", "p95", "std")
_EXACT_LIMIT = 2.0 ** 33                   # |x * 10**DECIMALS| below this: NumPy rounding is exact away from ties


def resolve_format(value: str | None = None) -> str:
    """`value`, else $WMB_PREDICTIONS_FORMAT, else DEFAULT_FORMAT. Raises ValueError for anything else."""
    fmt = (value or os.environ.get(ENV_VAR) or DEFAULT_FORMAT).strip().lower()
    if fmt not in FORMATS:
        raise ValueError(f"predictions format must be one of {', '.join(FORMATS)}, got {fmt!r}")
    return fmt


# ---------- rounding: identical to [round(float(x), 4) if finite else None for x in a], vectorised ----------

def _rounded(a: np.ndarray, nd: int = DECIMALS) -> tuple[np.ndarray, np.ndarray]:
    """(rounded float64 values, finite mask), equal element-wise to Python's round(float(x), nd)."""
    a = np.asarray(a, float).ravel()
    finite = np.isfinite(a)
    scale = 10.0 ** nd
    with np.errstate(invalid="ignore", over="ignore"):
        scaled = a * scale
        out = np.rint(scaled) / scale
        # NumPy's x*10**nd can land on the wrong side of a .5 tie, and is inexact for huge values: use Python there
        risky = finite & ~(np.abs(scaled) < _EXACT_LIMIT)
        risky |= finite & (np.abs(np.abs(scaled - np.floor(scaled)) - 0.5) < 1e-6)
    for i in np.flatnonzero(risky):
        out[i] = round(float(a[i]), nd)
    out[~finite] = np.nan
    return out, finite


def json_values(v, nd: int = DECIMALS) -> list | None:
    """Rounded list with NaN/inf as None (valid JSON); None stays None."""
    if v is None:
        return None
    out, finite = _rounded(v, nd)
    obj = out.astype(object)
    obj[~finite] = None
    return obj.tolist()


def gap_intervals(depth, mask) -> list[list[float]]:
    """[top, bottom] depth (rounded) of each contiguous run of mask."""
    m = np.asarray(mask, bool).ravel()
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2] - 1
    d = np.asarray(depth, float)
    return [[a, b] for a, b in zip(json_values(d[starts]), json_values(d[ends]))]


# ---------- one record per task, shared by both formats ----------

def well_records(tasks, preds, truth: dict) -> list[dict]:
    """Raw arrays per task: key, well_id, depth, true, p50, p05/p95 (band) or std, mask, context curves."""
    seen: dict[str, int] = {}
    out = []
    for t, p in zip(tasks, preds):
        k = seen.get(t.well_id, 0)
        seen[t.well_id] = k + 1
        band = p.p05 is not None and p.p95 is not None
        out.append({
            "key": t.well_id if k == 0 else f"{t.well_id}#{k}", "well_id": t.well_id,
            "depth": np.asarray(t.depth, float), "true": np.asarray(truth[(t.well_id, t.target)], float),
            "p50": np.asarray(p.p50, float),
            "p05": np.asarray(p.p05, float) if band else None, "p95": np.asarray(p.p95, float) if band else None,
            "std": None if band or p.std is None else np.asarray(p.std, float),
            "mask": np.asarray(t.predict_mask, bool),
            "context": {c: np.asarray(v, float) for c in CONTEXT_CURVES
                        if c != t.target and (v := truth.get((t.well_id, c))) is not None and np.isfinite(v).any()},
        })
    return out


def well_json(r: dict) -> dict:
    """The JSON object the viewer receives for one well (same keys and order in both formats)."""
    return {"key": r["key"], "well_id": r["well_id"], "depth": json_values(r["depth"]), "true": json_values(r["true"]),
            "p50": json_values(r["p50"]), "p05": json_values(r["p05"]), "p95": json_values(r["p95"]),
            "std": json_values(r["std"]), "gaps": gap_intervals(r["depth"], r["mask"]),
            "context": {c: json_values(v) for c, v in r["context"].items()}}


def header(base: dict, scenario: str) -> dict:
    return {"contender": base.get("contender"), "test": base.get("test"), "dataset": base["dataset"],
            "target": base["target"], "scenario": scenario}


def json_payload(base: dict, scenario: str, tasks, preds, truth: dict) -> dict:
    return {**header(base, scenario), "wells": [well_json(r) for r in well_records(tasks, preds, truth)]}


# ---------- npz ----------

def _encode(a: np.ndarray) -> np.ndarray:
    """Rounded values as fixed-point integers k = x * 10**DECIMALS (int32 if they fit, else int64; the dtype's minimum
    marks NaN), or the rounded float64 values if they are too large for exact integers. Decoding is exact."""
    vals, finite = _rounded(a)
    scaled = np.where(finite, vals * 10.0 ** DECIMALS, 0.0)
    biggest = float(np.abs(scaled).max(initial=0.0))
    if biggest >= 2.0 ** 52:
        return vals
    dtype = np.int32 if biggest < 2 ** 31 - 1 else np.int64
    k = np.rint(scaled).astype(dtype)
    k[~finite] = np.iinfo(dtype).min
    return k


def _decode(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind != "i":
        return a.astype(float)
    out = a.astype(float) / 10.0 ** DECIMALS
    out[a == np.iinfo(a.dtype).min] = np.nan
    return out


def _save(path: Path, arrays: dict) -> None:
    """Like np.savez_compressed (same .npz layout, read with np.load) but with fast deflate level 1 (~4x faster)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, a in arrays.items():
            with zf.open(f"{name}.npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, a if a.ndim == 0 else np.ascontiguousarray(a), allow_pickle=False)


def write_npz(path: Path, base: dict, scenario: str, tasks, preds, truth: dict) -> None:
    records = well_records(tasks, preds, truth)
    meta = {"version": NPZ_VERSION, "decimals": DECIMALS, **header(base, scenario),
            "wells": [{"key": r["key"], "well_id": r["well_id"], "context": list(r["context"])} for r in records]}
    arrays = {"meta": np.array(json.dumps(meta))}
    for i, r in enumerate(records):
        for name in _ARRAYS:
            if r[name] is not None:
                arrays[f"w{i}_{name}"] = _encode(r[name])
        arrays[f"w{i}_mask"] = r["mask"]
        for c, v in r["context"].items():
            arrays[f"w{i}_ctx_{c}"] = _encode(v)
    _save(path, arrays)


def _npz_meta(z) -> dict:
    meta = json.loads(z["meta"].item())
    if meta.get("version") != NPZ_VERSION:
        raise ValueError(f"unsupported predictions.npz version {meta.get('version')}")
    return meta


def read_npz_well(path: Path, well: str) -> dict:
    """Same result as reading predictions.json: {contender, dataset, target, scenario, well} (or {wells} for '*')."""
    with np.load(path, allow_pickle=False) as z:
        meta = _npz_meta(z)
        if well == "*":
            return {"wells": [w["key"] for w in meta["wells"]]}
        for i, w in enumerate(meta["wells"]):
            if well in (w["key"], w["well_id"]):
                arr = {n: (_decode(z[f"w{i}_{n}"]) if f"w{i}_{n}" in z.files else None) for n in _ARRAYS}
                rec = {"key": w["key"], "well_id": w["well_id"], **arr, "mask": z[f"w{i}_mask"],
                       "context": {c: _decode(z[f"w{i}_ctx_{c}"]) for c in w["context"]}}
                return {k: meta.get(k) for k in ("contender", "dataset", "target", "scenario")} | {"well": well_json(rec)}
    raise FileNotFoundError(f"well {well} not in predictions.npz")


# ---------- writing either format ----------

def remove(out: Path) -> None:
    for name in FILE_NAMES.values():
        (Path(out) / name).unlink(missing_ok=True)


def write(out: Path, fmt: str, base: dict, scenario: str, tasks, preds, truth: dict, json_builder=None) -> Path:
    """Replace any prediction file in `out` with one in `fmt`, atomically. json_builder defaults to json_payload."""
    fmt = resolve_format(fmt)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    remove(out)
    target = out / FILE_NAMES[fmt]
    with TemporaryDirectory(dir=out) as tmp:
        staged = Path(tmp) / target.name
        if fmt == "json":
            payload = (json_builder or json_payload)(base, scenario, tasks, preds, truth)
            staged.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        else:
            write_npz(staged, base, scenario, tasks, preds, truth)
        staged.replace(target)
    return target
