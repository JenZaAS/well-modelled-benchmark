"""Persistent per-well prediction cache for the viewer and benchmark runner.

Cache root: $WMB_PREDICT_CACHE, else <cache_dir>/predict (cache_dir defaults to
~/fh2026-cache). Never commit these files; they live outside the repo by default,
and results/ plus cache/ are already gitignored if someone points the root there.

Key: model name + model-code fingerprint (file contents) + dataset + test + target + scenario +
well + sorted input names + hashes of the actual input values and of the fitting set.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from wmb import paths
from wmb.contender import Prediction
from wmb.contract import Task, validate_prediction
from wmb.io import Well
from wmb.model_spec import yaml_candidates

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_LOG = logging.getLogger(__name__)


def cache_root() -> Path:
    override = os.environ.get("WMB_PREDICT_CACHE")
    if override:
        return Path(override).expanduser()
    return paths.cache_dir() / "predict"


def _safe_segment(name: str) -> str:
    s = str(name or "").strip()
    if not _SAFE.fullmatch(s):
        return hashlib.sha256(s.encode()).hexdigest()[:16]
    return s


def result_dir(out_root: Path, contender: str, dataset: str, target: str, scenario: str,
               test: str | None = None) -> Path:
    """scores.json directory. Includes test name when present so two tests that
    share dataset/target/scenario do not overwrite each other. Readers should
    still rglob for the older .../scenario/scores.json layout."""
    p = Path(out_root) / _safe_segment(contender) / _safe_segment(dataset) / _safe_segment(target) / _safe_segment(scenario)
    if test:
        p = p / _safe_segment(test)
    return p


def model_files(obj) -> list[Path]:
    """Contender module + its folder's py/yaml siblings + model.yaml candidates."""
    import inspect
    cls = obj if inspect.isclass(obj) else type(obj)
    files: list[Path] = []
    for target in (obj, cls):
        try:
            path = inspect.getsourcefile(target)
        except (OSError, TypeError):
            path = None
        if not path:
            mod = inspect.getmodule(target)
            path = getattr(mod, "__file__", None)
        if path and not str(path).startswith("<"):
            files.append(Path(path))
            folder = Path(path).parent
            files.extend(sorted(folder.glob("*.py")))
            files.extend(sorted(folder.glob("*.yaml")))
            files.extend(sorted(folder.glob("*.yml")))
            break
    files.extend(p for p in yaml_candidates(obj) if p.is_file())
    out, seen = [], set()
    for p in files:
        try:
            p = p.resolve()
        except OSError:
            continue
        if p.is_file() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def fingerprint(obj) -> str:
    """Stable hex digest of model source and yaml (file contents and names, not mtime or location)."""
    h = hashlib.sha256()
    for p in model_files(obj):
        _feed_label(h, p.name)
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:32]


def _feed(h, arr) -> None:
    """Hash float64 values; NaN becomes a fixed marker so it never collides with a real value."""
    if arr is None:
        h.update(b"<none>")
        return
    a = np.asarray(arr, dtype="<f8").ravel()
    nan = np.isnan(a)
    _feed_label(h, str(a.size))
    h.update(np.where(nan, 0.0, a + 0.0).tobytes())
    h.update(nan.tobytes())


def _feed_label(h, label: str) -> None:
    data = label.encode()
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def data_hash_from_task(task: Task, inputs: list[str]) -> str:
    """Identity of what the model sees for a Task: depth, mask, scenario and each used input's values."""
    h = hashlib.sha256()
    _feed_label(h, f"{task.scenario}/{task.gap_length_m}/{task.target}")
    _feed_label(h, json.dumps([task.dataset, task.split, task.meta], sort_keys=True, default=str))
    _feed(h, task.depth)
    _feed(h, task.predict_mask)
    for name in sorted(inputs):
        _feed_label(h, name)
        _feed(h, task.inputs[name].to_numpy(float) if name in task.inputs else None)
    if getattr(task, "target_observed", None) is not None:
        _feed_label(h, "target_observed")
        _feed(h, task.target_observed)
    return h.hexdigest()


def train_hash(wells: list[Well], target: str) -> str:
    """Hash the fitting set in the order supplied to the model, including its input curves."""
    h = hashlib.sha256()
    _feed_label(h, target)
    for w in wells:
        _feed_label(h, json.dumps([w.name, w.dataset, w.split, w.meta], sort_keys=True, default=str))
        _feed(h, w.depth)
        _feed_label(h, str(len(w.curves)))
        for name, values in w.curves.items():
            _feed_label(h, name)
            _feed(h, values)
    return h.hexdigest()


def cache_key(*, model: str, model_hash: str, dataset: str, target: str, well: str,
              inputs: list[str] | tuple[str, ...], data_hash: str, train_hash: str,
              test: str = "", scenario: str = "missing_curve") -> str:
    payload = {
        "model": model, "hash": model_hash, "dataset": dataset, "target": target,
        "well": well, "inputs": sorted(inputs), "test": test or "", "scenario": scenario or "missing_curve",
        "data": data_hash, "train": train_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _path_for(key: str, root: Path | None = None) -> Path:
    root = Path(root) if root is not None else cache_root()
    return root / key[:2] / f"{key}.json"


def load_record(key: str, root: Path | None = None) -> dict[str, Any] | None:
    path = _path_for(key, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        _LOG.warning("Could not read prediction cache entry %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("key") != key:
        _LOG.warning("Invalid prediction cache entry %s", path)
        return None
    return data


def load_prediction(key: str, task: Task) -> tuple[Prediction, str] | None:
    """Read and validate a cached prediction; invalid entries are treated as misses."""
    record = load_record(key)
    if record is None:
        return None
    try:
        timestamp = record.get("cached_at")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("missing timestamp")
        return validate_prediction(task, prediction_from_record(record)), timestamp
    except (AttributeError, TypeError, ValueError) as exc:
        _LOG.warning("Invalid cached prediction %s: %s", key, exc)
        return None


def save_record(record: dict[str, Any], root: Path | None = None) -> str:
    """Atomically publish a record and return its timestamp."""
    key = record["key"]
    path = _path_for(key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / path.name
        staged.write_text(json.dumps({**record, "cached_at": timestamp}, separators=(",", ":"), allow_nan=False),
                          encoding="utf-8")
        staged.replace(path)
    return timestamp


def _list(v):
    if v is None:
        return None
    a = np.asarray(v, float)
    return [None if not np.isfinite(x) else float(x) for x in a]


def record_from_prediction(key: str, pred: Prediction, **meta: Any) -> dict[str, Any]:
    """Full-resolution JSON-safe record of a validated Prediction."""
    q = {str(k): _list(v) for k, v in (pred.quantiles or {}).items()}
    return {"key": key, "p50": _list(pred.p50), "std": _list(pred.std), "p05": _list(pred.p05),
            "p95": _list(pred.p95), "quantiles": q or None, **meta}


def prediction_from_record(record: dict[str, Any]) -> Prediction:
    def a(v):
        return None if v is None else np.array([np.nan if x is None else x for x in v], float)
    q = {float(k): a(v) for k, v in (record.get("quantiles") or {}).items()}
    return Prediction(mean=a(record.get("p50")), std=a(record.get("std")), p05=a(record.get("p05")),
                      p95=a(record.get("p95")), quantiles=q or None)
