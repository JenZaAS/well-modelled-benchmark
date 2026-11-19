"""Back end of the viewer's Prediction tab: free-play prediction of one whole well, plus eligibility and best-model ranking.

Read-only: nothing is written under results/ and no score rows are produced. The target is hidden from the model
(missing_curve style: it is not among the task inputs) while the true curve, when it exists, is returned for display
and a quick MAE / R2. Each model sees only the inputs its model.yaml (or class metadata) declares, intersected with the
curves present in the well; see wmb.model_spec.
"""
from __future__ import annotations

import math
import re
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from benchmark.registry import discover_detailed
from wmb.contract import SCENARIO_MISSING, TARGET_CURVES, Task, task_input_columns, validate_prediction
from wmb.curves import CORE
from wmb.model_spec import eligible, inputs_used, load_model_spec
from wmb.viewer import benchmark_run_time

DEFAULT_TIMEOUT_S = 60.0
MAX_POINTS = 20000
MAX_MODELS = 24
FIT_CACHE_SIZE = 8
MAX_ACTIVE_MODELS = 4
RANK_METRICS = {"MAE": "low", "RMSE": "low", "R2": "high", "cover_90": "ideal"}
COVER_IDEAL = 0.9
_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

_lock = threading.Lock()
_registry_lock = threading.Lock()
_model_slots = threading.BoundedSemaphore(MAX_ACTIVE_MODELS)
_datasets: dict[str, tuple[list, list]] = {}
_fits: dict[tuple, Any] = {}
_registry_cache: list[tuple[dict, list[dict]]] = []


def clear_caches() -> None:
    with _lock:
        _datasets.clear()
        _fits.clear()
    with _registry_lock:
        _registry_cache.clear()


def _registry() -> tuple[dict, list[dict]]:
    with _registry_lock:
        if not _registry_cache:
            _registry_cache.append(discover_detailed(call=_inspect_model))
        return _registry_cache[0]


def _load_dataset(dataset: str) -> tuple[list, list]:
    from benchmark.make_tasks import has_spec
    from benchmark.run import load_dataset
    from wmb import paths, splits
    from wmb.io import iter_manifest_wells
    from wmb.viewer import load_dataset_manifest

    if has_spec(dataset):
        return load_dataset(dataset, 0)
    root = paths.get_dataset_path(dataset)
    if (root / "wells.json").is_file() or dataset not in paths.DATASETS:
        return load_dataset(dataset, 0)
    notes: list[str] = []
    manifest = load_dataset_manifest(dataset, root, dataset, notes)
    if manifest is None:
        raise FileNotFoundError(f"no well manifest for '{dataset}'" + (": " + "; ".join(notes) if notes else ""))
    wells = list(iter_manifest_wells(root, manifest=manifest))
    if not wells:
        raise FileNotFoundError(f"no LAS wells for '{dataset}'")
    return splits.well_split(wells, 0.25, 0)


def _dataset(dataset: str) -> tuple[list, list]:
    if not _NAME_OK.fullmatch(dataset or "") or ".." in dataset:
        raise ValueError("dataset must be a plain name")
    with _lock:
        if dataset in _datasets:
            return _datasets[dataset]
    loaded = _load_dataset(dataset)
    with _lock:
        _datasets[dataset] = loaded
    return loaded


def find_well(dataset: str, name: str):
    """(well, train_wells without that well). The well is looked up by name among the dataset's loaded wells."""
    train, test = _dataset(dataset)
    well = next((w for w in [*train, *test] if w.name == name), None)
    if well is None:
        raise FileNotFoundError(f"well '{name}' not found in dataset '{dataset}'")
    in_train = any(w.name == name for w in train)
    return well, [w for w in train if w.name != name], in_train


def present_curves(well, target: str | None = None) -> list[str]:
    """Curves with at least one finite value, in CORE order first; the target is excluded (it is hidden)."""
    have = [c for c, v in well.curves.items() if c not in ("DEPTH", target) and np.isfinite(np.asarray(v, float)).any()]
    return sorted(have, key=lambda c: (CORE.index(c) if c in CORE else len(CORE), c))


def _build_model_spec(factory):
    model = factory()
    return model, load_model_spec(model)


# ---------- listing ----------

def list_models(dataset: str | None, well_name: str | None, target: str) -> dict:
    """Every registered model with eligibility for this well and target (import failures included, as ineligible)."""
    _check_target(target)
    found, failures = _registry()
    curves: list[str] | None = None
    if dataset and well_name:
        well, _, _ = find_well(dataset, well_name)
        curves = present_curves(well, target)
    out = []
    for name, factory in found.items():
        entry: dict[str, Any] = {"name": name}
        try:
            model, spec = _inspect_model(lambda: _build_model_spec(factory))
        except (Exception, SystemExit) as exc:
            out.append({**entry, "eligible": False, "reason": f"could not be built: {type(exc).__name__}: {exc}", "error": True})
            continue
        entry.update(workstream=getattr(model, "workstream", ""), licence=spec.licence or getattr(model, "licence_name", None),
                     spec_source=spec.source)
        t = spec.for_target(target)
        entry["inputs"] = {"required": t.required, "optional": t.optional} if t else {"required": [], "optional": []}
        if not spec.supports(target):
            ok, reason = False, f"does not support target {target}"
        elif curves is None:
            ok, reason = False, "select a well"
        else:
            ok, reason = eligible(spec, target, curves)
            if ok:
                entry["inputs_used"] = inputs_used(spec, target, curves)
        out.append({**entry, "eligible": ok, "reason": reason})
    for f in failures:
        out.append({"name": f["name"], "workstream": f["workstream"], "eligible": False, "error": True,
                    "reason": f["error"], "inputs": {"required": [], "optional": []}})
    return {"dataset": dataset, "well": well_name, "target": target, "curves": curves or [], "models": out}


# ---------- ranking ----------

def _finite(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def rank_rows(rows: list[dict], dataset: str, target: str, scenario: str = SCENARIO_MISSING, metric: str = "MAE",
              eligible_names=None) -> list[dict]:
    """Scoreboard rows for (dataset, target, scenario) ranked best first on `metric`.

    One row per model (the latest `run_at` wins); rows without a finite metric are skipped; models not in
    `eligible_names` (when given) are skipped. MAE/RMSE: lower is better, R2: higher, cover_90: closer to 0.90.
    """
    if metric not in RANK_METRICS:
        raise ValueError(f"metric must be one of {', '.join(RANK_METRICS)}")
    latest: dict[str, dict] = {}
    for r in rows:
        if (r.get("dataset"), r.get("target"), r.get("scenario") or SCENARIO_MISSING) != (dataset, target, scenario):
            continue
        name = r.get("contender")
        if not isinstance(name, str) or (eligible_names is not None and name not in eligible_names):
            continue
        if name not in latest or benchmark_run_time(r) >= benchmark_run_time(latest[name]):
            latest[name] = r
    kind = RANK_METRICS[metric]
    key = {"low": lambda r: r[metric], "high": lambda r: -r[metric], "ideal": lambda r: abs(r[metric] - COVER_IDEAL)}[kind]
    ranked = sorted((r for r in latest.values() if _finite(r.get(metric)) is not None),
                    key=lambda r: (key(r), r["contender"]))
    return [{"rank": i + 1, "n": len(ranked), "model": r["contender"], "metric": metric, "value": r[metric],
             "run_at": r.get("run_at"), "dataset": dataset, "target": target, "scenario": scenario}
            for i, r in enumerate(ranked)]


def best_model(rows: list[dict], dataset: str, target: str, scenario: str = SCENARIO_MISSING, metric: str = "MAE",
               eligible_names=None) -> dict | None:
    ranked = rank_rows(rows, dataset, target, scenario, metric, eligible_names)
    return ranked[0] if ranked else None


# ---------- prediction ----------

def _call_with_timeout(fn: Callable[[threading.Event], Any], timeout_s: float):
    box: dict[str, Any] = {}
    cancelled = threading.Event()
    if not _model_slots.acquire(blocking=False):
        raise RuntimeError("prediction workers are busy; retry after they finish")

    def work() -> None:
        try:
            box["value"] = fn(cancelled)
        except (Exception, SystemExit) as exc:
            box["error"] = exc
        finally:
            _model_slots.release()

    th = threading.Thread(target=work, daemon=True)
    try:
        th.start()
    except RuntimeError:
        _model_slots.release()
        raise
    th.join(timeout_s)
    if th.is_alive():
        cancelled.set()
        raise TimeoutError(f"timed out after {timeout_s:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _inspect_model(fn: Callable[[], Any]):
    return _call_with_timeout(lambda _: fn(), DEFAULT_TIMEOUT_S)


def _task(well, target: str, keep: list[str]) -> Task:
    """Keep the standard blank columns and populate only the available declared inputs."""
    depth = np.array(well.depth, dtype=float, copy=True)
    n = len(depth)
    cols = {c: np.full(n, np.nan) for c in task_input_columns(target)}
    cols["DEPTH"] = depth
    cols.update({c: np.asarray(well.curves[c], float) for c in keep})
    return Task(well.name, depth, pd.DataFrame(cols), target, np.ones(n, bool), SCENARIO_MISSING,
                dataset=well.dataset, split=well.split)


def quick_scores(y: np.ndarray | None, p: np.ndarray) -> dict:
    if y is None:
        return {"mae": None, "r2": None, "n_scored": 0}
    m = np.isfinite(y) & np.isfinite(p)
    n = int(m.sum())
    if n == 0:
        return {"mae": None, "r2": None, "n_scored": 0}
    err = p[m] - y[m]
    sst = float(((y[m] - y[m].mean()) ** 2).sum())
    return {"mae": round(float(np.abs(err).mean()), 5), "r2": round(1 - float((err ** 2).sum()) / sst, 5) if n > 1 and sst > 0 else None,
            "n_scored": n}


def _arr(v, idx, nd: int = 4):
    if v is None:
        return None
    a = np.asarray(v, float)[idx]
    return [round(float(x), nd) if np.isfinite(x) else None for x in a]


def _run_one(dataset, name, factory, well, train, in_train, target, curves, timeout_s) -> dict:
    res: dict[str, Any] = {"name": name, "ok": False, "error": None}
    t0 = time.perf_counter()
    key = (dataset, name, target, well.name if in_train else None)

    def work(cancelled):
        with _lock:
            fitted = _fits.pop(key, None)
        model = fitted if fitted is not None else factory()
        spec = load_model_spec(model)
        ok, reason = eligible(spec, target, curves)
        if not ok:
            return {"error": reason}, fitted
        used = inputs_used(spec, target, curves)
        task = _task(well, target, used["required"] + used["optional"])
        if not cancelled.is_set() and fitted is None:
            model.fit(train, target)
        if cancelled.is_set():
            return {}, None
        pred = validate_prediction(task, model.predict_task(task))
        return {"ok": True, "_pred": pred, "inputs_used": used}, model

    try:
        result, fitted = _call_with_timeout(work, timeout_s)
        res.update(result)
        if fitted is not None:
            with _lock:
                while len(_fits) >= FIT_CACHE_SIZE:
                    _fits.pop(next(iter(_fits)))
                _fits[key] = fitted
    except (Exception, SystemExit) as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
    res["runtime_s"] = round(time.perf_counter() - t0, 3)
    return res


def _check_target(target: str) -> None:
    if target not in TARGET_CURVES:
        raise ValueError(f"target must be one of {', '.join(TARGET_CURVES)}")


def predict_well(dataset: str, well_name: str, target: str, model_names: list[str], *, timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_points: int = MAX_POINTS) -> dict:
    """Predict `target` for one whole well with each named model. One model failing never fails the others."""
    _check_target(target)
    if not isinstance(model_names, list) or not all(isinstance(m, str) for m in model_names) or len(model_names) > MAX_MODELS:
        raise ValueError(f"models must be a list of at most {MAX_MODELS} names")
    well, train, in_train = find_well(dataset, well_name)
    found, failures = _registry()
    broken = {f["name"]: f["error"] for f in failures}
    curves = present_curves(well, target)
    n = len(well.depth)
    idx = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)
    truth = np.asarray(well.curves[target], float) if target in well.curves and np.isfinite(well.curves[target]).any() else None
    models = []
    for name in dict.fromkeys(model_names):
        if name in found:
            r = _run_one(dataset, name, found[name], well, train, in_train, target, curves, timeout_s)
        else:
            r = {"name": name, "ok": False, "error": broken.get(name, f"unknown model '{name}'"), "runtime_s": 0.0}
        pred = r.pop("_pred", None)
        if pred is not None:
            band = pred.p05 is not None and pred.p95 is not None
            r.update(quick_scores(truth, pred.p50), p50=_arr(pred.p50, idx),
                     p05=_arr(pred.p05, idx) if band else None, p95=_arr(pred.p95, idx) if band else None,
                     std=None if band else _arr(pred.std, idx))
        models.append(r)
    return {"dataset": dataset, "well": well.name, "target": target, "scenario": SCENARIO_MISSING, "n_samples": n,
            "decimated": n > max_points, "depth": _arr(well.depth, idx), "true": _arr(truth, idx) if truth is not None else None,
            "curves": [c for c in present_curves(well)], "models": models}
