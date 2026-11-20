"""Run contenders on a dataset, score them, write results/<contender>/<dataset>/<target>/<scenario>/.

Examples:
    python -m benchmark.run --list
    python -m benchmark.run --dataset synthetic --target RHOB
    python -m benchmark.run --model template_linear --dataset synthetic --scenario all --target all
    python -m benchmark.run --dataset force2020 --target DTS --contenders black_gardner,blue_gbt

Tasks come from benchmark.make_tasks (synthetic, force2020). Other datasets (a folder with wells.json)
keep the older well-split behaviour for the missing_curve scenario.
"""
from __future__ import annotations

import argparse
import hashlib
import functools
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from benchmark import make_tasks as mt
from benchmark import test_def as td
from benchmark.registry import REPO, discover  # noqa: F401  (also puts src/ on sys.path)
from wmb import paths, predict_cache, scoring, splits
from wmb.contender import Prediction
from wmb.contract import GAP_LENGTHS_M, SCENARIO_MISSING, TARGET_CURVES, Task, validate_prediction
from wmb.curves import CORE

# Tests for these datasets run only when an aligned folder with wells.json is present.
OPTIONAL_DATASETS = frozenset({"nlog_dts", "kansas_kgs_dts"})


def dataset_manifest_present(name: str) -> bool:
    """True when the local folder exists and has a wells.json (aligned or other manifest data)."""
    try:
        root = paths.get_dataset_path(name)
    except (paths.DataPathsError, SystemExit, FileNotFoundError, OSError):
        return False
    return (root / "wells.json").is_file()


GAP_KEYS = [f"gap_{int(g)}" for g in GAP_LENGTHS_M]
SCENARIO_CHOICES = [SCENARIO_MISSING, "gap", *GAP_KEYS, "all"]


def load_dataset(name: str, seed: int):
    """(train_wells, test_wells). Spec datasets use their task spec; others use a seeded well split."""
    if mt.has_spec(name):
        return mt.load_wells(name, mt.load_spec(name))
    from wmb.io import iter_manifest_wells
    wells = list(iter_manifest_wells(paths.get_dataset_path(name)))
    return splits.well_split(wells, 0.25, seed)


def parse_scenarios(arg: str) -> list[str]:
    if arg == "all":
        return [SCENARIO_MISSING, *GAP_KEYS]
    return GAP_KEYS if arg == "gap" else [arg]


def hardware() -> dict:
    return {"machine": platform.machine(), "system": platform.system(), "processor": platform.processor(),
            "python_version": platform.python_version(), "cpu_count": os.cpu_count()}


@functools.lru_cache(maxsize=1)
def git_info() -> dict:
    """{git_commit, git_commit_full} of this repo's HEAD; empty when git or the repository is unavailable."""
    try:
        full = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"git_commit": full[:7], "git_commit_full": full} if full else {}


def _quantile(p: Prediction, level: float):
    for k, v in (p.quantiles or {}).items():
        if abs(float(k) - level) < 1e-9:
            return np.asarray(v, float)
    return None


def score_tasks(tasks: list[Task], preds: list[Prediction], truth: dict) -> dict:
    """Pool scored samples (predict_mask, finite truth, finite p50) over tasks and score them."""
    cols: dict[str, list] = {k: [] for k in ("y", "p50", "std", "p05", "p95", "p25", "p75")}
    for t, p in zip(tasks, preds):
        m = t.predict_mask
        cols["y"].append(np.asarray(truth[(t.well_id, t.target)], float)[m])
        cols["p50"].append(p.p50[m])
        for k, v in (("std", p.std), ("p05", p.p05), ("p95", p.p95), ("p25", _quantile(p, 0.25)), ("p75", _quantile(p, 0.75))):
            cols[k].append(np.full(int(m.sum()), np.nan) if v is None else np.asarray(v, float)[m])
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    opt = {k: (None if np.isnan(cat[k]).all() else cat[k]) for k in ("std", "p05", "p95", "p25", "p75")}
    return scoring.regression_scores(cat["y"], cat["p50"], opt["std"], p05=opt["p05"], p95=opt["p95"],
                                     p25=opt["p25"], p75=opt["p75"])


def _write(out, name: str, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(payload, indent=2, default=str))


def _arr(v, nd: int = 4):
    """Rounded list with NaN/inf as null (valid JSON); None stays None."""
    if v is None:
        return None
    a = np.asarray(v, float)
    return [round(float(x), nd) if np.isfinite(x) else None for x in a]


def _gap_intervals(depth: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """[top, bottom] depth of each contiguous run of predict_mask."""
    out, start = [], None
    for i, m in enumerate(np.asarray(mask, bool)):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i if m else i - 1
            out.append([round(float(depth[start]), 4), round(float(depth[end]), 4)])
            start = None
    return out


def prediction_payload(base: dict, key: str, tasks: list[Task], preds: list[Prediction], truth: dict) -> dict:
    """Compact per-well arrays for the viewer's Benchmark tab (results/.../predictions.json); not part of scoring."""
    seen: dict[str, int] = {}
    wells = []
    for t, p in zip(tasks, preds):
        k = seen.get(t.well_id, 0)
        seen[t.well_id] = k + 1
        has_band = p.p05 is not None and p.p95 is not None
        wells.append({
            "key": t.well_id if k == 0 else f"{t.well_id}#{k}", "well_id": t.well_id,
            "depth": _arr(t.depth), "true": _arr(truth[(t.well_id, t.target)]), "p50": _arr(p.p50),
            "p05": _arr(p.p05) if has_band else None, "p95": _arr(p.p95) if has_band else None,
            "std": None if has_band else _arr(p.std),
            "gaps": _gap_intervals(t.depth, t.predict_mask),
            "context": {curve: _arr(values) for curve in ("GR", "CALI", "RDEP")
                        if curve != t.target and (values := truth.get((t.well_id, curve))) is not None
                        and np.isfinite(values).any()},
        })
    return {"contender": base.get("contender"), "test": base.get("test"), "dataset": base["dataset"],
            "target": base["target"], "scenario": key, "wells": wells}


def _fail(row: dict, out, exc: Exception, classic: bool = True) -> dict:
    row["error"] = f"{type(exc).__name__}: {exc}"
    _write(out, "failure.json", {k: row.get(k) for k in ("contender", "test", "dataset", "target", "scenario", "error")}
           | {"traceback": traceback.format_exc(limit=4)})
    (out / "scores.json").unlink(missing_ok=True)  # a failed rerun must not leave a stale scoreboard row
    (out / "predictions.json").unlink(missing_ok=True)
    if classic and row["scenario"] == SCENARIO_MISSING:
        (out.parent / "scores.json").unlink(missing_ok=True)
    return row


def _check_task_depths(tasks: list[Task], truth: dict) -> None:
    """The cached task must describe the loaded test well exactly; never score by row index alone."""
    for t in tasks:
        d = truth.get((t.well_id, "DEPTH"))
        if d is None:
            raise ValueError(f"task well {t.well_id} is not among the loaded test wells")
        d = np.asarray(d, float)
        if len(d) != len(t.depth):
            raise ValueError(f"task {t.well_id}: depth length {len(t.depth)} != loaded well {len(d)}; rebuild the task cache")
        if not np.allclose(d, t.depth, rtol=0, atol=1e-4):
            raise ValueError(f"task {t.well_id}: depth values differ from the loaded well; rebuild the task cache")


def _task_cache_key(model, task: Task, dataset: str, test: str, thash: str, mhash: str) -> str:
    inputs = [c for c in task.inputs.columns if c != "DEPTH"]
    return predict_cache.cache_key(
        model=model.name, model_hash=mhash, dataset=dataset, target=task.target, well=task.well_id, inputs=inputs,
        data_hash=predict_cache.data_hash_from_task(task, inputs), train_hash=thash, test=test or "",
        scenario=task.scenario_key)


def _scenario_cache_keys(model, tasks: list[Task], base: dict, thash: str, mhash: str) -> list[str]:
    keys = [_task_cache_key(model, task, base["dataset"], base.get("test") or "", thash, mhash)
            for task in tasks]
    # A model may carry state between task calls. Isolate different task sequences.
    scope = hashlib.sha256("".join(keys).encode()).hexdigest()
    return [hashlib.sha256(f"{scope}:{i}:{key}".encode()).hexdigest() for i, key in enumerate(keys)]


def run_scenario(model, tasks: list[Task], truth: dict, base: dict, out_root, timings: dict,
                 per_test: bool = False, force: bool = False, train=None) -> dict:
    """Predict, validate and score one (model, target, scenario). Returns the result row (with `error` on failure).

    per_test writes under .../<scenario>/<test>/ and leaves the classic .../<scenario>/scores.json alone.
    When train is supplied, fit only on a miss and cache each prediction. `force` bypasses reads.
    """
    key = tasks[0].scenario_key
    row = {**git_info(), **base, "scenario": key, "n_tasks": len(tasks), **timings}
    out = predict_cache.result_dir(out_root, model.name, base["dataset"], base["target"], key,
                                   base.get("test") if per_test else None)
    try:
        _check_task_depths(tasks, truth)
        mhash = row["model_hash"] = predict_cache.fingerprint(model)
        if train is not None:
            thash = predict_cache.train_hash(train, base["target"])
            keys = _scenario_cache_keys(model, tasks, base, thash, mhash)
        else:
            keys = [None] * len(tasks)  # Caller supplied an already fitted model with unknown training identity.
        cached = [predict_cache.load_prediction(ck, task) if ck and not force else None
                  for task, ck in zip(tasks, keys)]
        # Replay the full sequence if any entry is missing, so stateful predictions are reproducible.
        use_cache = all(hit is not None for hit in cached)
        hits = len(tasks) if use_cache else 0
        t: dict = {"predict_s_batch": None}
        if train is not None:
            t["fit_s"] = None
            if not use_cache:
                with scoring.timer(t, "fit_s"):
                    model.fit(train, base["target"])
        preds = []
        predict_time = 0.0
        for task, ck, hit in zip(tasks, keys, cached):
            if use_cache and hit is not None:
                pred = hit[0]
            else:
                start = time.perf_counter()
                pred = validate_prediction(task, model.predict_task(task))
                predict_time += time.perf_counter() - start
                if ck:
                    try:
                        predict_cache.save_record(predict_cache.record_from_prediction(ck, pred, model=model.name))
                    except (OSError, ValueError, TypeError) as exc:
                        print(f"warning: prediction cache not written: {exc}", file=sys.stderr)
            preds.append(pred)
        row["cached_tasks"] = hits
        n_samples = int(sum(task.predict_mask.sum() for task in tasks))
        # A run with any cache hits does not measure representative model throughput.
        t["predict_s"] = None if hits else predict_time
        if not hits:
            try:
                with scoring.timer(t, "predict_s_batch"):
                    model.predict_batch(tasks)
            except Exception:  # timing only; the per-well predictions were already validated
                t["predict_s_batch"] = None
        row.update(score_tasks(tasks, preds, truth))
        row.update({k: (None if v is None else round(v, 4)) for k, v in t.items()})
        row.update(predict_s_per_well=None if hits else round(predict_time / len(tasks), 6),
                   predict_s_per_1000=None if hits else round(predict_time / max(n_samples, 1) * 1000, 6),
                   samples_per_s=round(n_samples / predict_time, 1) if not hits and predict_time > 0 else None,
                   n_samples=n_samples)
    except Exception as exc:
        return _fail(row, out, exc, classic=not per_test)
    (out / "failure.json").unlink(missing_ok=True)
    _write(out, "scores.json", row)
    prediction_path = out / "predictions.json"
    try:  # viewer-only extra; never let it fail a scored run
        prediction_path.unlink(missing_ok=True)
        with TemporaryDirectory(dir=out) as tmp:
            staged = Path(tmp) / prediction_path.name
            staged.write_text(json.dumps(
                prediction_payload({**base, "contender": model.name}, key, tasks, preds, truth),
                separators=(",", ":"), allow_nan=False,
            ))
            staged.replace(prediction_path)
    except (OSError, ValueError, TypeError) as exc:
        print(f"warning: predictions.json not written for {model.name}/{key}: {exc}", file=sys.stderr)
    if key == SCENARIO_MISSING and not per_test:  # older layout, kept for existing readers
        _write(out.parent, "scores.json", row)
    return row


def run_model(name: str, factory, train, test_tasks: list[Task], truth: dict, target: str, scenarios: list[str],
              dataset: str, out_root, test_name=None, per_test: bool = False, force: bool = False) -> list[dict]:
    """All selected scenarios of one target for one model. Failures are returned as rows with `error`.

    test_name: callable (target, scenario_key) -> test name; default is the shipped def that matches, else a generated name.
    per_test: write under .../<scenario>/<test>/ (the --test layout). force: ignore the prediction cache.
    """
    test_name = test_name or (lambda tg, k: td.generated_def(dataset, tg, k).name)
    by_key = {k: [t for t in test_tasks if t.target == target and t.scenario_key == k] for k in scenarios}
    by_key = {k: v for k, v in by_key.items() if v}
    if not by_key:
        return []
    rows = []
    for k, ts in by_key.items():
        # fresh model and fit per scenario so nothing observed in one scenario leaks into another
        timings: dict = {}
        test = test_name(target, k)
        out = predict_cache.result_dir(out_root, name, dataset, target, k, test if per_test else None)
        try:
            with scoring.timer(timings, "load_s"):
                model = factory()
            if hasattr(model, "supports") and not model.supports(target):
                return []
        except Exception as exc:
            rows.append(_fail({"contender": name, "test": test, "dataset": dataset, "target": target, "scenario": k},
                              out, exc, classic=not per_test))
            continue
        base = {"contender": model.name, "test": test, "workstream": model.workstream, "dataset": dataset,
                "target": target,
                "n_train_wells": len(train), "n_test_wells": len({t.well_id for t in ts}),
                "hardware": hardware(), "machine": f"{platform.processor() or platform.machine()} / {platform.system()}",
                "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        rows.append(run_scenario(model, ts, truth, base, out_root, {k_: round(v, 4) for k_, v in timings.items()},
                                 per_test=per_test, force=force, train=train))
    return rows


def _fmt(row: dict) -> str:
    g = lambda k: row[k] if row.get(k) is not None else float("nan")  # noqa: E731
    return (f"MAE={g('MAE'):.4f} R2={g('R2'):.3f} cover90={g('cover_90'):.2f} cover50={g('cover_50'):.2f} "
            f"fit={g('fit_s'):.2f}s pred={g('predict_s'):.2f}s")


def _truth(test: list) -> dict:
    truth = {(w.name, t): w.curves[t] for w in test for t in CORE if t in w.curves}
    truth.update({(w.name, "DEPTH"): w.depth for w in test})
    return truth


def _print_rows(n: str, got: list[dict]) -> None:
    for r in got:
        label = f"{n} {r['target']} {r['scenario']} [{r['test']}]"
        print(f"  {label:60s} " + (f"FAILED {r['error']}" if "error" in r else _fmt(r)))


def _write_runs(out_root, rows: list[dict], tag: str) -> None:
    runs = out_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%dT%H%M%S')}_{tag}"
    (runs / f"{stem}.json").write_text(json.dumps(rows, indent=2, default=str))
    pd.DataFrame([{k: v for k, v in r.items() if k != "hardware"} for r in rows]).to_csv(runs / f"{stem}.csv", index=False)


def run_tests(defs: list[td.TestDef], found: dict, names: list[str], cache, out_root, force: bool = False, skip_missing: bool = False) -> list[dict]:
    """Run each test def for every named model (fresh fit per def). A def that cannot run yields error rows."""
    loaded: dict = {}
    rows = []
    for d in defs:
        key = d.scenario_key
        print(f"test={d.name} dataset={d.dataset} target={d.target} scenario={key} wells={d.wells_split}")
        if skip_missing and d.dataset in OPTIONAL_DATASETS and not dataset_manifest_present(d.dataset):
            print(f"  skipped (optional dataset {d.dataset} is not present locally; see docs/data-alignment.md)")
            continue
        try:
            ds_key = (d.dataset, td.effective_seed(d))
            if ds_key not in loaded:
                train, test = load_dataset(*ds_key)
                loaded[ds_key] = (train, test, _truth(test))
            train, test, truth = loaded[ds_key]
            tasks = td.build_def_tasks(d, train, test, cache)
            if not tasks:
                raise ValueError(f"test {d.name} has no tasks: no selected test well has {d.target} for {key}")
        except (Exception, SystemExit) as exc:
            err = exc if isinstance(exc, Exception) else RuntimeError(str(exc) or "exited")
            for n in names:
                if n not in found:
                    print(f"unknown contender: {n}", file=sys.stderr); continue
                row = {"contender": n, "test": d.name, "dataset": d.dataset, "target": d.target, "scenario": key}
                rows.append(_fail(row, predict_cache.result_dir(out_root, n, d.dataset, d.target, key, d.name), err,
                                  classic=False))
                _print_rows(n, [row])
            continue
        print(f"  train_wells={len(train)} test_wells={len({t.well_id for t in tasks})} tasks={len(tasks)}")
        for n in names:
            if n not in found:
                print(f"unknown contender: {n}", file=sys.stderr); continue
            got = run_model(n, found[n], train, tasks, truth, d.target, [key], d.dataset, out_root, lambda *_: d.name,
                            per_test=True, force=force)
            if not got:
                print(f"  {n:24s} skipped (target not supported or no test tasks)")
            _print_rows(n, got)
            rows += got
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="well-modelled-benchmark runner")
    ap.add_argument("--test", help="test definition name from benchmark/test_defs/ (comma-separated), or 'all'; see --list-tests")
    ap.add_argument("--dataset", default=None, help="synthetic | force2020 | <dataset key or folder under data_root with wells.json> (default synthetic)")
    ap.add_argument("--target", default=None, help="a core curve (any for missing_curve; RHOB, DTC, DTS for gap) or 'all' (= RHOB, DTC, DTS); default RHOB")
    ap.add_argument("--scenario", help=f"one of {', '.join(SCENARIO_CHOICES)} (default missing_curve), or the scenario key of --test")
    ap.add_argument("--contenders", "--model", dest="contenders", default="all", help="comma-separated names, or 'all'")
    ap.add_argument("--seed", type=int, default=0, help="well split seed for datasets without a task spec")
    ap.add_argument("--cache", help="task cache directory (see benchmark.make_tasks)")
    ap.add_argument("--rerun", "--force", dest="rerun", action="store_true", help="ignore cached predictions and recompute")
    ap.add_argument("--list", action="store_true", help="list contenders and exit")
    ap.add_argument("--list-tests", action="store_true", help="list the runnable test definitions and exit")
    a = ap.parse_args(argv)

    if a.list:
        print("Contenders:"); discover(verbose=True); return 0
    if a.list_tests:
        for d in td.list_defs():
            print(f"{d.name:24s} {d.dataset:10s} {d.target:5s} {d.scenario_key:14s} {d.description}")
        return 0

    if a.test is not None:
        try:
            defs = td.list_defs() if a.test == "all" else [td.load_def(n.strip()) for n in a.test.split(",")]
        except td.TestDefError as exc:
            ap.error(str(exc))
        if not defs:
            ap.error("no runnable test definitions found")
        for d in defs:  # explicit flags must agree with every selected def
            for flag, given, want in (("--dataset", a.dataset, d.dataset), ("--target", a.target, d.target),
                                      ("--scenario", a.scenario, d.scenario_key)):
                if given is not None and given != want:
                    ap.error(f"{flag} {given} conflicts with test {d.name} ({want}); drop {flag} or pick another test")
        found = discover()
        names = list(found) if a.contenders == "all" else a.contenders.split(",")
        out_root = paths.results_root()
        rows = run_tests(defs, found, names, a.cache, out_root, force=a.rerun, skip_missing=(a.test == "all"))
        if rows:
            _write_runs(out_root, rows, next(iter({d.dataset for d in defs})) if len({d.dataset for d in defs}) == 1 else "tests")
        print(f"scores written under {out_root}")
        return 0

    a.dataset, a.target, a.scenario = a.dataset or "synthetic", a.target or "RHOB", a.scenario or SCENARIO_MISSING
    if a.scenario not in SCENARIO_CHOICES:
        ap.error(f"unknown scenario {a.scenario}; choose from {', '.join(SCENARIO_CHOICES)}")
    targets = list(TARGET_CURVES) if a.target == "all" else [a.target]
    if targets[0] not in CORE and a.target != "all":
        ap.error(f"unknown target {a.target}; choose from {', '.join(CORE)} or 'all'")
    scenarios = parse_scenarios(a.scenario)
    if scenarios != [SCENARIO_MISSING]:
        if any(t not in TARGET_CURVES for t in targets):
            ap.error(f"gap scenarios support {', '.join(TARGET_CURVES)}, not {a.target}")
        if not mt.has_spec(a.dataset):
            ap.error(f"dataset {a.dataset} has no task spec; only --scenario missing_curve is available")

    found = discover()
    names = list(found) if a.contenders == "all" else a.contenders.split(",")
    if a.seed != 0 and mt.has_spec(a.dataset):
        print(f"warning: --seed {a.seed} ignored; dataset {a.dataset} uses its committed task spec "
              f"(seed {mt.load_spec(a.dataset)['seed']})", file=sys.stderr)
    train, test = load_dataset(a.dataset, a.seed)
    truth = _truth(test)
    spec_targets = mt.load_spec(a.dataset)["targets"] if mt.has_spec(a.dataset) else []
    tasks: list[Task] = []
    for target in targets:
        if target in spec_targets:
            tasks += mt.load_tasks(a.dataset, a.cache, a.scenario, target, wells=(train, test))
        else:
            tasks += mt.build_tasks({"dataset": a.dataset, "seed": a.seed, "targets": [target], "gap_lengths_m": []}, test)[0]
    print(f"dataset={a.dataset} target={a.target} scenario={a.scenario} train_wells={len(train)} "
          f"test_wells={len(test)} tasks={len(tasks)}")
    out_root = paths.results_root()
    rows = []
    for n in names:
        if n not in found:
            print(f"unknown contender: {n}", file=sys.stderr); continue
        got = []
        for target in targets:
            got += run_model(n, found[n], train, tasks, truth, target, scenarios, a.dataset, out_root, force=a.rerun)
        if not got:
            print(f"  {n:24s} skipped (target not supported or no test tasks for {a.target}/{a.scenario})")
        _print_rows(n, got)
        rows += got
    if rows:
        _write_runs(out_root, rows, a.dataset)
    print(f"scores written under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
