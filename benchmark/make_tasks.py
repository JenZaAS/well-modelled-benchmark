"""Build the benchmark tasks from a committed spec and cache them as reloadable files.

    python -m benchmark.make_tasks --dataset synthetic --cache <dir>
    python -m benchmark.make_tasks --dataset force2020
    python -m benchmark.make_tasks --dataset force2020 --write-spec   # write benchmark/task_specs/force2020.windows.yaml

The spec (benchmark/task_specs/<dataset>.yaml) holds seed, targets, gap lengths and optionally the
well lists and gap windows (well, target, gap_length_m, start_md, end_md). Windows listed in the spec
are used as given; missing ones are placed with wmb.splits.place_gap and gap_seed. Cache layout:
<cache>/<dataset>/index.json plus one .npz per task. Only the spec is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.registry import REPO  # also puts src/ on sys.path
from wmb import paths, splits, synthetic
from wmb.contract import (GAP_LENGTHS_M, SCENARIO_GAP, SCENARIO_MISSING, TARGET_CURVES, Task, scenario_key,
                          task_input_columns)

SPEC_DIR = REPO / "benchmark" / "task_specs"
DEFAULT_SEED = 2020
INDEX_VERSION = 1
DEPTH_TOL = 1e-3


def spec_path(dataset: str) -> Path:
    return SPEC_DIR / f"{dataset}.yaml"


def load_spec(dataset: str, path: str | Path | None = None) -> dict:
    """Read the spec; a sibling <name>.windows.yaml (if present) adds well lists and gap windows."""
    import yaml

    p = Path(path) if path else spec_path(dataset)
    if not p.is_file():
        raise FileNotFoundError(f"no task spec for dataset '{dataset}' at {p}")
    spec = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    extra = p.with_suffix(".windows.yaml")
    if extra.is_file():
        spec.update(yaml.safe_load(extra.read_text(encoding="utf-8")) or {})
    spec.setdefault("dataset", dataset)
    spec.setdefault("seed", DEFAULT_SEED)
    spec.setdefault("targets", list(TARGET_CURVES))
    spec.setdefault("gap_lengths_m", [int(x) for x in GAP_LENGTHS_M])
    return spec


def has_spec(dataset: str) -> bool:
    return spec_path(dataset).is_file()


def cache_root(cache: str | Path | None = None) -> Path:
    """--cache, else WMB_TASK_CACHE, else <data cache_dir>/tasks, else <repo>/cache/tasks."""
    if cache:
        return Path(cache).expanduser()
    if os.environ.get("WMB_TASK_CACHE"):
        return Path(os.environ["WMB_TASK_CACHE"]).expanduser()
    try:
        return paths.cache_dir() / "tasks"
    except (paths.DataPathsError, SystemExit):
        return REPO / "cache" / "tasks"


def load_wells(dataset: str, spec: dict) -> tuple[list, list]:
    """(train_wells, test_wells) for a spec dataset. Whole-well split; explicit lists in the spec win."""
    if dataset == "synthetic":
        s = spec.get("synthetic", {})
        wells = synthetic.make_wells(s.get("n_wells", 8), s.get("n_samples", 1500), s.get("wells_seed", 0))
        default = splits.well_split(wells, s.get("test_fraction", 0.25), s.get("split_seed", 0))
    elif dataset == "force2020":
        from wmb.datasets import load_force2020
        default = splits.official_split(load_force2020(paths.get_dataset_path("force2020")))
    else:
        raise ValueError(f"no loader for spec dataset '{dataset}'")
    train, test = default
    if spec.get("train_wells") and spec.get("test_wells"):
        overlap = sorted(set(spec["train_wells"]) & set(spec["test_wells"]))
        if overlap:
            raise ValueError(f"spec train_wells and test_wells overlap: {overlap[:5]}{' ...' if len(overlap) > 5 else ''}")
        by = {w.name: w for w in train + test}
        missing = sorted(n for n in spec["train_wells"] + spec["test_wells"] if n not in by)
        if missing:
            raise ValueError(f"spec lists wells not found in the data: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if dataset == "force2020":
            bad_tr = sorted(set(spec["train_wells"]) - {w.name for w in train})
            bad_te = sorted(set(spec["test_wells"]) - {w.name for w in test})
            if bad_tr or bad_te:
                raise ValueError(f"force2020 spec wells disagree with the official split: train wells not official train "
                                 f"{bad_tr[:5]}, test wells not official test {bad_te[:5]}")
        train, test = [by[n] for n in spec["train_wells"]], [by[n] for n in spec["test_wells"]]
    return train, test


def _inputs_frame(well, target: str) -> pd.DataFrame:
    n = len(well.depth)
    cols = {}
    for c in task_input_columns(target):
        if c == "DEPTH":
            cols[c] = np.asarray(well.depth, float)
        else:
            cols[c] = np.asarray(well.curves[c], float) if c in well.curves else np.full(n, np.nan)
    return pd.DataFrame(cols)


def _window_mask(depth: np.ndarray, start_md: float, end_md: float) -> np.ndarray:
    return (depth >= start_md - DEPTH_TOL) & (depth <= end_md + DEPTH_TOL)


def _check_window(well, target: str, length: int, depth: np.ndarray, mask: np.ndarray, g: dict) -> None:
    """A listed window must be depth-continuous and about `length` m long; otherwise the spec needs regenerating."""
    label = f"gap window for {well.name} {target} {length} m ({g['start_md']}-{g['end_md']})"
    if not mask.any():
        raise ValueError(f"{label} contains no samples; regenerate the spec (--reroll --write-spec)")
    idx = np.nonzero(mask)[0]
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    step = splits.max_gap_step(depth)
    span = float(depth[i1 - 1] - depth[i0])
    if not splits.window_is_continuous(depth, i0, i1, step):
        raise ValueError(f"{label} jumps over a depth hole (> {step:.2f} m step); regenerate the spec (--reroll --write-spec)")
    if span < length - 1e-3 or span > length + step + 1e-3:
        raise ValueError(f"{label} spans {span:.3f} m, expected {length} to {length + step:.2f} m; "
                         f"regenerate the spec (--reroll --write-spec)")


def build_tasks(spec: dict, test_wells: list) -> tuple[list[Task], list[dict]]:
    """All tasks for the test wells, plus the resolved gap windows (for writing back into a spec)."""
    listed = {(g["well"], g["target"], int(g["gap_length_m"])): g for g in spec["gaps"]} if spec.get("gaps") is not None else None
    tasks, windows = [], []
    for well in sorted(test_wells, key=lambda w: w.name):
        for target in spec["targets"]:
            if not well.has(target):
                continue
            base = dict(well_id=well.name, depth=np.asarray(well.depth, float), inputs=_inputs_frame(well, target),
                        target=target, dataset=spec["dataset"], split=well.split)
            n = len(well.depth)
            tasks.append(Task(predict_mask=np.ones(n, bool), scenario=SCENARIO_MISSING, **base))
            y = np.asarray(well.curves[target], float)
            for length in spec["gap_lengths_m"]:
                length = int(length)
                if listed is not None:
                    g = listed.get((well.name, target, length))
                    if g is None:
                        continue
                    mask = _window_mask(base["depth"], float(g["start_md"]), float(g["end_md"]))
                    _check_window(well, target, length, base["depth"], mask, g)
                else:
                    gap = splits.place_gap(base["depth"], y, length, splits.gap_seed(spec["seed"], well.name, target, length))
                    if gap is None:
                        continue
                    mask = np.zeros(n, bool)
                    mask[gap[0]:gap[1]] = True
                if not mask.any() or not np.isfinite(y[mask]).all():
                    raise ValueError(f"gap window for {well.name} {target} {length} m is outside measured data")
                idx = np.nonzero(mask)[0]
                windows.append({"well": well.name, "target": target, "gap_length_m": length,
                                "start_md": round(float(base["depth"][idx[0]]), 4),
                                "end_md": round(float(base["depth"][idx[-1]]), 4)})
                observed = y.copy()
                observed[mask] = np.nan
                tasks.append(Task(predict_mask=mask, scenario=SCENARIO_GAP, gap_length_m=float(length),
                                  target_observed=observed, **base))
    return tasks, windows


def task_id(task: Task) -> str:
    return f"{task.scenario_key}__{task.target}__{task.well_id}"


def _file_name(tid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tid) + ".npz"


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def data_signature(test_wells: list) -> dict:
    """Compact per-well fingerprint of the depth axes the tasks were built on."""
    sig = {}
    for w in test_wells:
        d = np.ascontiguousarray(np.asarray(w.depth, float))
        sig[w.name] = {"n": int(len(d)), "first": round(float(d[0]), 4) if len(d) else None,
                       "last": round(float(d[-1]), 4) if len(d) else None,
                       "sha": hashlib.sha256(d.tobytes()).hexdigest()[:16]}
    return sig


def save_tasks(tasks: list[Task], spec: dict, out: Path, test_wells: list | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.npz"):
        old.unlink()
    index = {"version": INDEX_VERSION, "dataset": spec["dataset"], "seed": spec["seed"], "spec_hash": _spec_hash(spec),
             "data_signature": data_signature(test_wells or []), "tasks": []}
    for t in tasks:
        tid, fname = task_id(t), _file_name(task_id(t))
        arrays = {"depth": t.depth, "predict_mask": t.predict_mask,
                  "input_columns": np.array([c for c in t.inputs.columns if c != "DEPTH"])}
        arrays.update({f"in_{c}": t.inputs[c].to_numpy(float) for c in t.inputs.columns if c != "DEPTH"})
        if t.target_observed is not None:
            arrays["target_observed"] = t.target_observed
        np.savez_compressed(out / fname, **arrays)
        index["tasks"].append({"id": tid, "file": fname, "well_id": t.well_id, "target": t.target, "scenario": t.scenario,
                               "gap_length_m": t.gap_length_m, "split": t.split})
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


def _read_task(out: Path, entry: dict, dataset: str) -> Task:
    with np.load(out / entry["file"], allow_pickle=False) as z:
        depth = z["depth"]
        frame = {"DEPTH": depth, **{f"{c}": z[f"in_{c}"] for c in z["input_columns"]}}
        obs = z["target_observed"] if "target_observed" in z.files else None
        return Task(well_id=entry["well_id"], depth=depth, inputs=pd.DataFrame(frame), target=entry["target"],
                    predict_mask=z["predict_mask"], scenario=entry["scenario"], gap_length_m=entry["gap_length_m"],
                    target_observed=obs, dataset=dataset, split=entry["split"])


def make_tasks(dataset: str, spec_file: str | Path | None = None, cache: str | Path | None = None,
               overwrite: bool = False, write_spec: bool = False, reroll: bool = False,
               wells: tuple[list, list] | None = None) -> Path:
    """Build and cache the tasks; returns the dataset cache directory.

    An existing cache is kept only if both the spec hash and the depth signature of the test wells
    (`wells` = (train, test) if already loaded, else loaded here) match; otherwise it is rebuilt.
    """
    spec = load_spec(dataset, spec_file)
    if reroll:
        spec["gaps"] = None
    out = cache_root(cache) / dataset
    index = out / "index.json"
    train, test = wells if wells is not None else load_wells(dataset, spec)
    if index.is_file() and not (overwrite or reroll or write_spec):
        old = json.loads(index.read_text(encoding="utf-8"))
        if old.get("spec_hash") == _spec_hash(spec) and old.get("data_signature") == data_signature(test):
            return out
    tasks, windows = build_tasks(spec, test)
    save_tasks(tasks, spec, out, test)
    if write_spec or reroll:
        _write_windows(dataset, spec_file, spec, train, test, windows)
    return out


def _write_windows(dataset: str, spec_file, spec: dict, train: list, test: list, windows: list[dict]) -> Path:
    import yaml

    target = (Path(spec_file) if spec_file else spec_path(dataset)).with_suffix(".windows.yaml")
    body = {"train_wells": sorted(w.name for w in train), "test_wells": sorted(w.name for w in test), "gaps": windows}
    target.write_text("# Resolved wells and gap windows (no curve values). Generated by benchmark.make_tasks --write-spec.\n"
                      + yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=140), encoding="utf-8")
    return target


def _wanted(entry: dict, scenario: str | None, target: str | None) -> bool:
    key = scenario_key(entry["scenario"], entry["gap_length_m"])
    ok_s = scenario in (None, "all") or scenario == key or (scenario == SCENARIO_GAP and entry["scenario"] == SCENARIO_GAP)
    return ok_s and (target in (None, "all") or target == entry["target"])


def load_tasks(dataset: str, cache: str | Path | None = None, scenario: str | None = None,
               target: str | None = None, spec_file: str | Path | None = None,
               wells: tuple[list, list] | None = None) -> list[Task]:
    """Cached tasks (built first if the cache is missing or stale vs the spec or loaded wells), filtered by scenario/target.

    scenario: missing_curve | gap (all lengths) | gap_10 | gap_50 | gap_100 | all/None.
    """
    out = make_tasks(dataset, spec_file, cache, wells=wells)
    entries = json.loads((out / "index.json").read_text(encoding="utf-8"))["tasks"]
    return [_read_task(out, e, dataset) for e in entries if _wanted(e, scenario, target)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build and cache benchmark tasks from a task spec")
    ap.add_argument("--dataset", required=True, help="synthetic | force2020")
    ap.add_argument("--spec", help="spec yaml (default benchmark/task_specs/<dataset>.yaml)")
    ap.add_argument("--cache", help="task cache directory (default: $WMB_TASK_CACHE, <cache_dir>/tasks, <repo>/cache/tasks)")
    ap.add_argument("--overwrite", action="store_true", help="regenerate cached task files from the spec")
    ap.add_argument("--write-spec", action="store_true", help="write resolved wells and gap windows to <spec>.windows.yaml")
    ap.add_argument("--reroll", action="store_true", help="ignore gap windows in the spec and place them again")
    a = ap.parse_args(argv)
    try:
        out = make_tasks(a.dataset, a.spec, a.cache, a.overwrite, a.write_spec, a.reroll)
    except (FileNotFoundError, ValueError, paths.DataPathsError) as exc:
        print(f"make_tasks failed: {exc}", file=sys.stderr)
        return 1
    entries = json.loads((out / "index.json").read_text(encoding="utf-8"))["tasks"]
    keys = sorted({scenario_key(e["scenario"], e["gap_length_m"]) for e in entries})
    print(f"{len(entries)} tasks in {out} ({', '.join(keys)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
