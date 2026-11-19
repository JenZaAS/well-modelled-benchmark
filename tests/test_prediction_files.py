"""Viewer prediction files: the faster JSON is byte-identical to the old loops, and npz reads back as the same JSON."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark import run  # noqa: E402
from wmb import prediction_files as pf  # noqa: E402
from wmb.contender import Prediction  # noqa: E402
from wmb.contract import Task  # noqa: E402
from wmb.viewer import load_benchmark_prediction  # noqa: E402


# ---------- the previous implementation, kept here as the reference ----------

def old_arr(v, nd=4):
    if v is None:
        return None
    return [round(float(x), nd) if np.isfinite(x) else None for x in np.asarray(v, float)]


def old_gaps(depth, mask):
    out, start = [], None
    for i, m in enumerate(np.asarray(mask, bool)):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i if m else i - 1
            out.append([round(float(depth[start]), 4), round(float(depth[end]), 4)])
            start = None
    return out


def dump(x):
    return json.dumps(x, separators=(",", ":"), allow_nan=False)


def tricky_values(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    ties = (np.arange(-500, 500) + 0.5) / 1e4                          # x.xxxx5: rounding ties in decimal
    special = [0.0, -0.0, 0.03125, -0.03125, 2.675, 1.00005, 1e-5, -1e-5, 5e-324, 1e300, -1e300, 123456.78905,
               2.0 ** 33 / 1e4 + 0.5e-4, np.nan, np.inf, -np.inf]
    mags = 10.0 ** rng.uniform(-6, 9, n) * rng.choice([-1, 1], n)
    return np.concatenate([ties, special, mags, rng.normal(2.4, 0.2, n), np.round(rng.uniform(0, 300, n), 5)])


@pytest.mark.parametrize("nd", [2, 4])
def test_json_values_equal_python_round(nd):
    a = tricky_values()
    assert dump(pf.json_values(a, nd)) == dump(old_arr(a, nd))
    assert pf.json_values(None) is None and pf.json_values([]) == []


def test_gap_intervals_equal_the_loop():
    rng = np.random.default_rng(1)
    depth = 1000 + 0.152 * np.arange(300)
    masks = [np.zeros(300, bool), np.ones(300, bool), rng.random(300) < 0.3, rng.random(300) < 0.9]
    masks += [np.r_[np.zeros(299, bool), True], np.r_[True, np.zeros(299, bool)]]
    for m in masks:
        assert pf.gap_intervals(depth, m) == old_gaps(depth, m)


# ---------- a small but complete run: duplicates, NaN, band vs std, context ----------

def make_run(band=True):
    rng = np.random.default_rng(2)
    tasks, preds, truth = [], [], {}
    for k, wid in enumerate(["29/6-1", "W 2", "29/6-1"]):                 # the same well twice -> key "29/6-1#1"
        n = {"29/6-1": 120, "W 2": 130}[wid]                               # a repeated well keeps its depth axis
        depth = np.round(1500 + 0.1524 * np.arange(n), 4)
        y = rng.normal(2.4, 0.1, n)
        y[:7] = np.nan
        mask = np.zeros(n, bool)
        mask[30:60] = True
        mask[90:] = k == 1
        truth.update({(wid, "RHOB"): y, (wid, "DEPTH"): depth, (wid, "GR"): rng.uniform(20, 150, n),
                      (wid, "RDEP"): 10 ** rng.uniform(-1, 3, n), (wid, "CALI"): np.full(n, np.nan)})
        p50 = y + rng.normal(0, 0.05, n)
        p50[50:55] = np.nan
        std = np.where(np.isfinite(p50), 0.1, np.nan)
        pred = Prediction(p50, std, p05=p50 - 0.16, p95=p50 + 0.16) if band else Prediction(p50, std)
        tasks.append(Task(wid, depth, pd.DataFrame({"DEPTH": depth}), "RHOB", mask, "gap", 10.0))
        preds.append(pred)
    base = {"contender": "m1", "test": "t1", "dataset": "ds", "target": "RHOB"}
    return base, tasks, preds, truth


def old_payload(base, key, tasks, preds, truth):
    seen, wells = {}, []
    for t, p in zip(tasks, preds):
        k = seen.get(t.well_id, 0)
        seen[t.well_id] = k + 1
        has_band = p.p05 is not None and p.p95 is not None
        wells.append({
            "key": t.well_id if k == 0 else f"{t.well_id}#{k}", "well_id": t.well_id,
            "depth": old_arr(t.depth), "true": old_arr(truth[(t.well_id, t.target)]), "p50": old_arr(p.p50),
            "p05": old_arr(p.p05) if has_band else None, "p95": old_arr(p.p95) if has_band else None,
            "std": None if has_band else old_arr(p.std), "gaps": old_gaps(t.depth, t.predict_mask),
            "context": {c: old_arr(v) for c in ("GR", "CALI", "RDEP")
                        if c != t.target and (v := truth.get((t.well_id, c))) is not None and np.isfinite(v).any()}})
    return {"contender": base.get("contender"), "test": base.get("test"), "dataset": base["dataset"],
            "target": base["target"], "scenario": key, "wells": wells}


@pytest.mark.parametrize("band", [True, False])
def test_payload_is_byte_identical_to_the_previous_code(band):
    base, tasks, preds, truth = make_run(band)
    assert dump(run.prediction_payload(base, "gap_10", tasks, preds, truth)) == dump(old_payload(base, "gap_10", tasks, preds, truth))


@pytest.mark.parametrize("band", [True, False])
def test_npz_reads_back_as_the_same_json(tmp_path, band):
    base, tasks, preds, truth = make_run(band)
    for name, fmt in (("mj", "json"), ("mn", "npz")):
        written = pf.write(tmp_path / name / "ds" / "RHOB" / "gap_10", fmt, base, "gap_10", tasks, preds, truth)
        assert written.name == pf.FILE_NAMES[fmt]
    get = lambda m, w: load_benchmark_prediction(m, "ds", "RHOB", "gap_10", w, tmp_path)  # noqa: E731
    keys = get("mj", "*")
    assert keys == get("mn", "*") == {"wells": ["29/6-1", "W 2", "29/6-1#1"]}
    for w in keys["wells"] + ["W 2"]:
        assert dump(get("mj", w)) == dump(get("mn", w))
    assert len(get("mn", "W 2")["well"]["gaps"]) == 2                     # two hidden runs in that task
    with pytest.raises(FileNotFoundError):
        get("mn", "NOPE")


def test_large_and_extreme_values_survive_npz(tmp_path):
    base, tasks, preds, truth = make_run()
    for (wid, curve), v in list(truth.items()):
        if curve == "RDEP":
            truth[(wid, curve)] = np.where(np.arange(len(v)) % 3 == 0, 3e5, 1e12)    # int64 and float64 fallbacks
    pf.write(tmp_path / "mn" / "ds" / "RHOB" / "gap_10", "npz", base, "gap_10", tasks, preds, truth)
    pf.write(tmp_path / "mj" / "ds" / "RHOB" / "gap_10", "json", base, "gap_10", tasks, preds, truth)
    w = lambda m: load_benchmark_prediction(m, "ds", "RHOB", "gap_10", "W 2", tmp_path)  # noqa: E731
    assert dump(w("mj")) == dump(w("mn"))


def test_writing_one_format_removes_the_other_and_viewer_prefers_the_newer(tmp_path):
    base, tasks, preds, truth = make_run()
    out = tmp_path / "m" / "ds" / "RHOB" / "gap_10"
    pf.write(out, "json", base, "gap_10", tasks, preds, truth)
    pf.write(out, "npz", base, "gap_10", tasks, preds, truth)
    assert sorted(p.name for p in out.iterdir()) == ["predictions.npz"]
    (out / "predictions.json").write_text(json.dumps({"contender": "m", "wells": [{"key": "X", "well_id": "X"}]}))
    os.utime(out / "predictions.npz", (time.time() - 60, time.time() - 60))
    assert load_benchmark_prediction("m", "ds", "RHOB", "gap_10", "*", tmp_path) == {"wells": ["X"]}
    pf.remove(out)
    with pytest.raises(FileNotFoundError, match="predictions.json or predictions.npz"):
        load_benchmark_prediction("m", "ds", "RHOB", "gap_10", "*", tmp_path)


def test_resolve_format(monkeypatch):
    monkeypatch.delenv(pf.ENV_VAR, raising=False)
    assert pf.resolve_format() == pf.DEFAULT_FORMAT == "json"
    monkeypatch.setenv(pf.ENV_VAR, "NPZ")
    assert pf.resolve_format() == "npz" and pf.resolve_format("json") == "json"
    for bad in ("parquet", "csv"):
        with pytest.raises(ValueError):
            pf.resolve_format(bad)


@pytest.mark.parametrize("fmt", ["json", "npz"])
def test_run_scenario_writes_the_chosen_format_and_cleans_up_on_failure(tmp_path, monkeypatch, fmt):
    base, tasks, preds, truth = make_run()

    class Model:
        name = "m1"

        def predict_task(self, task):
            return next(p for t, p in zip(tasks, preds) if t is task)

        def predict_batch(self, ts):
            return [self.predict_task(t) for t in ts]

    model = Model()
    out = tmp_path / "m1" / "ds" / "RHOB" / "gap_10"
    other = {"json": "predictions.npz", "npz": "predictions.json"}[fmt]
    out.mkdir(parents=True)
    (out / other).write_text("stale")
    row = run.run_scenario(model, tasks, truth, base, tmp_path, {}, predictions_format=fmt)
    assert "error" not in row and (out / pf.FILE_NAMES[fmt]).is_file() and not (out / other).exists()
    def broken(*args):
        raise ValueError("synthetic export failure")

    if fmt == "npz":
        monkeypatch.setattr(pf, "write_npz", broken)
    else:
        monkeypatch.setattr(run, "prediction_payload", broken)
    row = run.run_scenario(model, tasks, truth, base, tmp_path, {}, predictions_format=fmt)
    assert "error" not in row and (out / "scores.json").is_file()
    assert not any((out / n).exists() for n in pf.FILE_NAMES.values())


def test_cli_flag_and_environment_choose_the_format(tmp_path):
    env = {**os.environ, "WMB_RESULTS": str(tmp_path / "res"), "WMB_TASK_CACHE": str(tmp_path / "tasks"),
           "WMB_PREDICT_CACHE": str(tmp_path / "predict_cache")}
    env.pop(pf.ENV_VAR, None)
    out = tmp_path / "res/template_linear/synthetic/RHOB/missing_curve/synthetic_rhob_missing"   # per-test layout
    cmd = [sys.executable, "-m", "benchmark.run", "--model", "template_linear", "--test", "synthetic_rhob_missing"]
    r = subprocess.run(cmd + ["--predictions", "npz"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "predictions.npz").is_file() and not (out / "predictions.json").exists()
    r = subprocess.run(cmd, cwd=REPO, env={**env, pf.ENV_VAR: "json"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "predictions.json").is_file() and not (out / "predictions.npz").exists()
    bad = subprocess.run(cmd, cwd=REPO, env={**env, pf.ENV_VAR: "parquet"}, capture_output=True, text=True)
    assert bad.returncode == 2 and "predictions format" in bad.stderr
