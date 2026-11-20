"""Optional binned forecasts (Prediction.bin_edges / bin_probs): validation, cache, prediction files, RPS. Synthetic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark import run  # noqa: E402
from wmb import predict_cache, prediction_files as pf, scoring  # noqa: E402
from wmb.contender import Prediction  # noqa: E402
from wmb.contract import SCENARIO_GAP, Task, validate_prediction  # noqa: E402
from wmb.viewer import load_benchmark_prediction  # noqa: E402

EDGES = np.round(np.arange(16, 30) / 10.0, 2)          # 14 internal edges -> 15 bins


def make_task(n=80, well="W-1"):
    depth = 1000 + 0.5 * np.arange(n)
    mask = np.zeros(n, bool)
    mask[20:50] = True
    observed = np.where(mask, np.nan, 2.4)
    return Task(well, depth, pd.DataFrame({"DEPTH": depth}), "RHOB", mask, SCENARIO_GAP, 10.0, observed)


def make_probs(task, seed=0):
    rng = np.random.default_rng(seed)
    p = np.full((len(task.depth), len(EDGES) + 1), np.nan)
    m = task.predict_mask
    raw = rng.dirichlet(np.ones(len(EDGES) + 1), int(m.sum()))
    p[m] = raw
    return p


def binned(task, seed=0):
    p50 = np.where(task.predict_mask, 2.35, np.nan)
    return Prediction(p50, p05=p50 - 0.2, p95=p50 + 0.2, bin_edges=EDGES, bin_probs=make_probs(task, seed))


def test_bins_pass_validation_and_are_renormalised_copies():
    t = make_task()
    pred = binned(t)
    pred.bin_probs[t.predict_mask] *= 1.0005                     # inside the tolerance: renormalised
    out = validate_prediction(t, pred)
    rows = np.isfinite(out.bin_probs).all(axis=1)
    assert (rows == t.predict_mask).all()
    assert np.allclose(out.bin_probs[rows].sum(axis=1), 1.0)
    pred.bin_probs[:] = 0.0                                      # a snapshot, not a view
    assert np.isfinite(out.bin_probs[rows]).all() and out.bin_probs[rows].sum() > 0
    plain = validate_prediction(t, Prediction(np.full(len(t.depth), 2.3)))
    assert plain.bin_edges is None and plain.bin_probs is None


@pytest.mark.parametrize("change, message", [
    (lambda p: setattr(p, "bin_edges", None), "together"),
    (lambda p: setattr(p, "bin_edges", EDGES[::-1]), "increasing"),
    (lambda p: setattr(p, "bin_probs", p.bin_probs[:, :-1]), "shape"),
    (lambda p: p.bin_probs.__setitem__((25, 3), np.nan), "mixed"),
    (lambda p: p.bin_probs.__setitem__((25, slice(None)), 0.5), "sum to 1"),
    (lambda p: p.bin_probs.__setitem__((25, slice(0, 2)), [-0.2, 1.2]), r"\[0, 1\]"),
])
def test_bad_bins_are_rejected(change, message):
    t = make_task()
    pred = binned(t)
    pred.bin_probs[25] = 1.0 / (len(EDGES) + 1)
    change(pred)
    with pytest.raises(ValueError, match=message):
        validate_prediction(t, pred)


def test_cache_record_round_trips_bins_sparsely():
    t = make_task()
    pred = validate_prediction(t, binned(t))
    rec = predict_cache.record_from_prediction("k", pred, model="m")
    assert len(rec["bins"]["index"]) == int(t.predict_mask.sum())    # only rows with a forecast are stored
    back = validate_prediction(t, predict_cache.prediction_from_record(json.loads(json.dumps(rec))))
    assert np.array_equal(back.bin_edges, pred.bin_edges)
    assert np.allclose(back.bin_probs, pred.bin_probs, equal_nan=True)
    assert "bins" not in predict_cache.record_from_prediction("k", validate_prediction(t, Prediction(pred.p50)))


def test_ranked_probability_score_by_hand():
    edges = np.array([1.0, 2.0])                                 # bins (-inf,1), [1,2), [2,inf)
    p = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]])
    y = np.array([1.5, 2.5, 2.0, np.nan])                         # 2.0 sits on an edge: the higher bin
    r = scoring.ranked_probability_score(y, edges, p)
    assert r[0] == 0.0
    assert r[1] == pytest.approx(((0.5 - 0) ** 2 + (1.0 - 0) ** 2) / 2)
    assert r[2] == pytest.approx((0 + 1.0) / 2)
    assert np.isnan(r[3])


def test_score_tasks_adds_rps_only_when_bins_exist():
    tasks = [make_task(well="A"), make_task(well="B")]
    truth = {(t.well_id, "RHOB"): np.full(len(t.depth), 2.45) for t in tasks}
    preds = [validate_prediction(t, binned(t, i)) for i, t in enumerate(tasks)]
    row = run.score_tasks(tasks, preds, truth)
    want = np.concatenate([scoring.ranked_probability_score(truth[(t.well_id, "RHOB")][t.predict_mask], EDGES,
                                                            p.bin_probs[t.predict_mask]) for t, p in zip(tasks, preds)])
    assert row["RPS_bins"] == pytest.approx(want.mean())
    assert row["n_bins"] == row["n"] == 60 and row["bin_cover"] == 1.0
    assert row["bin_edges"] == [float(e) for e in EDGES]
    plain = [validate_prediction(t, Prediction(p.p50, p05=p.p05, p95=p.p95)) for t, p in zip(tasks, preds)]
    assert "RPS_bins" not in run.score_tasks(tasks, plain, truth)


def write_run(tmp_path, with_bins=True):
    tasks = [make_task(well="A"), make_task(well="B")]
    truth = {(t.well_id, "RHOB"): np.full(len(t.depth), 2.45) for t in tasks}
    preds = [validate_prediction(t, binned(t, i) if with_bins else Prediction(np.full(len(t.depth), 2.3)))
             for i, t in enumerate(tasks)]
    base = {"contender": "m", "test": "t", "dataset": "ds", "target": "RHOB"}
    for name, fmt in (("mj", "json"), ("mn", "npz")):
        pf.write(tmp_path / name / "ds" / "RHOB" / "gap_10", fmt, base, "gap_10", tasks, preds, truth)
    return tasks, preds


def test_both_file_formats_serve_the_same_bins(tmp_path):
    tasks, preds = write_run(tmp_path)
    get = lambda m, w: load_benchmark_prediction(m, "ds", "RHOB", "gap_10", w, tmp_path)["well"]  # noqa: E731
    for t, p in zip(tasks, preds):
        j, n = get("mj", t.well_id), get("mn", t.well_id)
        assert json.dumps(j, sort_keys=True) == json.dumps(n, sort_keys=True)
        b = j["bins"]
        assert b["edges"] == [float(e) for e in EDGES]
        assert b["rows"] == np.flatnonzero(t.predict_mask).tolist()
        assert np.allclose(b["probs"], p.bin_probs[t.predict_mask], atol=5e-4 + 1e-12)


def test_files_without_bins_are_unchanged(tmp_path):
    write_run(tmp_path, with_bins=False)
    w = load_benchmark_prediction("mn", "ds", "RHOB", "gap_10", "A", tmp_path)["well"]
    assert "bins" not in w
    assert "bins" not in load_benchmark_prediction("mj", "ds", "RHOB", "gap_10", "A", tmp_path)["well"]
