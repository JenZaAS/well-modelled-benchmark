"""Gap placement, task building and caching. Synthetic only."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark import make_tasks as mt  # noqa: E402
from wmb import scoring, splits  # noqa: E402


def test_place_gap_is_deterministic_inside_finite_run_with_length():
    depth = np.arange(0, 400, 0.5)
    y = np.full(depth.size, 2.4)
    y[100:110] = np.nan
    y[500:520] = np.nan
    a = splits.place_gap(depth, y, 50.0, seed=7)
    assert a == splits.place_gap(depth, y, 50.0, seed=7)
    s, e = a
    assert np.isfinite(y[s:e]).all() and depth[e - 1] - depth[s] >= 50.0
    assert depth[e - 1] - depth[s] < 50.0 + 0.5 + 1e-9
    assert splits.place_gap(depth, y, 50.0, seed=8) is not None


def test_place_gap_none_when_impossible():
    depth = np.arange(0, 100, 0.5)
    y = np.full(depth.size, 2.4)
    y[::40] = np.nan
    assert splits.place_gap(depth, y, 50.0, seed=0) is None
    assert splits.place_gap(depth, np.full(depth.size, np.nan), 10.0, seed=0) is None
    assert splits.place_gap(depth, np.full(depth.size, 1.0), 500.0, seed=0) is None


def test_gap_seed_is_stable_and_distinct():
    assert splits.gap_seed(2020, "SYN-01", "RHOB", 10) == splits.gap_seed(2020, "SYN-01", "RHOB", 10)
    assert splits.gap_seed(2020, "SYN-01", "RHOB", 10) != splits.gap_seed(2020, "SYN-01", "RHOB", 50)
    assert splits.gap_seed(0, "SYN-01", "RHOB", 10) == int(hashlib.sha256(b"SYN-01|RHOB|10").hexdigest()[:8], 16)


def test_make_tasks_synthetic_reloads_and_is_reproducible(tmp_path):
    out_a = mt.make_tasks("synthetic", cache=tmp_path / "a")
    out_b = mt.make_tasks("synthetic", cache=tmp_path / "b", overwrite=True)
    ta = mt.load_tasks("synthetic", cache=tmp_path / "a")
    tb = mt.load_tasks("synthetic", cache=tmp_path / "b")
    assert out_a != out_b and len(ta) == len(tb) == 24
    for x, y in zip(ta, tb):
        assert x.well_id == y.well_id and x.scenario_key == y.scenario_key
        assert np.array_equal(x.predict_mask, y.predict_mask)
        assert np.array_equal(x.depth, y.depth)


def test_make_tasks_scenarios_and_hidden_target(tmp_path):
    cache = tmp_path / "c"
    tasks = mt.load_tasks("synthetic", cache=cache)
    assert {t.gap_length_m for t in tasks if t.scenario == "gap"} == {10.0, 50.0, 100.0}
    for t in tasks:
        assert t.target not in t.inputs.columns and "DEPTH" in t.inputs.columns
        if t.scenario == "missing_curve":
            assert t.target_observed is None and t.predict_mask.all()
        else:
            assert np.isnan(t.target_observed[t.predict_mask]).all()
            span = t.depth[t.predict_mask]
            assert span[-1] - span[0] >= t.gap_length_m - 1e-6
    assert {t.scenario_key for t in mt.load_tasks("synthetic", cache=cache, scenario="gap_50")} == {"gap_50"}
    assert {t.target for t in mt.load_tasks("synthetic", cache=cache, target="DTC")} == {"DTC"}
    assert len(mt.load_tasks("synthetic", cache=cache, scenario="gap")) == 18


def test_gap_windows_in_spec_are_inside_measured_data():
    spec = mt.load_spec("synthetic")
    train, test = mt.load_wells("synthetic", spec)
    assert not {w.name for w in train} & {w.name for w in test}
    _, windows = mt.build_tasks(spec, test)
    assert windows == spec["gaps"]


def test_scoring_uses_only_finite_truth():
    y = np.array([1.0, np.nan, 3.0, 4.0])
    p = np.array([1.0, 2.0, 3.0, np.nan])
    s = scoring.regression_scores(y, p, np.ones(4))
    assert s["n"] == 2 and s["MAE"] == 0.0


def test_place_gap_never_spans_a_depth_jump():
    depth = np.concatenate([np.arange(0, 30, 0.5), np.arange(60, 90, 0.5)])
    y = np.full(depth.size, 2.4)
    for seed in range(20):
        s, e = splits.place_gap(depth, y, 10.0, seed)
        assert depth[e - 1] - depth[s] < 11
    assert splits.place_gap(depth, y, 40.0, 0) is None


def _tiny_spec(**extra):
    return {"dataset": "synthetic", "seed": 1, "targets": ["RHOB"], "gap_lengths_m": [10], **extra}


def test_spec_train_test_overlap_rejected():
    spec = mt.load_spec("synthetic")
    train, test = mt.load_wells("synthetic", spec)
    names = [w.name for w in train + test]
    bad = {**spec, "train_wells": names[:3], "test_wells": names[2:5]}
    with pytest.raises(ValueError, match="overlap"):
        mt.load_wells("synthetic", bad)


def test_cache_rebuilt_when_depths_change_and_runner_rejects_shifted_depth(tmp_path):
    from benchmark import run as runner
    spec_file = tmp_path / "s.yaml"  # no listed windows, so gaps are re-placed on the shifted wells
    spec_file.write_text("seed: 3\ntargets: [RHOB]\n")
    train, test = mt.load_wells("synthetic", mt.load_spec("synthetic"))
    first = mt.load_tasks("synthetic", cache=tmp_path, spec_file=spec_file, wells=(train, test))
    shifted = [type(w)(w.name, w.depth + 5.0, w.curves, w.dataset, w.split, w.meta) for w in test]
    again = mt.load_tasks("synthetic", cache=tmp_path, spec_file=spec_file, wells=(train, shifted))
    assert np.allclose(again[0].depth, first[0].depth + 5.0)
    truth = {(w.name, "DEPTH"): w.depth for w in test}
    with pytest.raises(ValueError, match="depth values differ"):
        runner._check_task_depths(again, truth)
    runner._check_task_depths(first, truth)
