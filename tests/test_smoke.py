"""Smoke tests: run with no data (synthetic wells only)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from wmb import scoring, splits, synthetic  # noqa: E402
from wmb.curves import match_property  # noqa: E402


def test_mnemonics():
    assert match_property("DTSM") == "DTS"
    assert match_property("DT") == "DTC"
    assert match_property("RHOZ") == "RHOB"
    assert match_property("DRHO") is None


def test_scoring_perfect_and_coverage():
    y = np.linspace(2.0, 2.6, 1000)
    s = scoring.regression_scores(y, y + np.random.default_rng(0).normal(0, 0.05, y.size), np.full(y.size, 0.05))
    assert s["MAE"] < 0.06 and 0.6 < s["cover_1sigma"] < 0.76


def test_split_is_by_well():
    tr, te = splits.well_split(synthetic.make_wells(), 0.25, 0)
    assert not {w.name for w in tr} & {w.name for w in te}


def test_registry_finds_all_workstreams():
    from benchmark.registry import discover
    found = discover()
    assert {"black_gardner", "blue_gbt", "red_floor_train_mean"} <= set(found)


def test_runner_end_to_end(tmp_path):
    import os
    env = {**os.environ, "WMB_RESULTS": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "benchmark.run", "--dataset", "synthetic", "--target", "RHOB"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    scores = [json.loads(p.read_text()) for p in tmp_path.glob("*/*/*/scores.json")]
    assert len(scores) >= 3
    by = {s["contender"]: s for s in scores}
    assert by["black_gardner"]["MAE"] < by["red_floor_train_mean"]["MAE"]   # physics beats the floor
