"""Runner scenarios, result layout and failure isolation. Synthetic only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark import make_tasks as mt  # noqa: E402
from benchmark import run as runner  # noqa: E402
from wmb.contender import BaseContender, Prediction  # noqa: E402
from wmb.curves import CORE  # noqa: E402


def run_cli(tmp_path, *args):
    env = {**os.environ, "WMB_RESULTS": str(tmp_path / "res"), "WMB_TASK_CACHE": str(tmp_path / "tasks")}
    return subprocess.run([sys.executable, "-m", "benchmark.run", "--dataset", "synthetic", *args], cwd=REPO,
                          env=env, capture_output=True, text=True)


def test_missing_curve_writes_scores_in_both_layouts(tmp_path):
    r = run_cli(tmp_path, "--target", "RHOB", "--model", "template_linear", "--scenario", "missing_curve")
    assert r.returncode == 0, r.stderr
    new = json.loads((tmp_path / "res/template_linear/synthetic/RHOB/missing_curve/scores.json").read_text())
    old = json.loads((tmp_path / "res/template_linear/synthetic/RHOB/scores.json").read_text())
    assert new == old and new["has_uncertainty"] and 0.8 < new["cover_90"] <= 1 and new["hardware"]["cpu_count"]
    assert new["samples_per_s"] and new["predict_s_batch"] is not None and new["load_s"] is not None
    assert list((tmp_path / "res/runs").glob("*_synthetic.json"))


def test_gap_scenario_writes_each_length(tmp_path):
    r = run_cli(tmp_path, "--target", "DTC", "--model", "template_linear", "--scenario", "gap")
    assert r.returncode == 0, r.stderr
    for key in ("gap_10", "gap_50", "gap_100"):
        s = json.loads((tmp_path / f"res/template_linear/synthetic/DTC/{key}/scores.json").read_text())
        assert s["scenario"] == key and s["n"] > 0
    only = run_cli(tmp_path, "--target", "DTS", "--model", "template_linear", "--scenario", "gap_50")
    assert only.returncode == 0 and not (tmp_path / "res/template_linear/synthetic/DTS/gap_10").exists()


def test_gap_rejects_non_target_curve(tmp_path):
    assert run_cli(tmp_path, "--target", "GR", "--scenario", "gap").returncode == 2


def test_older_core_target_still_runs_for_missing_curve(tmp_path):
    r = run_cli(tmp_path, "--target", "GR", "--model", "red_floor_train_mean")
    assert r.returncode == 0 and (tmp_path / "res/red_floor_train_mean/synthetic/GR/missing_curve/scores.json").exists()


class Raises(BaseContender):
    name, workstream, required_inputs = "raises", "test", ["GR"]

    def fit(self, train_wells, target):
        pass

    def predict(self, well, target):
        raise RuntimeError("boom")


class WrongLen(Raises):
    name = "wrong_len"

    def predict(self, well, target):
        return Prediction(np.zeros(3))


def test_failures_are_isolated_and_recorded(tmp_path):
    spec = mt.load_spec("synthetic")
    train, test = mt.load_wells("synthetic", spec)
    tasks = mt.build_tasks(spec, test)[0]
    truth = {(w.name, t): w.curves[t] for w in test for t in CORE if t in w.curves}
    truth.update({(w.name, "DEPTH"): w.depth for w in test})
    for cls, text in ((Raises, "boom"), (WrongLen, "length")):
        rows = runner.run_model(cls.name, cls, train, tasks, truth, "RHOB", ["missing_curve"], "synthetic", tmp_path)
        assert len(rows) == 1 and text in rows[0]["error"]
        assert json.loads((tmp_path / cls.name / "synthetic/RHOB/missing_curve/failure.json").read_text())["error"]
        assert not (tmp_path / cls.name / "synthetic/RHOB/missing_curve/scores.json").exists()


def test_scoreboard_reads_both_layouts(tmp_path):
    from benchmark import scoreboard
    run_cli(tmp_path, "--target", "RHOB", "--model", "template_linear", "--scenario", "all")
    os.environ["WMB_RESULTS"] = str(tmp_path / "res")
    try:
        rows = scoreboard.collect()
    finally:
        del os.environ["WMB_RESULTS"]
    assert {r["scenario"] for r in rows} == {"missing_curve", "gap_10", "gap_50", "gap_100"}


class Memorizer(BaseContender):
    """Remembers every observed target value it is shown, and uses it to 'predict' (a leak if shared across scenarios)."""
    name, workstream, required_inputs = "memorizer", "test", ["GR"]

    def __init__(self):
        self.seen = {}

    def fit(self, train_wells, target):
        pass

    def predict_task(self, task):
        n = len(task.depth)
        if task.target_observed is not None:
            for d, v in zip(task.depth, task.target_observed):
                if np.isfinite(v):
                    self.seen[round(float(d), 4)] = v
        out = np.array([self.seen.get(round(float(d), 4), 0.0) for d in task.depth])
        return Prediction(out, np.full(n, 0.1))


def test_scenarios_do_not_leak_between_runs(tmp_path):
    spec = mt.load_spec("synthetic")
    train, test = mt.load_wells("synthetic", spec)
    tasks = mt.build_tasks(spec, test)[0]
    truth = {(w.name, t): w.curves[t] for w in test for t in CORE if t in w.curves}
    truth.update({(w.name, "DEPTH"): w.depth for w in test})
    allrows = runner.run_model("memorizer", Memorizer, train, tasks, truth, "RHOB", runner.parse_scenarios("all"), "synthetic", tmp_path / "a")
    alone = runner.run_model("memorizer", Memorizer, train, tasks, truth, "RHOB", ["gap_50"], "synthetic", tmp_path / "b")
    pick = lambda rows: next(r for r in rows if r["scenario"] == "gap_50")  # noqa: E731
    assert all("error" not in r for r in allrows)
    assert pick(allrows)["MAE"] == pick(alone)["MAE"] and pick(allrows)["fit_s"] is not None


def test_failed_rerun_removes_stale_scores(tmp_path):
    spec = mt.load_spec("synthetic")
    train, test = mt.load_wells("synthetic", spec)
    tasks = mt.build_tasks(spec, test)[0]
    truth = {(w.name, t): w.curves[t] for w in test for t in CORE if t in w.curves}
    truth.update({(w.name, "DEPTH"): w.depth for w in test})
    out = tmp_path / "raises/synthetic/RHOB"
    (out / "missing_curve").mkdir(parents=True)
    (out / "missing_curve/scores.json").write_text("{}")
    (out / "scores.json").write_text("{}")
    runner.run_model("raises", Raises, train, tasks, truth, "RHOB", ["missing_curve"], "synthetic", tmp_path)
    assert not (out / "missing_curve/scores.json").exists() and not (out / "scores.json").exists()


def test_seed_with_spec_warns(tmp_path):
    r = run_cli(tmp_path, "--target", "RHOB", "--model", "template_linear", "--seed", "5")
    assert r.returncode == 0 and "--seed 5 ignored" in r.stderr
