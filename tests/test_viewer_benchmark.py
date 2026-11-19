"""Read-only benchmark endpoints of the viewer, on a temporary results folder with tiny synthetic files."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest

from benchmark import run
from wmb import viewer
from wmb.contender import Prediction
from wmb.contract import Task


def _write(root, contender, dataset, target, scenario, preds=True, *, well="W1", mae=0.1, r2=0.9):
    out = root / contender / dataset / target / scenario
    out.mkdir(parents=True)
    row = {"contender": contender, "dataset": dataset, "target": target, "scenario": scenario,
           "MAE": mae, "RMSE": 0.2, "R2": r2, "hardware": {"cpu_count": 1}}
    (out / "scores.json").write_text(json.dumps(row))
    if preds:
        (out / "predictions.json").write_text(json.dumps({
            "contender": contender, "dataset": dataset, "target": target, "scenario": scenario,
            "wells": [{"key": well, "well_id": well, "depth": [0, 1], "true": [1.0, None], "p50": [1.1, 1.2],
                       "p05": None, "p95": None, "std": [0.1, 0.1], "gaps": [[1, 1]]}]}))


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setenv("WMB_RESULTS", str(tmp_path))
    return tmp_path


@pytest.fixture
def base_url(results):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), viewer.ViewerHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serves_shared_viewer_scripts(base_url):
    html = urllib.request.urlopen(base_url + "/wellmap.html").read().decode()
    for name in ("plot_utils.js", "plot_style.js", "benchmark_quality.js", "benchmark.js", "well_filters.js", "prediction.js"):
        assert f'src="{name}"' in html
        with urllib.request.urlopen(base_url + "/" + name) as r:
            body = r.read()
        assert r.status == 200 and (b"function" in body or name.endswith(".js"))


def test_empty_results_list_is_200_and_empty(base_url):
    assert _get(base_url + "/api/benchmark/runs") == (200, {"rows": []})


def test_runs_and_prediction_happy_path(results, base_url):
    well = "29/6-1"
    _write(results, "m1", "ds", "RHOB", "missing_curve", well=well)
    status, body = _get(base_url + "/api/benchmark/runs")
    assert status == 200 and len(body["rows"]) == 1
    assert body["rows"][0]["contender"] == "m1" and body["rows"][0]["MAE"] == 0.1
    q = "contender=m1&dataset=ds&target=RHOB&scenario=missing_curve"
    status, body = _get(f"{base_url}/api/benchmark/prediction?{q}&well={quote(well, safe='')}")
    assert status == 200 and body["well"]["p50"] == [1.1, 1.2] and body["well"]["gaps"] == [[1, 1]]
    assert _get(f"{base_url}/api/benchmark/prediction?{q}&well=*")[1] == {"wells": [well]}


def test_runs_return_valid_json_for_undefined_r2_and_sort_perfect_mae_first(results, base_url):
    _write(results, "m1", "ds", "RHOB", "missing_curve")
    _write(results, "m2", "ds", "RHOB", "missing_curve", mae=0, r2=float("nan"))
    with urllib.request.urlopen(base_url + "/api/benchmark/runs") as response:
        body = response.read()
    assert b"NaN" not in body
    perfect, other = json.loads(body)["rows"]
    assert perfect["MAE"] == 0 and perfect["R2"] is None
    assert other["R2"] == 0.9


def test_missing_prediction_is_404(results, base_url):
    _write(results, "m1", "ds", "RHOB", "missing_curve", preds=False)
    q = "contender=m1&dataset=ds&target=RHOB&scenario=missing_curve"
    assert _get(f"{base_url}/api/benchmark/prediction?{q}&well=W1")[0] == 404
    _write(results, "m2", "ds", "RHOB", "missing_curve")
    assert _get(f"{base_url}/api/benchmark/prediction?contender=m2&dataset=ds&target=RHOB&scenario=missing_curve&well=NOPE")[0] == 404


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "a%5Cb", "%2e%2e"])
def test_path_traversal_rejected(results, base_url, bad):
    (results.parent / "secret").mkdir(exist_ok=True)
    _write(results, "m1", "ds", "RHOB", "missing_curve")
    q = f"contender={bad}&dataset=ds&target=RHOB&scenario=missing_curve&well=W1"
    assert _get(f"{base_url}/api/benchmark/prediction?{q}")[0] in (400, 404)


@pytest.mark.parametrize("failure", ["predict", "export"])
def test_run_writes_predictions_and_removes_them_on_failed_rerun(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("WMB_PREDICT_CACHE", str(tmp_path / "predict-cache"))

    class Model:
        name = "m1"

        def predict_task(self, task):
            return Prediction(mean=np.array([np.nan, 1.2, 1.3, 1.4, 1.5]))

        def predict_batch(self, tasks):
            return [self.predict_task(task) for task in tasks]

    depth = np.array([0.0004, 1.0004, 2.0004, 3.0004, 4.0004])
    mask = np.array([False, True, True, False, True])
    task = Task("29/6-1", depth, pd.DataFrame(), "RHOB", mask, "gap", 10)
    truth = {(task.well_id, "RHOB"): np.array([np.nan, 1.1, 1.2, 1.3, 1.4]),
             (task.well_id, "DEPTH"): depth}
    base = {"contender": "m1", "dataset": "synthetic", "target": "RHOB"}
    model = Model()
    out = tmp_path / "m1" / "synthetic" / "RHOB" / "gap_10"
    pred = out / "predictions.json"
    assert "error" not in run.run_scenario(model, [task], truth, base, tmp_path, {})
    (well,) = json.loads(pred.read_text())["wells"]
    assert well["true"] == [None, 1.1, 1.2, 1.3, 1.4]
    assert well["p50"] == [None, 1.2, 1.3, 1.4, 1.5]
    assert well["gaps"] == [well["depth"][1:3], [well["depth"][-1]] * 2]

    def fail(*args):
        raise ValueError("synthetic failure")

    if failure == "predict":
        monkeypatch.setattr(model, "predict_task", fail)
    else:
        monkeypatch.setattr(run, "prediction_payload", fail)
    row = run.run_scenario(model, [task], truth, base, tmp_path, {}, force=True)
    assert (out / "scores.json").exists() == (failure == "export")
    assert not pred.exists()


def test_hardware_and_git_commit_returned_when_present_and_absent_otherwise(results, base_url):
    _write(results, "m1", "ds", "RHOB", "missing_curve")
    out = results / "m2" / "ds" / "RHOB" / "missing_curve"
    out.mkdir(parents=True)
    (out / "scores.json").write_text(json.dumps({
        "contender": "m2", "dataset": "ds", "target": "RHOB", "scenario": "missing_curve", "MAE": 0.2,
        "hardware": {"system": "Windows", "cpu_count": 16}, "git_commit": "abc1234", "git_commit_full": "abc1234" + "0" * 33,
    }))
    rows = {r["contender"]: r for r in _get(base_url + "/api/benchmark/runs")[1]["rows"]}
    assert rows["m2"]["hardware"] == {"system": "Windows", "cpu_count": 16}
    assert rows["m2"]["git_commit"] == "abc1234" and rows["m2"]["git_commit_full"].startswith("abc1234")
    assert rows["m1"]["hardware"] == {"cpu_count": 1}
    assert "git_commit" not in rows["m1"]


def test_run_scenario_writes_git_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "git_info", lambda: {"git_commit": "abc1234", "git_commit_full": "abc1234" + "0" * 33})

    class Model:
        name = "m1"

        def predict_task(self, task):
            return Prediction(mean=np.array([1.2, 1.3]))

        def predict_batch(self, tasks):
            return [self.predict_task(task) for task in tasks]

    depth = np.array([0.0, 1.0])
    task = Task("W", depth, pd.DataFrame(), "RHOB", np.array([True, True]), "gap", 10)
    truth = {("W", "RHOB"): np.array([1.1, 1.2]), ("W", "DEPTH"): depth}
    row = run.run_scenario(Model(), [task], truth, {"contender": "m1", "dataset": "synthetic", "target": "RHOB"}, tmp_path, {})
    assert "error" not in row
    saved = json.loads((tmp_path / "m1" / "synthetic" / "RHOB" / "gap_10" / "scores.json").read_text())
    assert saved["git_commit"] == "abc1234" and saved["git_commit_full"].startswith("abc1234")


def test_git_info_soft_fails_without_git(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    run.git_info.cache_clear()
    monkeypatch.setattr(run.subprocess, "run", boom)
    assert run.git_info() == {}
    run.git_info.cache_clear()


def test_optional_test_field_keeps_separate_rows(results, base_url):
    """An optional `test` on scores.json (and a deeper folder) must not collapse rows."""
    _write(results, "m1", "ds", "RHOB", "missing_curve", mae=0.3)
    # New layout: .../scenario/<test>/scores.json
    deep = results / "m1" / "ds" / "RHOB" / "missing_curve" / "official"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "scores.json").write_text(json.dumps({
        "contender": "m1", "dataset": "ds", "target": "RHOB", "scenario": "missing_curve",
        "test": "official", "MAE": 0.05, "RMSE": 0.1, "R2": 0.95,
    }))
    status, body = _get(base_url + "/api/benchmark/runs")
    assert status == 200
    tests = sorted((r.get("test") or "") for r in body["rows"])
    assert tests == ["", "official"]
    assert {r["MAE"] for r in body["rows"]} == {0.3, 0.05}
