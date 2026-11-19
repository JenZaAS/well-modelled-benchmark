"""Prediction tab back end: /api/predict, /api/predict/models and best-model ranking. Synthetic data only."""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark import predict_api  # noqa: E402
from wmb import viewer  # noqa: E402
from wmb.contender import BaseContender, Prediction  # noqa: E402
from workstreams._template_model.contender import TemplateLinear  # noqa: E402


class NeedsSklearn(BaseContender):
    name, workstream, required_inputs, targets = "needs_dep", "test", ["DTC"], ["RHOB", "DTC", "DTS"]

    def fit(self, train_wells, target):
        raise ModuleNotFoundError("No module named 'sklearn'")


class NeedsPef(NeedsSklearn):
    name, required_inputs = "needs_pef", ["PEF"]


class RhobOnly(NeedsSklearn):
    name, targets = "rhob_only", ["RHOB"]


class Slow(BaseContender):
    name, workstream, required_inputs, targets = "slow", "test", ["DTC"], ["RHOB"]

    def fit(self, train_wells, target):
        pass

    def predict(self, well, target):
        threading.Event().wait(5)
        return Prediction(np.ones(len(well.depth)))


REG = ({c.name: c for c in (TemplateLinear, NeedsSklearn, NeedsPef, RhobOnly, Slow)},
       [{"name": "blue_boosted_trees", "workstream": "blue_boosted_trees", "error": "import failed: ModuleNotFoundError: sklearn"}])


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WMB_RESULTS", str(tmp_path / "results"))
    predict_api.clear_caches()
    monkeypatch.setattr(predict_api, "_registry", lambda: REG)
    yield tmp_path / "results"
    predict_api.clear_caches()


def test_predict_happy_path_with_truth_and_one_failing_model(isolated):
    out = predict_api.predict_well("synthetic", "SYN-06", "RHOB", ["template_linear", "needs_dep", "needs_pef", "nope"], timeout_s=20)
    by = {m["name"]: m for m in out["models"]}
    n = len(out["depth"])
    assert out["well"] == "SYN-06" and out["scenario"] == "missing_curve" and n == 1500
    assert out["true"] is not None and "RHOB" in out["curves"] and "DTC" in out["curves"]
    good = by["template_linear"]
    assert good["ok"] and good["error"] is None and len(good["p50"]) == n
    assert good["mae"] is not None and good["mae"] < 0.2 and good["r2"] is not None
    assert good["inputs_used"]["required"] == ["DTC"] and set(good["inputs_used"]["optional"]) == {"GR", "NPHI"}
    assert good["p05"] is not None and good["runtime_s"] >= 0
    assert not by["needs_dep"]["ok"] and "sklearn" in by["needs_dep"]["error"]
    assert not by["needs_pef"]["ok"] and "missing required curve" in by["needs_pef"]["error"]
    assert not by["nope"]["ok"] and "unknown model" in by["nope"]["error"]
    json.dumps(out, allow_nan=False)
    assert not isolated.exists() or not list(isolated.rglob("*"))


@pytest.mark.parametrize("required, expected", [([], {"DEPTH"}), (["DTC", "EXTRA"], {"DEPTH", "DTC", "EXTRA"})])
def test_model_sees_only_its_declared_inputs(monkeypatch, required, expected):
    seen = {}

    class Spy(NeedsSklearn):
        name, required_inputs, optional_inputs = "spy", required, []

        def fit(self, train_wells, target):
            pass

        def predict_task(self, task):
            seen["finite"] = {c for c in task.inputs.columns if np.isfinite(task.inputs[c]).any()}
            return Prediction(np.full(len(task.depth), 2.4))

    reg = ({"spy": Spy}, [])
    monkeypatch.setattr(predict_api, "_registry", lambda: reg)
    well, _, _ = predict_api.find_well("synthetic", "SYN-06")
    well.curves["EXTRA"] = np.ones(len(well.depth))
    out = predict_api.predict_well("synthetic", "SYN-06", "RHOB", ["spy"])
    assert out["models"][0]["ok"] and seen["finite"] == expected


def test_timed_out_fit_does_not_predict_or_populate_cache(monkeypatch):
    release = threading.Event()
    started = threading.Event()
    predicted = threading.Event()
    finished = threading.Event()

    class LateFit(NeedsSklearn):
        def fit(self, train_wells, target):
            started.set()
            release.wait(5)

        def predict_task(self, task):
            predicted.set()
            return Prediction(np.ones(len(task.depth)))

    monkeypatch.setattr(predict_api, "_registry", lambda: ({"late": LateFit}, []))
    slots = predict_api._model_slots

    class ObservedSlots:
        acquire = slots.acquire

        def release(self):
            slots.release()
            finished.set()

    monkeypatch.setattr(predict_api, "_model_slots", ObservedSlots())
    try:
        out = predict_api.predict_well("synthetic", "SYN-06", "RHOB", ["late"], timeout_s=0.2)
        assert started.is_set() and "timed out" in out["models"][0]["error"]
    finally:
        release.set()
    assert finished.wait(2)
    assert not predicted.is_set() and not predict_api._fits


def test_dataset_uses_viewer_manifest_when_wells_json_is_missing(tmp_path, monkeypatch):
    from wmb import io, paths

    manifest = {"manifest_version": 1, "dataset": {"id": "volve"}, "wells": [
        {"name": name, "lat": 60, "lon": 2, "files": [{"path": f"{name}.las", "format": "las"}]}
        for name in ("W1", "W2")
    ]}
    path = tmp_path / "wells.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(paths, "get_dataset_path", lambda name: tmp_path)
    monkeypatch.setattr(io, "read_las", lambda path, *args, **kwargs: io.Well(path.stem, np.arange(2)))
    well, train, _ = predict_api.find_well("volve", "W1")
    assert well.name == "W1" and all(w.name != "W1" for w in train)
    predict_api.clear_caches()
    manifest["wells"][0]["files"][0]["path"] = "../outside.las"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its dataset"):
        predict_api.find_well("volve", "W1")


def test_system_exit_in_model_does_not_abort_request(monkeypatch):
    def broken():
        raise SystemExit("dependency failed")

    monkeypatch.setattr(predict_api, "_registry", lambda: ({"broken": broken, "template_linear": TemplateLinear}, []))
    out = predict_api.predict_well("synthetic", "SYN-06", "RHOB", ["broken", "template_linear"])
    assert "dependency failed" in out["models"][0]["error"] and out["models"][1]["ok"]


def test_timeout_is_reported_per_model():
    out = predict_api.predict_well("synthetic", "SYN-06", "RHOB", ["slow", "template_linear"], timeout_s=0.3)
    by = {m["name"]: m for m in out["models"]}
    assert "timed out" in by["slow"]["error"] and by["template_linear"]["ok"]


def test_well_in_train_is_excluded_from_fit(monkeypatch):
    seen = {}

    class Spy(TemplateLinear):
        name = "spy_train"

        def fit(self, train_wells, target):
            seen["names"] = {w.name for w in train_wells}
            super().fit(train_wells, target)

    monkeypatch.setattr(predict_api, "_registry", lambda: ({"spy_train": Spy}, []))
    predict_api.predict_well("synthetic", "SYN-01", "RHOB", ["spy_train"])
    assert seen["names"] and "SYN-01" not in seen["names"]


def test_decimation_keeps_payload_small():
    out = predict_api.predict_well("synthetic", "SYN-06", "RHOB", ["template_linear"], max_points=200)
    assert out["decimated"] and len(out["depth"]) == 200 == len(out["models"][0]["p50"]) and out["n_samples"] == 1500


@pytest.mark.parametrize("args, exc", [
    (("synthetic", "NOPE", "RHOB", []), FileNotFoundError),
    (("../x", "SYN-06", "RHOB", []), ValueError),
    (("synthetic", "SYN-06", "BOGUS", []), ValueError),
    (("synthetic", "SYN-06", "RHOB", "template_linear"), ValueError),
])
def test_bad_requests(args, exc):
    with pytest.raises(exc):
        predict_api.predict_well(*args)


def test_list_models_marks_eligibility_and_failures():
    out = predict_api.list_models("synthetic", "SYN-06", "RHOB")
    by = {m["name"]: m for m in out["models"]}
    assert by["template_linear"]["eligible"] and by["template_linear"]["inputs_used"]["required"] == ["DTC"]
    assert by["template_linear"]["inputs"]["optional"] == ["GR", "NPHI"]
    assert not by["needs_pef"]["eligible"] and "PEF" in by["needs_pef"]["reason"]
    assert by["needs_dep"]["eligible"]          # the missing dependency only shows when it runs
    assert not by["blue_boosted_trees"]["eligible"] and "sklearn" in by["blue_boosted_trees"]["reason"]
    assert not predict_api.list_models("synthetic", "SYN-06", "DTS")["models"][3]["eligible"]      # rhob_only
    assert "does not support" in predict_api.list_models("synthetic", "SYN-06", "DTS")["models"][3]["reason"]
    assert not predict_api.list_models(None, None, "RHOB")["models"][0]["eligible"]


# ---------- ranking ----------

def _row(name, run_at="2026-01-01T00:00:00", dataset="ds", target="RHOB", scenario="missing_curve", **m):
    return {"contender": name, "dataset": dataset, "target": target, "scenario": scenario, "run_at": run_at, **m}


ROWS = [
    _row("a", MAE=0.30, RMSE=0.1, R2=0.5, cover_90=0.60),
    _row("b", MAE=0.10, RMSE=0.3, R2=0.9, cover_90=0.97),
    _row("c", MAE=0.20, RMSE=0.2, R2=0.7, cover_90=0.89),
    _row("x", MAE=0.01, RMSE=0.0, R2=1.0, cover_90=0.90, target="DTC"),
    _row("nan", MAE=None, RMSE=float("nan"), R2=None, cover_90=None),
]


@pytest.mark.parametrize("metric, order", [("MAE", "bca"), ("RMSE", "acb"), ("R2", "bca"), ("cover_90", "cba")])
def test_rank_rows_by_metric(metric, order):
    ranked = predict_api.rank_rows(ROWS, "ds", "RHOB", metric=metric)
    assert "".join(r["model"] for r in ranked) == order
    assert [r["rank"] for r in ranked] == [1, 2, 3] and ranked[0]["n"] == 3


def test_best_model_skips_ineligible_and_uses_latest_run():
    assert predict_api.best_model(ROWS, "ds", "RHOB", eligible_names={"a", "c"})["model"] == "c"
    assert predict_api.best_model(ROWS, "ds", "RHOB", eligible_names=set()) is None
    assert predict_api.best_model(ROWS, "other", "RHOB") is None
    assert predict_api.best_model(ROWS, "ds", "RHOB", scenario="gap_10") is None
    rows = ROWS + [_row("a", run_at="2026-02-01T00:00:00", MAE=0.05, RMSE=0, R2=1, cover_90=0.9),
                   _row("a", run_at="2025-01-01T00:00:00", MAE=0.001, RMSE=0, R2=1, cover_90=0.9)]
    best = predict_api.best_model(rows, "ds", "RHOB")
    assert best["model"] == "a" and best["value"] == 0.05 and best["run_at"] == "2026-02-01T00:00:00"
    with pytest.raises(ValueError):
        predict_api.rank_rows(ROWS, "ds", "RHOB", metric="speed")


def test_latest_run_is_chosen_before_filtering_missing_metrics():
    rows = ROWS + [_row("b", run_at="2026-02-01T01:00:00+0200", MAE=0.01),
                   _row("b", run_at="2026-01-31T23:30:00Z", MAE=None)]
    assert predict_api.best_model(rows, "ds", "RHOB")["model"] == "c"


def test_collector_keeps_newest_score_across_layouts(isolated):
    legacy = isolated / "a" / "ds" / "RHOB" / "scores.json"
    current = legacy.parent / "missing_curve" / "scores.json"
    current.parent.mkdir(parents=True)
    legacy.write_text(json.dumps(_row("a", run_at="2026-03-01T00:00:00Z", MAE=0.1)))
    current.write_text(json.dumps(_row("a", run_at="2026-02-01T00:00:00Z", MAE=0.2)))
    row, = viewer.collect_benchmark_rows()
    assert row["MAE"] == 0.1


# ---------- HTTP ----------

class _Remote(viewer.ViewerHandler):
    client_address = property(lambda self: ("10.1.2.3", 5555), lambda self, v: None)


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def base_url():
    srv, url = _serve(viewer.ViewerHandler)
    yield url
    srv.shutdown()
    srv.server_close()


def _req(url, body=None, headers=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    h = {"Content-Type": "application/json", **(headers or {})}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h)) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_predict_and_models(base_url, isolated):
    status, out = _req(base_url + "/api/predict", {"dataset": "synthetic", "well": "SYN-06", "target": "RHOB", "models": ["template_linear", "needs_dep"]})
    assert status == 200
    by = {m["name"]: m for m in out["models"]}
    assert by["template_linear"]["ok"] and by["template_linear"]["mae"] is not None and "sklearn" in by["needs_dep"]["error"]
    status, listing = _req(base_url + "/api/predict/models?dataset=synthetic&well=SYN-06&target=RHOB&metric=MAE")
    assert status == 200 and listing["best"] is None and any(m["name"] == "template_linear" and m["eligible"] for m in listing["models"])
    assert not isolated.exists() or not list(isolated.rglob("*"))


@pytest.mark.parametrize("body, raw, code", [
    ({"dataset": "synthetic", "well": "NOPE", "target": "RHOB"}, None, 404),
    ({"dataset": "synthetic", "target": "RHOB"}, None, 400),
    ({"dataset": "synthetic", "well": "SYN-06", "target": "RHOB", "models": "x"}, None, 400),
    ({"dataset": "synthetic", "well": "SYN-06", "target": "RHOB", "models": {}}, None, 400),
    ({"dataset": "synthetic", "well": "SYN-06", "target": "GR", "models": []}, None, 400),
    (None, b"not json", 400),
])
def test_http_predict_errors(base_url, body, raw, code):
    assert _req(base_url + "/api/predict", body, raw=raw)[0] == code


def test_http_predict_rejects_non_json_content_type_and_foreign_origin(base_url):
    body = {"dataset": "synthetic", "well": "SYN-06", "target": "RHOB", "models": []}
    assert _req(base_url + "/api/predict", body, headers={"Content-Type": "text/plain"})[0] == 400
    assert _req(base_url + "/api/predict", body, headers={"Origin": "http://evil.example"})[0] == 403
    assert _req(base_url + "/api/predict", body, headers={"Origin": "http://localhost:8765"})[0] == 403
    assert _req(base_url + "/api/predict", body, headers={"Host": "not-local.example"})[0] == 403
    assert _req(base_url + "/api/predict", body, headers={"Origin": base_url})[0] == 200


def test_http_predict_rejects_non_localhost_clients():
    srv, url = _serve(_Remote)
    try:
        body = {"dataset": "synthetic", "well": "SYN-06", "target": "RHOB", "models": []}
        assert _req(url + "/api/predict", body)[0] == 403
        assert _req(url + "/api/predict/models?dataset=synthetic&well=SYN-06&target=RHOB")[0] == 403
    finally:
        srv.shutdown()
        srv.server_close()


def test_prediction_js_is_served(base_url):
    with urllib.request.urlopen(base_url + "/prediction.js") as r:
        assert r.status == 200 and b"WmbPrediction" in r.read()


def test_is_local_client():
    assert all(viewer.is_local_client((h, 1)) for h in ("127.0.0.1", "::1", "::ffff:127.0.0.1"))
    assert not viewer.is_local_client(("192.168.1.5", 1))
