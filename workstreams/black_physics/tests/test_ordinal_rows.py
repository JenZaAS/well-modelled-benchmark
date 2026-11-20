"""The registered black_ordinal_* rows: frozen-head checks and the binned Prediction they return. Synthetic only."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from wmb.contender import BaseContender, Prediction  # noqa: E402
from wmb.contract import SCENARIO_MISSING, Task, validate_prediction  # noqa: E402
from workstreams.black_physics.models.ordinal_density import wrapper as W  # noqa: E402
from workstreams.black_physics.ordinal_density import bins as B  # noqa: E402
from workstreams.black_physics.ordinal_density import infer as INF  # noqa: E402


def gap_task(n=400, a=150, b=250, scenario="gap"):
    depth = 1000 + 0.152 * np.arange(n)
    rhob = 2.35 + 0.08 * np.sin(np.arange(n) / 17)
    m = np.zeros(n, bool)
    m[a:b] = True
    obs = np.where(m, np.nan, rhob)
    inputs = pd.DataFrame({"DEPTH": depth, "GR": np.linspace(40, 90, n)})
    if scenario != "gap":
        return Task("W-1", depth, inputs, "RHOB", np.ones(n, bool), SCENARIO_MISSING)
    return Task("W-1", depth, inputs, "RHOB", m, "gap", 50.0, obs, "synthetic", "test")


class FakeBase(BaseContender):
    """A base fill that is the flanks' mean plus a known offset, with a P10/P90 band."""
    name = "fake_base"
    workstream = "test"

    def fit(self, train_wells, target):
        self.provenance = {"checkpoint_sha256": "abc"}

    def predict_task(self, task):
        if task.target_observed is None:
            return Prediction(np.full(len(task.depth), np.nan))
        p50 = np.where(task.predict_mask, np.nanmean(task.target_observed) + 0.01, np.nan)
        return Prediction(p50, quantiles={0.1: p50 - 0.05, 0.5: p50, 0.9: p50 + 0.05})


class Row(W._OrdinalGap):
    name = "black_ordinal_fake"
    base_cls = FakeBase


FORCE = [SimpleNamespace(dataset="force2020")]          # official training wells: the frozen head is used


def freeze(tmp_path, monkeypatch, checkpoint="abc", shift=0.0):
    """A frozen empirical head (residuals around +shift) and its lock, as the cli `freeze` stage writes them."""
    from workstreams.black_physics.ordinal_density import heads as H

    resid = np.linspace(-0.15, 0.15, 61) + shift
    base = np.full(61, 2.3)
    spec = H.EmpiricalHead(by_flank=False).fit(base, base + resid, np.full(61, 0.05), np.ones(61)).as_dict()
    rec = {"model": "fake_base", "run_id": "r", "head": "empirical_pooled|raw", "spec": spec,
           "tails": B.Tails(np.array([1.5]), np.array([3.0]), False, False).as_dict(),
           "code_sha256": INF.code_sha256(), "base_checkpoint_sha256": checkpoint}
    path = tmp_path / "fake_base.json"
    path.write_text(json.dumps(rec, sort_keys=True))
    lock = tmp_path / "frozen_heads.yaml"
    lock.write_text(f"fake_base:\n  sha256: {INF.file_sha256(path)}\n  head: empirical_pooled|raw\n")
    monkeypatch.setattr(INF, "frozen_dir", lambda: tmp_path)
    monkeypatch.setattr(W, "LOCK", lock)
    return path


def test_row_keeps_the_base_fill_and_adds_valid_bins(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    freeze(tmp_path, monkeypatch)
    row = Row()
    row.fit(FORCE, "RHOB")
    task = gap_task()
    pred = validate_prediction(task, row.predict_task(task))
    base = FakeBase().predict_task(task)
    np.testing.assert_array_equal(pred.p50, base.p50)                  # same point estimate as the base row
    rows = np.isfinite(pred.bin_probs).all(axis=1)
    assert (rows == task.predict_mask).all()
    assert np.allclose(pred.bin_probs[rows].sum(axis=1), 1.0)
    np.testing.assert_array_equal(pred.bin_edges, B.EDGES)
    m = task.predict_mask
    assert (pred.p05[m] <= pred.p50[m]).all() and (pred.p95[m] >= pred.p50[m]).all()
    assert np.isnan(pred.p05[~m]).all() and np.isnan(pred.bin_probs[~m]).all()
    assert row.provenance["head"] == "empirical_pooled|raw" and row.provenance["band_widened"] == 0
    assert len(row.predict_batch([task, task])) == 2


def test_finer_bins_refine_the_same_forecast(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    freeze(tmp_path, monkeypatch)
    task = gap_task()
    coarse, fine = Row(), type("FineRow", (Row,), {"bin_step": 0.02})()
    for r in (coarse, fine):
        r.fit(FORCE, "RHOB")
    pc = validate_prediction(task, coarse.predict_task(task))
    pf = validate_prediction(task, fine.predict_task(task))
    m = task.predict_mask
    assert pf.bin_probs.shape[1] == 67 and np.array_equal(pf.bin_edges, B.fine_edges(0.02))
    np.testing.assert_allclose(B.coarsen(pf.bin_probs[m], pf.bin_edges), pc.bin_probs[m], atol=1e-12)
    np.testing.assert_array_equal(pf.p50, pc.p50)
    assert np.isnan(pf.bin_probs[~m]).all()
    assert fine.provenance["bin_step"] == 0.02


def test_band_is_widened_to_contain_the_fill(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    freeze(tmp_path, monkeypatch, shift=0.4)                            # every error positive: q05 above the fill
    row = Row()
    row.fit(FORCE, "RHOB")
    task = gap_task()
    pred = validate_prediction(task, row.predict_task(task))
    m = task.predict_mask
    np.testing.assert_array_equal(pred.p05[m], pred.p50[m])
    assert row.provenance["band_widened"] >= int(m.sum())


def test_whole_curve_tasks_abstain(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    freeze(tmp_path, monkeypatch)
    row = Row()
    row.fit(FORCE, "RHOB")
    task = gap_task(scenario="missing")
    pred = validate_prediction(task, row.predict_task(task))
    assert np.isnan(pred.p50).all() and pred.bin_probs is None


def test_other_wells_read_no_frozen_head(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    path = freeze(tmp_path, monkeypatch)
    path.unlink()                                                       # would raise StaleHead if it were read
    row = Row()
    row.fit([SimpleNamespace(dataset="synthetic")], "RHOB")
    task = gap_task()
    pred = validate_prediction(task, row.predict_task(task))
    m = task.predict_mask
    sigma = 0.05 / 1.2815516                                            # the fake base's P10-P90 is p50 +- 0.05
    np.testing.assert_allclose(pred.p95[m] - pred.p50[m], 1.6448536 * sigma, rtol=1e-6)
    np.testing.assert_allclose(pred.bin_probs[m].sum(axis=1), 1.0)
    assert "not FORCE-trained" in row.provenance["head"]


@pytest.mark.parametrize("problem", ["missing", "edited", "code", "checkpoint", "unlisted"])
def test_stale_or_missing_heads_are_refused(tmp_path, monkeypatch, problem):
    pytest.importorskip("yaml")
    path = freeze(tmp_path, monkeypatch, checkpoint="other" if problem == "checkpoint" else "abc")
    if problem == "missing":
        path.unlink()
    elif problem == "edited":
        path.write_text(path.read_text() + " ")
    elif problem == "code":
        monkeypatch.setattr(INF, "code_sha256", lambda: "0" * 64)
    elif problem == "unlisted":
        W.LOCK.write_text("other_model:\n  sha256: x\n")
    with pytest.raises(INF.StaleHead):
        Row().fit(FORCE, "RHOB")


def test_rows_register_only_with_the_flag(monkeypatch):
    import importlib

    import workstreams.black_physics.contender as C

    monkeypatch.delenv("WMB_BLACK_ORDINAL", raising=False)
    off = importlib.reload(C)
    assert not {c.name for c in off.CONTENDERS} & {"black_ordinal_timestral", "black_ordinal_tspulse_finetune"}
    assert {c.name for c in off.OPT_IN} >= {"black_ordinal_timestral", "black_ordinal_tspulse_finetune"}
    monkeypatch.setenv("WMB_BLACK_ORDINAL", "1")
    on = importlib.reload(C)
    assert {c.name for c in on.CONTENDERS} >= {"black_ordinal_timestral", "black_ordinal_tspulse_finetune"}
    monkeypatch.delenv("WMB_BLACK_ORDINAL")
    importlib.reload(C)


def test_committed_lock_names_both_base_models():
    yaml = pytest.importorskip("yaml")
    lock = yaml.safe_load(W.LOCK.read_text(encoding="utf-8"))
    assert set(lock) == {"red_timestral", "red_tspulse_finetune"}
    for v in lock.values():
        assert len(v["sha256"]) == 64 and len(v["code_sha256"]) == 64 and "|" in v["head"]
