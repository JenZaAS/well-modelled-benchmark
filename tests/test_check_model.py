"""check_model: the template passes, a deliberately bad model fails. Synthetic only."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark.check_model import check_contender  # noqa: E402
from wmb.contender import BaseContender, Prediction  # noqa: E402


class WrongLength(BaseContender):
    name, workstream, required_inputs = "bad_length", "test", ["GR"]

    def fit(self, train_wells, target):
        pass

    def predict(self, well, target):
        return Prediction(np.zeros(len(well.depth) - 1))


class BadQuantiles(WrongLength):
    name = "bad_quantiles"

    def predict(self, well, target):
        n = len(well.depth)
        return Prediction(np.ones(n), p05=np.full(n, 2.0), p95=np.full(n, 3.0))


class NanSafePoint(BaseContender):
    name, workstream = "nan_point", "test"

    def fit(self, train_wells, target):
        pass

    def predict(self, well, target):
        return Prediction(np.full(len(well.depth), np.nan))


def test_template_passes_cli():
    r = subprocess.run([sys.executable, "-m", "benchmark.check_model", "template_linear"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1].startswith("OK template_linear")
    assert "LICENSE" not in r.stdout and "licence_name" not in r.stdout


def test_wrong_length_fails():
    with pytest.raises(ValueError, match="length"):
        check_contender(WrongLength)


def test_non_monotone_quantiles_fail():
    with pytest.raises(ValueError, match="p05 > p50"):
        check_contender(BadQuantiles)


def test_all_nan_point_model_is_valid_and_warns_about_inputs():
    summary, warnings = check_contender(NanSafePoint)
    assert summary.startswith("OK") and any("required_inputs" in w for w in warnings)
    assert any("LICENSE" in w for w in warnings)
    assert any("licence_name" in w for w in warnings)


def test_unknown_model_exits_nonzero():
    r = subprocess.run([sys.executable, "-m", "benchmark.check_model", "no_such_model"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 1 and "unknown model" in r.stderr


class TaskOnly(BaseContender):
    name, workstream, required_inputs = "task_only", "test", ["GR"]

    def fit(self, train_wells, target):
        pass

    def predict_task(self, task):
        return Prediction(np.full(len(task.depth), 2.4), np.full(len(task.depth), 0.1))


def test_task_only_model_instantiates_and_passes_check():
    TaskOnly()
    summary, _ = check_contender(TaskOnly)
    assert summary.startswith("OK")
    with pytest.raises(NotImplementedError, match="predict_task"):
        BaseContender.predict(TaskOnly(), None, "RHOB")


def test_missing_model_yaml_warns_and_template_yaml_is_clean():
    _, warnings = check_contender(TaskOnly)
    assert any("no model.yaml" in w and "was used instead" in w for w in warnings)
    from workstreams._template_model.contender import TemplateLinear

    summary, warnings = check_contender(TemplateLinear)
    assert summary.startswith("OK template_linear") and warnings == []


def test_invalid_model_yaml_fails(tmp_path, monkeypatch):
    (tmp_path / "model.yaml").write_text("name: task_only\nlicence: MIT\ntargets: {RHOB: {required: [dtc]}}\nextra: 1\n")
    monkeypatch.setattr("wmb.model_spec.yaml_candidates", lambda model: [tmp_path / "model.yaml"])
    with pytest.raises(ValueError, match="model.yaml is invalid.*unknown key 'extra'"):
        check_contender(TaskOnly)


def test_yaml_targets_the_class_does_not_support_warn(tmp_path, monkeypatch):
    (tmp_path / "model.yaml").write_text("name: task_only\nlicence: MIT\ntargets: {RHOB: {required: [GR]}}\n")
    monkeypatch.setattr("wmb.model_spec.yaml_candidates", lambda model: [tmp_path / "model.yaml"])

    class Narrow(TaskOnly):
        targets = ["DTC"]

    _, warnings = check_contender(Narrow)
    assert any("does not support: RHOB" in w for w in warnings)
    assert any("no entry for it" in w for w in warnings)

