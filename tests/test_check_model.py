"""check_model: the template passes, a deliberately bad model fails. Synthetic only."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark.check_model import check_contender, licence_warnings  # noqa: E402
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



TOY_SRC = """
from wmb.contender import BaseContender, Prediction
import numpy as np

class Toy(BaseContender):
    name = "toy_licence"
    workstream = "team_x"
    required_inputs = ["GR"]
    licence_name = "MIT"

    def fit(self, train_wells, target):
        pass

    def predict(self, well, target):
        return Prediction(np.full(len(well.depth), 2.4), np.full(len(well.depth), 0.1))
"""


def _load_toy(tmp_path, licence="model", filename="LICENSE.md"):
    """A wrapper under workstreams/<team>/models/<name>/, optionally with a licence file."""
    import importlib.util

    team = tmp_path / "workstreams" / "team_x"
    model_dir = team / "models" / "toy"
    model_dir.mkdir(parents=True)
    (model_dir / "contender.py").write_text(TOY_SRC, encoding="utf-8")
    if licence == "model":
        (model_dir / filename).write_text("MIT\n", encoding="utf-8")
    elif licence == "team":
        (team / filename).write_text("MIT\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("toy_licence_mod", model_dir / "contender.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Toy


def test_licence_in_model_folder_is_silent(tmp_path):
    Toy = _load_toy(tmp_path, "model")
    assert licence_warnings(Toy()) == []
    summary, warnings = check_contender(Toy)
    assert summary.startswith("OK")
    assert not any("LICENSE" in w for w in warnings)


def test_licence_txt_in_model_folder_is_accepted(tmp_path):
    Toy = _load_toy(tmp_path, "model", "LICENSE.txt")
    assert licence_warnings(Toy()) == []


def test_licence_only_in_team_folder_warns_but_passes(tmp_path):
    Toy = _load_toy(tmp_path, "team")
    notes = licence_warnings(Toy())
    assert notes and "not the model folder" in notes[0] and "team_x" in notes[0].replace("\\", "/")
    summary, warnings = check_contender(Toy)
    assert summary.startswith("OK")
    assert any("not the model folder" in w for w in warnings)


def test_missing_licence_file_warns_but_passes(tmp_path):
    Toy = _load_toy(tmp_path, "none")
    summary, warnings = check_contender(Toy)
    assert summary.startswith("OK")
    joined = " ".join(warnings).replace("\\", "/")
    assert "LICENSE.md" in joined and "models/toy" in joined
