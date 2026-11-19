"""model.yaml loader / validator / eligibility. Synthetic only."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from wmb.contender import BaseContender  # noqa: E402
from wmb.model_spec import eligible, inputs_used, load_model_spec, parse_spec  # noqa: E402

GOOD = {"name": "m", "licence": "MIT", "targets": {"RHOB": {"required": ["DTC"], "optional": ["GR", "NPHI"]}}}


def test_template_yaml_is_valid_and_matches_the_model():
    from workstreams._template_model.contender import TemplateLinear

    spec = load_model_spec(TemplateLinear)
    assert spec.source == "yaml" and not spec.errors and not spec.warnings
    assert spec.name == "template_linear" and spec.licence == "MIT"
    assert set(spec.targets) == {"RHOB", "DTC", "DTS"}
    assert spec.targets["RHOB"].required == ["DTC"] and spec.targets["DTS"].optional == ["RHOB", "GR", "NPHI"]


def test_fallback_to_class_metadata_without_yaml():
    class NoYaml(BaseContender):
        name, workstream, licence_name = "no_yaml", "test", "Apache-2.0"
        required_inputs, optional_inputs, targets = ["DEPTH", "DTC"], ["GR", "RHOB"], ["RHOB", "DTS"]

        def fit(self, train_wells, target):
            pass

    spec = load_model_spec(NoYaml())
    assert spec.source == "class" and spec.licence == "Apache-2.0" and spec.path is None
    assert spec.for_target("DTS").required == ["DTC"] and spec.for_target("DTS").optional == ["GR", "RHOB"]
    assert spec.for_target("RHOB").optional == ["GR"]          # the target is never its own input
    assert spec.for_target("DTC") is None


def test_class_without_targets_supports_any_target():
    class Any_(BaseContender):
        name, workstream, inputs = "any_t", "test", ["GR"]

        def fit(self, train_wells, target):
            pass

    spec = load_model_spec(Any_)
    assert spec.supports("PEF") and eligible(spec, "PEF", ["GR"]) == (True, "")


def test_eligible_reasons_and_inputs_used():
    spec = parse_spec(GOOD)
    assert not spec.errors
    assert eligible(spec, "RHOB", ["DTC"]) == (True, "")
    ok, reason = eligible(spec, "RHOB", ["GR"])
    assert not ok and "missing required curve" in reason and "DTC" in reason
    ok, reason = eligible(spec, "DTS", ["DTC"])
    assert not ok and "does not support target DTS" in reason
    assert inputs_used(spec, "RHOB", ["DTC", "NPHI", "PEF"]) == {"required": ["DTC"], "optional": ["NPHI"]}


@pytest.mark.parametrize("data, text", [
    ({**GOOD, "extra": 1}, "unknown key 'extra'"),
    ({**GOOD, "name": ""}, "name is required"),
    ({**GOOD, "targets": {}}, "non-empty"),
    ({**GOOD, "targets": {"RHOB": {"required": ["dtc"]}}}, "not a canonical curve name"),
    ({**GOOD, "targets": {"RHOB": {"required": ["DEPTH"]}}}, "DEPTH is always present"),
    ({**GOOD, "targets": {"RHOB": {"required": ["RHOB"]}}}, "own inputs"),
    ({**GOOD, "targets": {"RHOB": {"required": ["GR"], "optional": ["GR"]}}}, "both required and optional"),
    ({**GOOD, "targets": {"RHOB": {"requird": ["GR"]}}}, "unknown key 'requird'"),
    ({**GOOD, "targets": {"RHOB": {"required": "GR"}}}, "list of curve names"),
    ({**GOOD, "targets": {"rho b!": {}}}, "not a canonical curve name"),
    ([1, 2], "mapping"),
])
def test_invalid_specs_report_errors(data, text):
    spec = parse_spec(data)
    assert any(text in e for e in spec.errors), spec.errors


def test_non_core_curve_and_missing_licence_warn():
    spec = parse_spec({"name": "m", "targets": {"RHOB": {"required": ["MYCURVE"]}}})
    assert not spec.errors
    assert any("not a core curve" in w for w in spec.warnings) and any("licence" in w for w in spec.warnings)


def test_invalid_yaml_keeps_class_metadata_and_reports_errors(tmp_path, monkeypatch):
    class Local(BaseContender):
        name, workstream, required_inputs, targets = "local", "test", ["GR"], ["RHOB"]

        def fit(self, train_wells, target):
            pass

    (tmp_path / "model.yaml").write_text("name: local\ntargets: {}\nbogus: 1\n")
    monkeypatch.setattr("wmb.model_spec.yaml_candidates", lambda model: [tmp_path / "model.yaml"])
    spec = load_model_spec(Local)
    assert spec.source == "class" and spec.errors and spec.for_target("RHOB").required == ["GR"]


def test_shared_model_yaml_for_another_model_is_ignored(tmp_path, monkeypatch):
    class Other(BaseContender):
        name, workstream, targets = "other", "test", ["DTC"]

        def fit(self, train_wells, target):
            pass

    (tmp_path / "model.yaml").write_text("name: someone_else\nlicence: MIT\ntargets: {RHOB: {}}\n")
    monkeypatch.setattr("wmb.model_spec.yaml_candidates", lambda model: [tmp_path / "model.yaml"])
    spec = load_model_spec(Other)
    assert spec.source == "class" and not spec.errors and spec.supports("DTC")
