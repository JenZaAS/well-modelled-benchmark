"""The blue models are registered, in their own folders, and meet the benchmark contract."""
from pathlib import Path

import pytest

from benchmark.check_model import check_contender, licence_warnings
from benchmark.registry import discover
from wmb.model_spec import load_model_spec

BLUE = ["blue_gbt", "blue_ml_gbt", "blue_ml_gbt_trend", "blue_ml_linear", "blue_ml_linear_trend", "blue_ml_rf"]
MODELS = Path(__file__).resolve().parents[1] / "models"


def test_all_blue_models_are_registered_once():
    found = discover()
    assert set(BLUE) <= set(found)
    assert all(found[n]().workstream == "blue_boosted_trees" for n in BLUE)


@pytest.mark.parametrize("name", BLUE)
def test_model_meets_the_contract_without_warnings(name):
    factory = discover()[name]
    summary, warnings = check_contender(factory)
    assert summary.startswith(f"OK {name}") and warnings == []      # model.yaml, LICENSE.md and licence fields all present
    assert load_model_spec(factory).source == "yaml"
    assert licence_warnings(factory()) == []


def test_registered_models_have_no_stratigraphy_input():
    """Benchmark wells carry no formation labels, so a registered variant must run without them."""
    for name in BLUE[1:]:
        model = discover()[name]()
        assert not model.use_strat and not model.context


def test_every_model_folder_has_a_licence_and_a_readme():
    folders = [p for p in MODELS.iterdir() if p.is_dir() and not p.name.startswith("__")]
    assert {p.name for p in folders} == {"xgboost_quantile", "xgboost_gbt", "linear_regression", "random_forest"}
    for p in folders:
        assert (p / "LICENSE.md").is_file() and (p / "README.md").is_file() and list(p.glob("*.model.yaml"))
