"""Missing-dependency hints for the Prediction tab (no team workstream edits)."""
from __future__ import annotations

from wmb.deps import (
    friendly_module_error,
    import_groups,
    missing_from_source,
    missing_reason,
    needs_install_message,
)


def test_sklearn_message():
    assert needs_install_message("sklearn") == "needs scikit-learn — pip install scikit-learn"
    exc = ModuleNotFoundError("No module named 'sklearn'")
    exc.name = "sklearn"
    assert friendly_module_error(exc) == "needs scikit-learn — pip install scikit-learn"


def test_try_except_import_is_one_group(monkeypatch):
    src = """
try:
    import xgboost as xgb
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
"""
    required, groups = import_groups(src)
    assert "sklearn" not in required and "xgboost" not in required
    assert groups and groups[0] == {"xgboost", "sklearn"}
    # Group is missing only when every alternative is missing (do not assume CI package set).
    monkeypatch.setattr("wmb.deps.module_installed", lambda name: False)
    assert "sklearn" in missing_from_source(src)
    monkeypatch.setattr("wmb.deps.module_installed", lambda name: name == "xgboost")
    assert missing_from_source(src) == []


def test_blue_gbt_factory_reports_sklearn_if_missing():
    from workstreams.blue_boosted_trees.contender import GBT
    reason = missing_reason(GBT)
    # If the machine has sklearn or xgboost, this is None; otherwise the install hint.
    if reason is not None:
        assert "pip install" in reason
