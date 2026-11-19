"""Tests for the blue-team ML study. Run: python -m pytest workstreams/blue_boosted_trees/tests"""
import numpy as np
import pandas as pd
import pytest

from workstreams.blue_boosted_trees.ml_study import report
from workstreams.blue_boosted_trees.ml_study.features import FeatureBuilder
from workstreams.blue_boosted_trees.ml_study.metrics import evaluate, target_space
from workstreams.blue_boosted_trees.ml_study.models import MODEL_KINDS, MLContender
from wmb import synthetic


def _wells_with_strat(n_wells: int = 5):
    """Synthetic wells with two groups split at mid-depth and formations split in quarters."""
    wells = synthetic.make_wells(n_wells=n_wells, n_samples=400, seed=1)
    for w in wells:
        q = np.digitize(w.depth, np.quantile(w.depth, [0.25, 0.5, 0.75]))
        w.meta["group"] = np.where(q < 2, "SHALLOW GP.", "DEEP GP.")
        w.meta["formation"] = np.array([f"FM{i}" for i in q])
    return wells


def test_metrics_bias_sign_and_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert evaluate(y, y)["MAE"] == 0 and evaluate(y, y)["R2"] == 1
    assert evaluate(y, y + 0.5)["bias"] == pytest.approx(0.5)   # over-prediction is positive
    assert evaluate(y, np.full(4, np.nan))["n"] == 0


@pytest.mark.parametrize("encoding, extra_cols", [("ordinal", 2), ("onehot", 2 + 4)])
def test_strat_features(encoding, extra_cols):
    wells = _wells_with_strat()
    fb = FeatureBuilder(["GR", "RHOB"], use_strat=True, strat_encoding=encoding).fit(wells)
    X = fb.transform(wells[0])
    assert X.shape == (400, 3 + extra_cols) and len(fb.names) == X.shape[1]
    assert fb.order["group"] == ["SHALLOW GP.", "DEEP GP."]        # shallow -> deep by median depth


def test_unseen_strat_label_is_not_an_error():
    wells = _wells_with_strat()
    fb = FeatureBuilder(["GR"], use_strat=True, strat_encoding="ordinal").fit(wells)
    new = wells[0].without("GR")
    new.meta = {"group": np.full(400, "NEW GP."), "formation": np.full(400, "NEW FM")}
    assert np.isnan(fb.transform(new)[:, -2:]).all()


@pytest.mark.parametrize("kind", MODEL_KINDS)
@pytest.mark.parametrize("use_strat", [False, True])
def test_contender_fits_and_predicts_without_target(kind, use_strat):
    wells = _wells_with_strat()
    train, test = wells[:4], wells[4]
    model = MLContender(kind, use_strat, max_train_rows=500)
    model.fit(train, "RHOB")
    assert "RHOB" not in model.inputs
    pred = model.predict(test.without("RHOB"), "RHOB").mean
    assert pred.shape == test.depth.shape and np.isfinite(pred).all()


def test_resistivity_is_modelled_in_log10_and_returned_in_ohm_m():
    wells = _wells_with_strat()
    model = MLContender("linear", use_strat=False, max_train_rows=500)
    model.fit(wells[:4], "RDEP")
    pred = model.predict(wells[4].without("RDEP"), "RDEP").mean
    assert (pred > 0).all() and (pred < 1e4 + 1).all()
    assert target_space("RDEP", np.array([10.0, 100.0])).tolist() == [1.0, 2.0]
    assert target_space("RHOB", np.array([2.5])).tolist() == [2.5]


def test_missing_strat_meta_fails_fast():
    wells = synthetic.make_wells(n_wells=3, n_samples=100)
    with pytest.raises(ValueError, match="meta"):
        MLContender("linear", use_strat=True).fit(wells, "RHOB")


def _metrics_frame():
    rows = []
    for target, r2 in (("GR", 0.30), ("RHOB", 0.80)):
        for variant, bump in (("gbt", 0.0), ("gbt_strat", 0.05), ("linear", -0.2), ("linear_strat", -0.1),
                              ("rf", 0.01), ("rf_strat", 0.06)):
            rows.append({"target": target, "variant": variant, "R2": r2 + bump, "MAE": 1 - r2 - bump, "bias": 0.1 - bump})
    return pd.DataFrame(rows)


def test_report_tables():
    m = _metrics_frame()
    cmp = report.comparison_table(m, "R2")
    assert list(cmp.index) == ["GR", "RHOB"] and cmp.shape[1] == 6
    assert (report.best_variant_per_log(cmp, "R2") == "Random forest + stratigraphy").all()
    assert (report.best_variant_per_log(report.comparison_table(m, "MAE"), "MAE")
            == "Random forest + stratigraphy").all()
    effect = report.stratigraphy_effect(m, "R2")
    assert effect.loc["GR", "Gradient boosting (XGBoost)"] == pytest.approx(0.05)
    assert effect.loc["RHOB", "Multilinear regression"] == pytest.approx(0.1)
