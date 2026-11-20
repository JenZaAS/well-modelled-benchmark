"""Tests for the blue-team ML study. Run: python -m pytest workstreams/blue_boosted_trees/tests"""
import numpy as np
import pandas as pd
import pytest

from workstreams.blue_boosted_trees.ml_study import report
from workstreams.blue_boosted_trees.ml_study.features import FeatureBuilder
from workstreams.blue_boosted_trees.ml_study.metrics import evaluate, target_space
from workstreams.blue_boosted_trees.ml_study.models import MODEL_KINDS, MedianImputer, MLContender, variant_name
from workstreams.blue_boosted_trees.ml_study.run_study import merge_metrics
from wmb import synthetic
from wmb.io import Well


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
@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True)], ids=["base", "ctx", "trend"])
@pytest.mark.parametrize("use_strat", [False, True])
def test_contender_fits_and_predicts_without_target(kind, use_strat, flags):
    wells = _wells_with_strat()
    train, test = wells[:4], wells[4]
    model = MLContender(kind, use_strat, max_train_rows=500, context=flags[0], trend=flags[1])
    model.fit(train, "RHOB")
    assert "RHOB" not in model.inputs
    assert not any(n.startswith("RHOB") for n in model.features.names)   # no neighbouring-depth features of the target
    pred = model.predict(test.without("RHOB"), "RHOB").mean
    assert pred.shape == test.depth.shape and np.isfinite(pred).all()


def test_context_features_shift_and_average():
    depth = np.arange(0.0, 100.0, 1.0)                       # 1 m spacing: 5 m = 5 samples
    well = Well("w", depth, {"GR": depth.copy()})            # GR equals depth, so values are easy to check
    fb = FeatureBuilder(["GR"], context=True).fit([well])
    col = dict(zip(fb.names, fb.transform(well).T))
    assert col["GR_above5m"][10] == 5 and col["GR_below5m"][10] == 15   # above = shallower, below = deeper
    assert np.isnan(col["GR_above5m"][:5]).all() and np.isnan(col["GR_below5m"][-5:]).all()
    assert col["GR_avg5m"][50] == pytest.approx(50) and np.isfinite(col["GR_avg20m"]).all()
    assert fb.n_continuous == len(fb.names) == 1 + 4 + 1                # log + 4 context columns + depth


def test_trend_features_describe_the_change_with_depth():
    depth = np.arange(0.0, 100.0, 1.0)                       # 1 m spacing: the 10 m window is 10 samples
    up = Well("w", depth, {"GR": depth.copy()})              # GR increases with depth
    fb = FeatureBuilder(["GR"], trend=True).fit([up])
    col = dict(zip(fb.names, fb.transform(up).T))
    assert col["GR_above10m"][50] == pytest.approx(44.5)     # mean of the 10 samples above, sample itself excluded
    assert col["GR_below10m"][50] == pytest.approx(55.5)
    assert col["GR_trend10m"][50] == pytest.approx(11.0)     # below minus above: positive = increases with depth
    assert col["GR_std10m"][50] == pytest.approx(np.std(np.arange(10.0), ddof=1))
    assert np.isnan(col["GR_above10m"][0]) and np.isnan(col["GR_below10m"][-1])
    assert fb.n_continuous == len(fb.names) == 1 + 4 + 1
    flat = Well("w", depth, {"GR": np.full(100, 7.0)})
    assert np.allclose(dict(zip(fb.names, fb.transform(flat).T))["GR_trend10m"][20:80], 0)


def test_median_imputer_matches_scikit_learn():
    from sklearn.impute import SimpleImputer
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[:, 3] = rng.normal(size=200)                           # a column with no gaps gets no flag
    expected = SimpleImputer(strategy="median", add_indicator=True).fit_transform(X)
    np.testing.assert_allclose(MedianImputer().fit(X).transform(X), expected)
    empty = np.column_stack([X, np.full(200, np.nan)])       # a column with no data at all is dropped
    assert MedianImputer().fit(empty).transform(empty).shape == expected.shape


def test_variant_names_and_labels():
    assert variant_name("rf", True) == "rf_strat"                       # names of the earlier results still valid
    assert variant_name("gbt", True, True) == "gbt_ctx_strat" and variant_name("gbt", trend=True) == "gbt_trend"
    assert len(report.VARIANTS) == 18 and len(set(report.VARIANTS)) == 18
    assert report.variant_label("rf_ctx_strat") == "Random forest + context + stratigraphy"
    assert report.variant_label("gbt_trend_strat") == "Gradient boosting (XGBoost) + trend + stratigraphy"
    assert report.variant_label("gbt") == "Gradient boosting (XGBoost)"


def test_merge_metrics_replaces_only_rerun_variants():
    old = pd.DataFrame([{"target": "GR", "variant": "gbt", "R2": 0.1}, {"target": "GR", "variant": "rf", "R2": 0.2}])
    merged = merge_metrics(old, [{"target": "GR", "variant": "gbt", "R2": 0.9, "context": False, "trend": False},
                                 {"target": "GR", "variant": "gbt_trend", "R2": 0.5, "context": False, "trend": True}])
    assert sorted(merged["variant"]) == ["gbt", "gbt_trend", "rf"]
    by = merged.set_index("variant")
    assert by.loc["gbt", "R2"] == 0.9 and by.loc["gbt_trend", "trend"]
    assert not by.loc["rf", "context"] and not by.loc["rf", "trend"]    # older rows have no flags


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


def test_context_effect_table():
    m = pd.DataFrame([{"target": "GR", "variant": v, "R2": r2} for v, r2 in
                      (("gbt", 0.30), ("gbt_ctx", 0.45), ("gbt_strat", 0.35), ("gbt_ctx_strat", 0.50))])
    effect = report.context_effect(m, "R2")
    assert effect.loc["GR", "Gradient boosting (XGBoost)"] == pytest.approx(0.15)
    assert effect.loc["GR", "Gradient boosting (XGBoost) + stratigraphy"] == pytest.approx(0.15)
    assert report.available_variants(m) == ["gbt", "gbt_strat", "gbt_ctx", "gbt_ctx_strat"]
    trend = pd.DataFrame([{"target": "GR", "variant": v, "R2": r2} for v, r2 in
                          (("gbt", 0.30), ("gbt_trend", 0.40), ("gbt_strat", 0.35), ("gbt_trend_strat", 0.33))])
    assert report.trend_effect(trend, "R2").loc["GR", "Gradient boosting (XGBoost)"] == pytest.approx(0.10)
    assert report.trend_effect(trend, "R2").loc["GR", "Gradient boosting (XGBoost) + stratigraphy"] == pytest.approx(-0.02)
    assert report.stratigraphy_effect(trend, "R2").loc["GR", "Gradient boosting (XGBoost) + trend"] == pytest.approx(-0.07)
