"""Contract tests: Prediction aliases, validate_prediction, Task.to_well. Synthetic only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from wmb.contender import Prediction  # noqa: E402
from wmb.contract import SCENARIO_GAP, SCENARIO_MISSING, Task, task_input_columns, validate_prediction  # noqa: E402


def make_task(scenario=SCENARIO_MISSING, n=100):
    depth = 1000 + 0.5 * np.arange(n)
    inputs = pd.DataFrame({"DEPTH": depth, "GR": np.linspace(30, 90, n), "DTC": np.full(n, np.nan)})
    mask = np.ones(n, bool)
    observed = None
    if scenario == SCENARIO_GAP:
        mask = np.zeros(n, bool)
        mask[20:40] = True
        observed = np.where(mask, np.nan, 2.4)
    return Task("W-1", depth, inputs, "RHOB", mask, scenario, 10.0 if observed is not None else None, observed)


def test_mean_p50_alias():
    m = np.arange(3.0)
    assert Prediction(m).p50 is m
    assert Prediction(None, p50=m).mean is m
    assert Prediction(m, np.ones(3)).std is not None


def test_valid_point_prediction_and_nan_allowed():
    t = make_task()
    p50 = np.full(100, 2.4)
    p50[:5] = np.nan
    out = validate_prediction(t, Prediction(mean=p50))
    assert out.p50 is out.mean and out.p05 is None


def test_gaussian_interval_filled_from_std():
    t = make_task()
    out = validate_prediction(t, Prediction(np.full(100, 2.0), np.full(100, 0.1)))
    assert np.allclose(out.p95 - out.p05, 2 * 1.6448536269514722 * 0.1)


def test_quantiles_fill_p05_p95():
    t = make_task()
    q = {0.05: np.full(100, 1.0), 0.5: np.full(100, 2.0), 0.95: np.full(100, 3.0)}
    out = validate_prediction(t, Prediction(quantiles=q))
    assert out.p50[0] == 2.0 and out.p05[0] == 1.0 and out.p95[0] == 3.0


def test_validation_errors_are_specific():
    t = make_task(n=120)
    with pytest.raises(ValueError, match="p50 length 100 != depth length 120"):
        validate_prediction(t, Prediction(np.zeros(100)))
    with pytest.raises(ValueError, match="p50 \\(or mean\\) is required"):
        validate_prediction(t, Prediction())
    with pytest.raises(ValueError, match="p05 > p50 at 12 samples"):
        p05 = np.zeros(120)
        p05[:12] = 5.0
        validate_prediction(t, Prediction(np.ones(120), p05=p05, p95=np.full(120, 2.0)))
    with pytest.raises(ValueError, match="std < 0 at 1 samples"):
        std = np.ones(120)
        std[3] = -1
        validate_prediction(t, Prediction(np.ones(120), std))
    with pytest.raises(ValueError, match="std length"):
        validate_prediction(t, Prediction(np.ones(120), np.ones(5)))


def test_to_well_missing_curve_omits_target():
    w = make_task().to_well()
    assert "RHOB" not in w.curves and "GR" in w.curves and "DEPTH" not in w.curves


def test_to_well_gap_includes_masked_target():
    t = make_task(SCENARIO_GAP)
    w = t.to_well()
    assert np.isnan(w.curves["RHOB"][t.predict_mask]).all() and np.isfinite(w.curves["RHOB"][~t.predict_mask]).all()
    assert t.scenario_key == "gap_10"


def test_task_input_columns_exclude_target():
    cols = task_input_columns("DTC")
    assert cols[0] == "DEPTH" and "DTC" not in cols and "RHOB" in cols


def test_validated_prediction_is_a_snapshot():
    t = make_task()
    p50, std = np.full(100, 2.0), np.full(100, 0.1)
    orig = Prediction(p50, std, quantiles={0.25: np.full(100, 1.9), 0.75: np.full(100, 2.1)})
    out = validate_prediction(t, orig)
    orig.p50[:] = 9.0
    std[:] = 9.0
    orig.quantiles[0.25][:] = 9.0
    assert (out.p50 == 2.0).all() and (out.std == 0.1).all() and (out.quantiles[0.25] == 1.9).all()


def test_non_monotone_quantiles_rejected():
    t = make_task()
    q = {0.25: np.full(100, 3.0), 0.75: np.full(100, 2.0), 0.5: np.full(100, 2.5)}
    with pytest.raises(ValueError, match="quantile 0.5 < quantile 0.25"):
        validate_prediction(t, Prediction(quantiles=q))
