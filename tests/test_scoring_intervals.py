"""Interval metrics in wmb.scoring. Synthetic only."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wmb import scoring  # noqa: E402


def test_perfect_prediction():
    y = np.linspace(1, 3, 200)
    s = scoring.regression_scores(y, y, np.full(200, 0.1))
    assert s["MAE"] == 0 and s["RMSE"] == 0 and s["R2"] == 1 and s["cover_90"] == 1 and s["has_uncertainty"]


def test_gaussian_noise_is_calibrated():
    rng = np.random.default_rng(0)
    y = rng.normal(2.4, 0.2, 20000)
    mean = y + rng.normal(0, 0.1, y.size)
    s = scoring.regression_scores(y, mean, np.full(y.size, 0.1))
    assert abs(s["cover_90"] - 0.90) < 0.02 and abs(s["cover_50"] - 0.50) < 0.02
    assert abs(s["interval_width_90"] - 2 * 1.6448536269514722 * 0.1) < 1e-9


def test_quantile_interval_without_std_has_no_cover_50():
    y = np.zeros(100)
    s = scoring.regression_scores(y, y, p05=np.full(100, -1.0), p95=np.full(100, 1.0))
    assert s["cover_90"] == 1 and s["interval_width_90"] == 2 and "cover_50" not in s and "pinball_05" in s
    s = scoring.regression_scores(y, y, p05=np.full(100, -1.0), p95=np.full(100, 1.0), p25=np.full(100, 0.5), p75=np.full(100, 1.0))
    assert s["cover_50"] == 0


def test_point_only_has_no_fake_coverage():
    y = np.linspace(1, 2, 50)
    s = scoring.regression_scores(y, y + 0.1)
    assert s["has_uncertainty"] is False
    assert not {"cover_90", "cover_50", "interval_width_90", "CRPS", "cover_1sigma"} & set(s)


def test_accuracy_in_units_of_y():
    y = np.array([100.0, 110.0, 120.0])
    s = scoring.regression_scores(y, y + np.array([2.0, -2.0, 2.0]))
    assert abs(s["MAE"] - 2.0) < 1e-12 and abs(s["RMSE"] - 2.0) < 1e-12


def test_nan_pairs_are_skipped():
    s = scoring.regression_scores(np.array([1.0, np.nan, 3.0]), np.array([1.0, 5.0, np.nan]), np.ones(3))
    assert s["n"] == 1


def test_mixed_interval_and_std_samples_both_contribute_to_cover():
    y = np.zeros(100)
    p05 = np.where(np.arange(100) < 50, -1.0, np.nan)
    p95 = np.where(np.arange(100) < 50, 1.0, np.nan)
    std = np.where(np.arange(100) < 50, np.nan, 1.0)
    s = scoring.regression_scores(y, y, std, p05=p05, p95=p95)
    assert s["cover_90"] == 1 and abs(s["interval_width_90"] - (2 + 2 * 1.6448536269514722) / 2) < 1e-9
    assert s["cover_50"] == 1
