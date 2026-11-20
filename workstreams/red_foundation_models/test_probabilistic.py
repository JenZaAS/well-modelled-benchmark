"""Unit tests for the quantile-head / conformal helpers (no granite-tsfm)."""
import numpy as np
import pytest

from workstreams.red_foundation_models.tsfm_depth_model import prediction_from_band
from workstreams.red_foundation_models.probabilistic import (
    QUANTILE_LEVELS,
    apply_conformal,
    conformal_width,
    pinball,
    read_meta,
    write_meta,
)
from workstreams.red_foundation_models.models.ttm.finetune_ttm import _split_wells, parse_args


def test_pinball_median_is_mae_half():
    y = np.array([0.0, 1.0, 2.0])
    q = np.array([0.0, 1.0, 3.0])
    # one residual of +0 and one of -1 at tau=0.5 -> 0.5 * 1 / 3
    assert pinball(y, q, 0.5) == pytest.approx(0.5 * 1.0 / 3.0)


def test_conformal_width_covers_held_out():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=400)
    q05 = np.full_like(y, -0.2)
    q95 = np.full_like(y, 0.2)
    width = conformal_width(y, q05, q95, alpha=0.10)
    lo, _, hi = apply_conformal(q05, np.zeros_like(y), q95, width)
    cover = np.mean((y >= lo) & (y <= hi))
    # Finite-sample CQR is slightly conservative; should be at least ~90%.
    assert width > 0
    assert cover >= 0.90


def test_apply_conformal_does_not_invert():
    q50 = np.array([1.0, 2.0])
    lo, mid, hi = apply_conformal(np.array([0.9, 1.9]), q50, np.array([1.1, 2.1]), width=-0.5)
    assert mid.tolist() == [1.0, 2.0]
    assert np.all(lo <= mid) and np.all(mid <= hi)


def test_prediction_from_band_sets_p05_p95():
    q = np.array([[1.0, 1.5], [2.0, 2.0], [3.0, 2.8]])
    pred = prediction_from_band(q, QUANTILE_LEVELS)
    assert pred.p05.tolist() == [1.0, 1.5]
    assert pred.p50.tolist() == [2.0, 2.0]
    assert pred.p95.tolist() == [3.0, 2.8]
    assert set(pred.quantiles) == {0.05, 0.50, 0.95}


def test_prediction_from_band_point_path_has_no_p05():
    q = np.array([[1.0], [2.0], [3.0]])
    pred = prediction_from_band(q, (0.1, 0.5, 0.9))
    assert pred.p05 is None and pred.p95 is None
    assert pred.std is not None
    # 80% half-width of 1 / 1.28155...
    assert pred.std[0] == pytest.approx(1.0 / 1.2815515655446004, rel=1e-6)
    assert 0.1 in pred.quantiles and 0.9 in pred.quantiles


def test_parse_args_probabilistic_defaults_conformal_on():
    args = parse_args(["--probabilistic"])
    assert args.probabilistic is True
    assert args.conformal is None  # resolved to True in main()
    args = parse_args(["--probabilistic", "--no-conformal"])
    assert args.conformal is False
    args = parse_args([])
    assert args.probabilistic is False


def test_split_wells_never_overlaps():
    class W:
        def __init__(self, name):
            self.name = name
    wells = [W(str(i)) for i in range(10)]
    train, calib = _split_wells(wells, 0.2, seed=0)
    assert len(calib) == 2 and len(train) == 8
    assert {w.name for w in train}.isdisjoint({w.name for w in calib})
    assert {w.name for w in train + calib} == {str(i) for i in range(10)}


def test_write_read_meta(tmp_path):
    write_meta(tmp_path, {"probabilistic": True, "conformal_width": 0.12})
    meta = read_meta(tmp_path)
    assert meta["probabilistic"] is True
    assert meta["conformal_width"] == pytest.approx(0.12)
    assert read_meta(tmp_path / "missing") == {}
