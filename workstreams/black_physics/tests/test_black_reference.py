"""Gap-only straight-line interpolation reference used in the validation note. Synthetic only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from wmb.contract import Task, validate_prediction  # noqa: E402
from workstreams.black_physics.validate_force import linear_gap_reference  # noqa: E402


def gap_task(depth, y, mask):
    observed = np.where(mask, np.nan, y)
    return Task("W", depth, pd.DataFrame({"DEPTH": depth}), "RHOB", mask, "gap", 10.0, observed)


def test_interpolates_a_straight_line_exactly():
    depth = np.arange(100.0) * 0.5
    y = 2.0 + 0.01 * depth
    mask = (np.arange(100) >= 40) & (np.arange(100) < 60)
    p = linear_gap_reference(gap_task(depth, y, mask))
    validate_prediction(gap_task(depth, y, mask), p)
    np.testing.assert_allclose(p.p50[mask], y[mask])
    assert np.isnan(p.p50[~mask]).all() and p.std is None


def test_uses_nearest_measured_flank_and_abstains_without_one():
    depth = np.arange(100.0) * 0.5
    y = np.full(100, 2.4)
    y[35:40] = np.nan                                            # unmeasured just above the gap
    mask = (np.arange(100) >= 40) & (np.arange(100) < 60)
    assert np.allclose(linear_gap_reference(gap_task(depth, y, mask)).p50[mask], 2.4)
    y_open = y.copy()
    y_open[60:] = np.nan                                         # no measured value below
    assert np.isnan(linear_gap_reference(gap_task(depth, y_open, mask)).p50).all()


def test_does_not_bridge_a_depth_hole():
    depth = np.r_[np.arange(50.0) * 0.5, 100 + np.arange(50.0) * 0.5]   # 75 m jump after row 49
    y = np.full(100, 2.4)
    mask = (np.arange(100) >= 40) & (np.arange(100) < 50)       # gap ends at the hole: no lower flank
    assert np.isnan(linear_gap_reference(gap_task(depth, y, mask)).p50).all()


def test_missing_curve_tasks_get_no_reference():
    depth = np.arange(10.0)
    t = Task("W", depth, pd.DataFrame({"DEPTH": depth}), "RHOB", np.ones(10, bool), "missing_curve")
    assert np.isnan(linear_gap_reference(t).p50).all()
