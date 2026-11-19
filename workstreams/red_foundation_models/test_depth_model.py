import numpy as np
import pytest

from benchmark import registry  # exposes src/ as in the shared runner
from wmb.io import Well
from .contender import Chronos2
from .experiment import depth_interpolation, masked_case


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def forecast(h, past, future, horizon):
        calls.append((h.copy(), past, future))
        y = np.full(horizon, h[-1])
        return y - 1, y, y + 1

    monkeypatch.setattr(Chronos2, "load_backend", staticmethod(lambda: forecast))
    return calls


def test_gap_uses_only_visible_flanks_and_preserves_depth(recorder):
    z = np.arange(500.)
    w = Well("held", z, {"RHOB": z, "GR": 2 * z})
    masked, mask = masked_case(w, "RHOB", "gap", 32)
    m = Chronos2(shots=0)
    m.fit([], "RHOB")
    pred = m.predict(masked, "RHOB")
    assert mask.any() and np.isnan(masked.curves["RHOB"][mask]).all()
    np.testing.assert_allclose(pred.mean, z)
    assert np.all(pred.std[mask] > 0)
    assert all("RHOB" not in p and "RHOB" not in f for _, p, f in recorder)


def test_whole_curve_prompts_and_unseen_facies(recorder):
    z = np.arange(8.)
    train = Well("train", z, {"RHOB": z, "LITH": np.repeat([30000, 70000], 4)})
    test = Well("test", z, {"LITH": np.full(8, 99999)})
    model = Chronos2(facies=True)
    model.fit([train], "RHOB")
    out = model.predict(test, "RHOB")
    assert np.isfinite(out.mean).all()
    h, past, future = recorder[0]
    np.testing.assert_array_equal(h, z)
    np.testing.assert_array_equal(past["facies_30000"], [1] * 4 + [0] * 4)
    assert future["facies_unknown"].all() and not future["facies_30000"].any()
    assert "LITH" not in future


def test_no_forecast_across_acquisition_break(recorder):
    z = np.array([0., 1., 2., 90., 91., 92.])
    w = Well("held", z, {"RHOB": np.array([1., 2., 3., np.nan, np.nan, np.nan])})
    model = Chronos2(shots=0)
    model.fit([], "RHOB")
    assert np.isnan(model.predict(w, "RHOB").mean[3:]).all()
    assert not recorder


def test_interpolation_benchmark_uses_depth_and_respects_breaks():
    depth = np.array([0., 1., 3., 90., 91.])
    well = Well("masked", depth, {"RHOB": np.array([2., np.nan, 8., np.nan, np.nan])})
    np.testing.assert_allclose(depth_interpolation(well, "RHOB"), [2., 4., 8., np.nan, np.nan])
