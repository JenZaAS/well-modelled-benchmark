"""Offline adapter tests and an opt-in real checkpoint check."""

import os

import numpy as np
import pytest

from benchmark.registry import discover
from wmb.contract import TARGET_CURVES
from wmb.io import Well
from workstreams.purple_tabular_foundation_models.contender import KumoTabularContender


class RecordingKumo(KumoTabularContender):
    def _fit_backend(self, x: np.ndarray, y: np.ndarray) -> None:
        self.context, self.labels = x, y
        self.batches: list[np.ndarray] = []

    def _predict_batch(self, x: np.ndarray) -> np.ndarray:
        self.batches.append(x)
        return x[:, :1] + np.array([-1, 0, 1])


@pytest.mark.parametrize("target", TARGET_CURVES)
def test_context_and_query(target: str) -> None:
    depth = np.arange(10.0)
    a = Well("a", depth, {target: depth + 20, "GR": depth + 100})
    b = Well("b", depth[:3], {target: np.array([40, np.nan, 42])})
    model = RecordingKumo(context_rows=6, batch_rows=4)
    model.fit([b, a], target)
    np.testing.assert_array_equal(model.labels.ravel(), [20, 23, 26, 29, 40, 42])
    assert model.context.shape == (6, 2)
    query = Well("c", depth, {target: -depth, "target_observed": depth, "LITH": depth})
    pred = model.predict(query, target)
    assert [len(x) for x in model.batches] == [4, 4, 2]
    x = np.concatenate(model.batches)
    assert x.shape == (10, 2) and np.isnan(x[:, 1]).all()
    for actual, expected in [(pred.p05, depth - 1), (pred.p50, depth), (pred.p95, depth + 1)]:
        np.testing.assert_array_equal(actual, expected)


def test_registry_and_invalid_state() -> None:
    model = discover()["purple_kumo_tabular"]()
    assert isinstance(model, KumoTabularContender)
    well = Well("empty", np.arange(2.0), {"RHOB": np.full(2, np.nan)})
    with pytest.raises(ValueError, match="No finite training labels"):
        model.fit([well], "RHOB")
    with pytest.raises(ValueError, match="before predicting"):
        model.predict(well, "RHOB")


@pytest.mark.skipif(os.getenv("WMB_TEST_KUMO") != "1", reason="requires pretrained weights")
def test_pretrained_regression() -> None:
    depth = np.arange(32.0)
    well = Well("train", depth, {"RHOB": 2 + depth / 50, "GR": np.sin(depth)})
    model = KumoTabularContender(context_rows=32, batch_rows=4)
    model.fit([well], "RHOB")
    pred = model.predict(Well("query", depth[:7] + 0.5), "RHOB")
    assert pred.p50.shape == (7,) and np.isfinite(pred.p50).all()
    assert np.all(pred.p05 <= pred.p50) and np.all(pred.p50 <= pred.p95)
