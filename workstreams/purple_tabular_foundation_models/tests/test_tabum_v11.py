"""Fair row selection, model registration and real-quantile checks."""

import os

import numpy as np
import pytest

from benchmark.registry import discover
from wmb.contract import TARGET_CURVES
from wmb.io import Well
from workstreams.purple_tabular_foundation_models.contender import KumoTabularContender, TabUMContender


@pytest.mark.parametrize("target", TARGET_CURVES)
def test_matches_kumo_context(target: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert discover()["purple_tabum_v11"] is TabUMContender
    depth = np.arange(20.0)
    wells = [Well(name, depth, {target: depth + k, "GR": depth * 2}) for k, name in enumerate("ba")]
    seen = []
    for model in (KumoTabularContender(context_rows=8), TabUMContender(context_rows=8)):
        monkeypatch.setattr(model, "_fit_backend", lambda x, y: seen.append((x, y)))
        model.fit(wells, target)
    for a, b in zip(*seen):
        np.testing.assert_array_equal(a, b)
    assert seen[1][0].shape == (8, 2)  # depth and GR only


@pytest.mark.skipif(os.getenv("WMB_TEST_TABUM") != "1", reason="requires pretrained tabUM")
def test_native_quantiles_with_missing_logs() -> None:
    depth = np.arange(32.0)
    train = Well("train", depth, {"RHOB": 2 + depth / 50, "GR": np.sin(depth)})
    model = TabUMContender(context_rows=32, batch_rows=4)
    model.fit([train], "RHOB")
    query = Well("held-out", depth[:7] + 0.5)
    pred = model.predict(query, "RHOB")
    assert pred.p50.shape == (7,) and np.isfinite(pred.p50).all()
    assert np.all(pred.p05 <= pred.p50) and np.all(pred.p50 <= pred.p95)
    x = np.column_stack((query.depth[:4], np.full(4, np.nan))).astype(np.float32)
    for actual, q in [(pred.p05, 0.05), (pred.p50, 0.50), (pred.p95, 0.95)]:
        np.testing.assert_allclose(actual[:4], model._reg.predict_quantile(x, q), atol=1e-5)
