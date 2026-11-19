import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

from . import backend


@pytest.fixture
def adapter():
    timesfm, hub = Mock(), Mock()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained.return_value
    with patch.dict(sys.modules, timesfm=timesfm, huggingface_hub=hub):
        forecast = backend.load(256, 32)
    hub.snapshot_download.assert_called_once_with(
        backend.MODEL_ID, revision=backend.REVISION, local_files_only=True,
    )
    return model, forecast


def test_pairs_named_covariates_and_selects_target_deciles(adapter):
    model, forecast = adapter
    deciles = np.arange(20).reshape(2, 10)
    model.forecast_with_covariates.return_value = (None, [deciles])
    got = forecast([1, 4], {"GR": [5, 6], "DTC": [7, 8]},
                   {"DTC": [9, 8], "GR": [2, 3]}, 2)
    args = model.forecast_with_covariates.call_args.kwargs
    np.testing.assert_array_equal(args["inputs"], [[1, 4]])
    np.testing.assert_array_equal(args["dynamic_numerical_covariates"]["GR"], [[5, 6, 2, 3]])
    np.testing.assert_array_equal(got, [[1, 11], [5, 15], [9, 19]])


def test_no_covariates_excludes_backcast(adapter):
    model, forecast = adapter
    model.forecast.return_value = (None, [np.arange(30).reshape(3, 10)])
    np.testing.assert_array_equal(forecast([1], {}, {}, 1), [[21], [25], [29]])


@pytest.mark.parametrize("past,future", [({"GR": [1]}, {}), ({"GR": [1]}, {"GR": [2, 3]})])
def test_rejects_misaligned_covariates(adapter, past, future):
    _, forecast = adapter
    with pytest.raises(ValueError):
        forecast([1], past, future, 1)


@pytest.mark.skipif(os.getenv("WMB_TEST_TIMESFM25") != "1", reason="requires cached TimesFM 2.5")
def test_known_log_spike_changes_only_its_forecast_depth():
    forecast = backend.load(256, 32)
    gr = np.tile([0., 1., 1., 0.], 64)
    target = 2.3 + 0.2 * gr
    flat = np.zeros(13)
    spike = flat.copy()
    spike[6] = 1.
    baseline = forecast(target, {"GR": gr}, {"GR": flat}, len(flat))
    changed = forecast(target, {"GR": gr}, {"GR": spike}, len(spike))
    assert changed.shape == (3, 13) and np.isfinite(changed).all()
    assert np.all(np.diff(changed, axis=0) >= 0)
    delta = changed[1] - baseline[1]
    np.testing.assert_allclose(delta[spike == 0], 0, atol=1e-5)
    assert delta[6] > 0.15
