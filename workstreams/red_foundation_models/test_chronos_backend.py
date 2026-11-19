"""Set WMB_TEST_CHRONOS=1 with cached weights to include CPU integration."""
import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from workstreams.red_foundation_models.chronos_backend import load


class ChronosContractTest(unittest.TestCase):
    def test_refuses_unpaired_or_misaligned_covariates(self):
        packages = {name: Mock() for name in ("torch", "chronos", "huggingface_hub")}
        with patch.dict(sys.modules, packages):
            forecast = load()
        packages["huggingface_hub"].snapshot_download.assert_called_once_with(
            "amazon/chronos-2", local_files_only=True,
        )
        history = np.arange(4)
        for past, future in (({"x": history}, {}), ({"x": history[:2]}, {"x": history}),
                             ({"x": history}, {"x": history[:2]})):
            with self.subTest(past=past, future=future), self.assertRaises(ValueError):
                forecast(history, past, future, 4)


@unittest.skipUnless(os.environ.get("WMB_TEST_CHRONOS") == "1", "requires cached Chronos-2")
class ChronosBackendTest(unittest.TestCase):
    def test_known_future_changes_forecast(self):
        forecast = load()
        history = np.sin(np.arange(128) * 0.1)
        past = {"signal": history.copy()}
        high, low = (forecast(history, past, {"signal": np.full(8, x)}, 8) for x in (1, -1))
        for result in (high, low):
            self.assertEqual(len(result), 3)
            for q in result:
                self.assertEqual(q.shape, (8,))
                self.assertTrue(np.isfinite(q).all())
            self.assertTrue((result[0] <= result[1]).all() and (result[1] <= result[2]).all())
        self.assertFalse(np.allclose(high[1], low[1]))
        for q in forecast(history, {}, {}, 1):
            self.assertEqual(q.shape, (1,))
