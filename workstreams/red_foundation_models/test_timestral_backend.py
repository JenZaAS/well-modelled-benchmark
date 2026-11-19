import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from workstreams.red_foundation_models import timestral_backend


class TimestralBackendTest(unittest.TestCase):
    def test_pairs_covariates_by_name_and_returns_target_quantiles(self):
        package = Mock()
        loader = package.TimestralForecaster.from_pretrained
        infer = loader.return_value.forecast
        infer.return_value.quantile.side_effect = lambda q: {(0, 0): Mock(numpy=lambda: np.array([q]))}
        with patch.dict(sys.modules, timestral=package):
            forecast = timestral_backend.load()
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        q = forecast(np.array([1, 2]), {"b": [7, 8], "a": [3, 4]}, {"a": [5], "b": [9]}, 1)
        np.testing.assert_array_equal(infer.call_args.kwargs["known_covariates"], [[[3, 4, 5], [7, 8, 9]]])
        np.testing.assert_array_equal(q, [[0.1], [0.5], [0.9]])

    def test_refuses_unpaired_or_misaligned_covariates(self):
        with patch.dict(sys.modules, timestral=Mock()):
            forecast = timestral_backend.load()
        for future in ({}, {"a": [1, 2]}, {"a": [[1]]}):
            with self.subTest(future=future), self.assertRaises(ValueError):
                forecast(np.array([1]), {"a": [0]}, future, 1)
