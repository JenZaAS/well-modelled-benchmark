"""Benchmark wrapper for IBM Granite TinyTimeMixer (TTM R2)."""
from pathlib import Path

from ...cache import (
    has_finetuned_ttm,
    has_probabilistic_ttm,
    ttm_finetuned_dir,
    ttm_probabilistic_dir,
)
from ...tsfm_depth_model import DepthForecaster
from ...probabilistic import QUANTILE_LEVELS
from . import backend


class TTM(DepthForecaster):
    name = "red_ttm"
    licence_name = "Apache-2.0"
    licence_url = "https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2"
    licence_restrictions = "Retain the licence and applicable notices; mark modified files."
    context = backend.CONTEXT
    horizon = backend.HORIZON

    def load_backend(self):
        target = getattr(self, "target", "RHOB")
        if has_probabilistic_ttm(target):
            weights = ttm_probabilistic_dir(target)
            self.finetuned_from = Path(weights)
            self.band_levels = QUANTILE_LEVELS
        elif has_finetuned_ttm(target):
            weights = ttm_finetuned_dir(target)
            self.finetuned_from = Path(weights)
            self.band_levels = (0.1, 0.5, 0.9)
        else:
            weights = None
            self.finetuned_from = None
            self.band_levels = (0.1, 0.5, 0.9)
        return backend.load(self.context, self.horizon, weights)
