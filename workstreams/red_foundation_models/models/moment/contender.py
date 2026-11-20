"""Benchmark wrapper for AutonLab MOMENT-1-small."""
from ...tsfm_depth_model import DepthForecaster
from . import backend


class Moment(DepthForecaster):
    name = "red_moment"
    licence_name = "MIT"
    licence_url = "https://huggingface.co/AutonLab/MOMENT-1-small"
    licence_restrictions = "Retain the copyright and permission notice."
    context = backend.CONTEXT
    horizon = backend.HORIZON

    def load_backend(self):
        return backend.load(self.context, self.horizon)
