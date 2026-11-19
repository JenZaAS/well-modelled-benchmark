"""Station wrapper for TimesFM 2.5 with numerical XReg covariates."""
from ...depth_model import DepthForecaster
from . import backend


class TimesFM25(DepthForecaster):
    name = "red_timesfm25"
    licence_name = "Apache-2.0"
    licence_url = "https://huggingface.co/google/timesfm-2.5-200m-pytorch"
    licence_restrictions = "Retain the licence and applicable notices; mark modified files."

    def load_backend(self):
        return backend.load(self.context, self.horizon)
