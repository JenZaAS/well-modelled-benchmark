"""Benchmark wrapper for IBM Granite TinyTimeMixer (TTM R2)."""
import hashlib
import os
from pathlib import Path

from ...cache import (
    has_finetuned_ttm,
    has_probabilistic_ttm,
    red_cache_dir,
    red_hf_home,
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


class FrozenR2DTS(TTM):
    """The fixed IBM revision used as the DTS pilot's point-forecast control."""

    name = "red_ttm_r2_frozen"
    targets = ["DTS"]
    revision = "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"

    def load_backend(self):
        from huggingface_hub import snapshot_download

        folder = snapshot_download(backend.MODEL_ID, revision=self.revision,
                                   cache_dir=str(red_hf_home() / "hub"),
                                   allow_patterns=["config.json", "model.safetensors"])
        return backend.load(self.context, self.horizon, Path(folder))


class CombinedPilotDTS(TTM):
    """Combined three-well pilot; never substitute frozen or single-well weights."""

    name = "red_ttm_r2_dts_pilot_combined"
    targets = ["DTS"]
    hashes = {
        "model.safetensors": "03cef128e7d1c7d5a8530af6772447fb44e34cf11c9ee6e3c5cd5c07202e3225",
        "config.json": "ab7a271e679239e405fd6180ac9ceee68327b48c7477bf6a85c48ba0a0c8d1ce",
    }

    def load_backend(self):
        default = red_cache_dir() / "ttm-pilots" / "20261120-102336" / "combined"
        folder = Path(os.environ.get("WMB_TTM_DTS_PILOT", default)).expanduser()
        for filename, digest in self.hashes.items():
            path = folder / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; set WMB_TTM_DTS_PILOT to the combined checkpoint directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Checkpoint mismatch: {path} is not the scored DTS pilot artifact")
        return backend.load(self.context, self.horizon, folder)
