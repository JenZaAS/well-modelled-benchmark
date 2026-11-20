"""Red team contenders. Enable optional foundation models by name before discovery."""
from __future__ import annotations

import logging
import os

import numpy as np

from wmb.contender import BaseContender, Prediction
from .models.chronos2.contender import Chronos2
from .models.granite.contender import FT as GRANITE_FT
from .models.granite.contender import GRANITE
from .models.moment.contender import Moment
from .models.timestral.contender import Timestral
from .models.timestral_target_only.contender import TimestralTargetOnly
from .models.timesfm25.contender import TimesFM25
from .models.ttm.contender import CombinedPilotDTS, FrozenR2DTS, TTM


class TrainMeanFloor(BaseContender):
    name, workstream, inputs = "red_floor_train_mean", "red_foundation_models", []

    def fit(self, train_wells, target):
        y = np.concatenate([w.curves[target] for w in train_wells if target in w.curves])
        self.mu, self.sd = float(np.nanmean(y)), float(np.nanstd(y))

    def predict(self, well, target):
        n = len(well.depth)
        return Prediction(np.full(n, self.mu), np.full(n, self.sd))


MODELS = {"chronos2": Chronos2, "timestral": Timestral, "timesfm25": TimesFM25, "ttm": TTM, "moment": Moment,
          "ttm_r2_dts_pilot": (CombinedPilotDTS, FrozenR2DTS),
          # Granite 50 m density-gap rows (Granite environment; artifacts from workstreams/black_physics/granite)
          "granite": (*GRANITE, *GRANITE_FT),
          "timestral_target_only": TimestralTargetOnly}
# Flags from before these rows moved here from black_physics; each still enables its rows.
LEGACY_FLAGS = {"WMB_BLACK_GRANITE": "granite", "WMB_BLACK_TIMESTRAL": "timestral_target_only"}


def _enabled_keys() -> list[str]:
    """Keys from WMB_RED_MODELS and the legacy WMB_WHITE_MODELS (ttm, moment), without repeats."""
    keys: list[str] = []
    for var in ("WMB_RED_MODELS", "WMB_WHITE_MODELS"):
        for key in os.getenv(var, "").split(","):
            key = key.strip()
            if key and key not in MODELS:
                logging.getLogger(__name__).warning("Ignoring unknown model key %r in %s", key, var)
            elif key and key not in keys:
                keys.append(key)
    for var, key in LEGACY_FLAGS.items():
        if os.getenv(var, "").strip() not in ("", "0") and key not in keys:
            keys.append(key)
    return keys


def _classes(key: str) -> tuple:
    entry = MODELS[key]
    return entry if isinstance(entry, tuple) else (entry,)


# The shared runner executes *every* registered model by default, even on core-only laptops.
_KEYS = _enabled_keys()
CONTENDERS = [TrainMeanFloor, *(cls for key in _KEYS for cls in _classes(key))]
# The rows left out above, with the flag that registers each; the viewer's Prediction tab lists them as opt-in.
OPT_IN = {cls: f"WMB_RED_MODELS={key}" for key in MODELS if key not in _KEYS for cls in _classes(key)}
