"""Red team contenders. Enable optional foundation models by name before discovery."""
from __future__ import annotations

import logging
import os

import numpy as np

from wmb.contender import BaseContender, Prediction
from .models.chronos2.contender import Chronos2
from .models.moment.contender import Moment
from .models.timestral.contender import Timestral
from .models.timesfm25.contender import TimesFM25
from .models.ttm.contender import TTM


class TrainMeanFloor(BaseContender):
    name, workstream, inputs = "red_floor_train_mean", "red_foundation_models", []

    def fit(self, train_wells, target):
        y = np.concatenate([w.curves[target] for w in train_wells if target in w.curves])
        self.mu, self.sd = float(np.nanmean(y)), float(np.nanstd(y))

    def predict(self, well, target):
        n = len(well.depth)
        return Prediction(np.full(n, self.mu), np.full(n, self.sd))


MODELS = {"chronos2": Chronos2, "timestral": Timestral, "timesfm25": TimesFM25, "ttm": TTM, "moment": Moment}


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
    return keys


# The shared runner executes *every* registered model by default, even on core-only laptops.
CONTENDERS = [TrainMeanFloor, *(MODELS[key] for key in _enabled_keys())]
