"""Red team contenders. Enable optional foundation models by name before discovery."""
from __future__ import annotations

import os

import numpy as np

from wmb.contender import BaseContender, Prediction
from .models.chronos2.contender import Chronos2
from .models.timestral.contender import Timestral
from .models.timesfm25.contender import TimesFM25


class TrainMeanFloor(BaseContender):
    name, workstream, inputs = "red_floor_train_mean", "red_foundation_models", []

    def fit(self, train_wells, target):
        y = np.concatenate([w.curves[target] for w in train_wells if target in w.curves])
        self.mu, self.sd = float(np.nanmean(y)), float(np.nanstd(y))

    def predict(self, well, target):
        n = len(well.depth)
        return Prediction(np.full(n, self.mu), np.full(n, self.sd))


MODELS = {"chronos2": Chronos2, "timestral": Timestral, "timesfm25": TimesFM25}
# The shared runner executes *every* registered model by default, even on core-only laptops.
CONTENDERS = [TrainMeanFloor, *(MODELS[key.strip()] for key in
              os.getenv("WMB_RED_MODELS", "").split(",") if key.strip())]
