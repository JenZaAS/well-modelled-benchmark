"""Red team contenders. Enable optional foundation models by name before discovery."""
from __future__ import annotations

import os

import numpy as np

from wmb.contender import BaseContender, Prediction
from .depth_model import DepthForecaster


class TrainMeanFloor(BaseContender):
    name, workstream, inputs = "red_floor_train_mean", "red_foundation_models", []

    def fit(self, train_wells, target):
        y = np.concatenate([w.curves[target] for w in train_wells if target in w.curves])
        self.mu, self.sd = float(np.nanmean(y)), float(np.nanstd(y))

    def predict(self, well, target):
        n = len(well.depth)
        return Prediction(np.full(n, self.mu), np.full(n, self.sd))


class Chronos2(DepthForecaster):
    name = "red_chronos2"

    @staticmethod
    def load_backend():
        from .chronos_backend import load
        return load()


class Timestral(DepthForecaster):
    name = "red_timestral"

    @staticmethod
    def load_backend():
        from .timestral_backend import load
        return load()


MODELS = {"chronos2": Chronos2, "timestral": Timestral}
# The shared runner executes *every* registered model by default, even on core-only laptops.
CONTENDERS = [TrainMeanFloor, *(MODELS[key.strip()] for key in
              os.getenv("WMB_RED_MODELS", "").split(",") if key.strip())]
