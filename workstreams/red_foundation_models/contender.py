"""Red team: time-series foundation models (Chronos-2, MOMENT, TTM).

PLACEHOLDER so the benchmark runs end to end: predicts the training mean with the
training spread as sigma (the 'know nothing' floor every real model must beat).
TODO (red team): add Chronos2Contender / MomentContender / TTMContender here, same
interface. Weights load from the Hugging Face cache (HF_HOME under FH2026_CACHE),
never from the repo. Ideas and settings: docs/models.md and tools/laptop_setup/.
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction


class TrainMeanFloor(BaseContender):
    name, workstream, inputs = "red_floor_train_mean", "red_foundation_models", []

    def fit(self, train_wells, target):
        y = np.concatenate([w.curves[target] for w in train_wells if target in w.curves])
        self.mu, self.sd = float(np.nanmean(y)), float(np.nanstd(y))

    def predict(self, well, target):
        n = len(well.depth)
        return Prediction(np.full(n, self.mu), np.full(n, self.sd))


CONTENDERS = [TrainMeanFloor]
