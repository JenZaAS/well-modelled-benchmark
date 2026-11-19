"""Black team: physics / empirical baselines. Fast, explainable floors to beat.

- Gardner (1974): RHOB from DTC, rho = a * Vp^b (Vp in m/s), a=0.31, b=0.25; a and b refitted on training wells.
- Castagna mudrock line (1985): Vs = 0.8621*Vp - 1.1724 (km/s), gives DTS from DTC.
Uncertainty: constant sigma = std of training residuals (simple, honest starting point).
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction


def _concat(wells, *curves):
    rows = [np.column_stack([w.curves[c] for c in curves]) for w in wells if all(c in w.curves for c in curves)]
    a = np.vstack(rows) if rows else np.empty((0, len(curves)))
    return a[np.isfinite(a).all(axis=1)]


class Gardner(BaseContender):
    name, workstream, inputs, targets = "black_gardner", "black_physics", ["DTC"], ["RHOB"]

    def fit(self, train_wells, target):
        d = _concat(train_wells, "DTC", "RHOB")
        d = d[(d[:, 0] > 40) & (d[:, 0] < 200)]
        lv, lr = np.log(304800 / d[:, 0]), np.log(d[:, 1])
        self.b, la = np.polyfit(lv, lr, 1) if len(d) > 10 else (0.25, np.log(0.31))
        self.a = float(np.exp(la))
        self.sigma = float(np.std(d[:, 1] - self.a * (304800 / d[:, 0]) ** self.b)) if len(d) else 0.1

    def predict(self, well, target):
        dtc = well.curves.get("DTC", np.full_like(well.depth, np.nan))
        mean = self.a * (304800 / dtc) ** self.b
        return Prediction(mean, np.full_like(mean, self.sigma))


class MudrockDTS(BaseContender):
    name, workstream, inputs, targets = "black_castagna_dts", "black_physics", ["DTC"], ["DTS"]

    def fit(self, train_wells, target):
        d = _concat(train_wells, "DTC", "DTS")
        self.sigma = float(np.std(d[:, 1] - self._dts(d[:, 0]))) if len(d) else 20.0

    @staticmethod
    def _dts(dtc):
        vs = 0.8621 * (304.8 / dtc) - 1.1724              # km/s
        return 304.8 / np.where(vs > 0.3, vs, np.nan)

    def predict(self, well, target):
        mean = self._dts(well.curves.get("DTC", np.full_like(well.depth, np.nan)))
        return Prediction(mean, np.full_like(mean, self.sigma))


CONTENDERS = [Gardner, MudrockDTS]
