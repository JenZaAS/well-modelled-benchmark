"""Thin station wrapper shared by every black physics model.

A subclass supplies three hooks:
    default_params()            parameters before any fit (fixed equations: the published constants)
    fit_params(train_wells)     parameters from training wells (fixed equations: default_params())
    mean_from(params, well)     point prediction on the full depth index, NaN where the model abstains

`fit` estimates the uncertainty scale (out-of-fold when `uses_fitted_parameters`, else training-well
residuals) and then fits the point model on all training wells. `predict` reads only the declared input
curves by canonical name and returns a Prediction whose std has the same support as the mean.
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction
from workstreams.black_physics.models._shared import calibration as cal
from workstreams.black_physics.models._shared.units import finite_or_nan

MIT_URL = "https://opensource.org/license/mit"
RESTRICTIONS = "none for this code (original, MIT); the equations are published methods credited in the model card"


class PhysicsContender(BaseContender):
    workstream = "black_physics"
    optional_inputs: list[str] = []
    licence_name = "MIT"
    licence_url = MIT_URL
    licence_restrictions = RESTRICTIONS
    card = ""                                  # model card, relative to workstreams/black_physics/

    def __init__(self) -> None:
        self.params = self.default_params()
        self.calibration = cal.Calibration.unfitted()

    # hooks ---------------------------------------------------------------------------------------
    @property
    def uses_fitted_parameters(self) -> bool:
        return False

    def default_params(self):
        return None

    def fit_params(self, train_wells):
        return self.default_params()

    def mean_from(self, params, well) -> np.ndarray:
        raise NotImplementedError

    # station contract ----------------------------------------------------------------------------
    def _require(self, target: str) -> None:
        if not self.supports(target):
            raise ValueError(f"{self.name} predicts {', '.join(self.targets or [])}, not {target}")

    def fit(self, train_wells, target):
        self._require(target)
        wells = list(train_wells)
        if self.uses_fitted_parameters:
            self.calibration = cal.out_of_fold(wells, target, self.fit_params, self.mean_from)
            self.params = self.fit_params(wells)
        else:
            self.params = self.fit_params(wells)
            params = self.params
            self.calibration = cal.in_sample(wells, target, lambda w: self.mean_from(params, w))

    def predict(self, well, target):
        self._require(target)
        mean = finite_or_nan(self.mean_from(self.params, well)).reshape(-1)
        if len(mean) != len(well.depth):
            raise ValueError(f"{self.name}: prediction length {len(mean)} != depth length {len(well.depth)}")
        return Prediction(mean, self.calibration.std_like(mean))

    def describe(self) -> dict:
        """Parameters and calibration for model cards and validation notes (no well data)."""
        p = self.params
        params = p.as_dict() if hasattr(p, "as_dict") else p
        return {"id": self.name, "targets": list(self.targets or []), "required_inputs": self.declared_required_inputs(),
                "params": params, "calibration": self.calibration.as_dict(), "card": self.card}
