"""Station wrappers: black_gardner (locally fitted power law) and black_gardner_fixed (published)."""
from __future__ import annotations

from workstreams.black_physics.models._shared.base import PhysicsContender
from workstreams.black_physics.models._shared.curves import curve, pooled
from workstreams.black_physics.models.gardner.equation import PUBLISHED, fit_gardner, gardner_density


class _GardnerBase(PhysicsContender):
    inputs = ["DTC"]
    required_inputs = ["DTC"]
    targets = ["RHOB"]
    card = "models/gardner/README.md"

    def default_params(self):
        return PUBLISHED

    def mean_from(self, params, well):
        return gardner_density(curve(well, "DTC"), params.a, params.b)

    @property
    def a(self) -> float:
        return self.params.a

    @property
    def b(self) -> float:
        return self.params.b

    @property
    def sigma(self) -> float | None:
        return self.calibration.scale


class Gardner(_GardnerBase):
    """RHOB from DTC with a and b refitted on the training wells (published fallback)."""
    name = "black_gardner"

    @property
    def uses_fitted_parameters(self) -> bool:
        return True

    def fit_params(self, train_wells):
        d = pooled(train_wells, "DTC", "RHOB")
        return fit_gardner(d[:, 0], d[:, 1])


class GardnerFixed(_GardnerBase):
    """RHOB from DTC with the published a = 0.31, b = 0.25 in every basin."""
    name = "black_gardner_fixed"
