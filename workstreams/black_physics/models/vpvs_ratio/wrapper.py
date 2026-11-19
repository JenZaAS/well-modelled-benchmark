"""Station wrapper: black_vpvs_ratio (one fitted DTS/DTC ratio, an empirical control)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from workstreams.black_physics.models._shared.base import PhysicsContender
from workstreams.black_physics.models._shared.curves import curve
from workstreams.black_physics.models.vpvs_ratio.equation import fit_ratio, ratio_dts


@dataclass(frozen=True)
class RatioFit:
    ratio: float | None                # None = no paired training well; predictions are unsupported
    n_wells: int

    def as_dict(self) -> dict:
        return asdict(self)


class VpVsRatio(PhysicsContender):
    """DTS = r x DTC with r the median over training wells of each well's median DTS/DTC."""
    name = "black_vpvs_ratio"
    inputs = ["DTC"]
    required_inputs = ["DTC"]
    targets = ["DTS"]
    card = "models/vpvs_ratio/README.md"

    @property
    def uses_fitted_parameters(self) -> bool:
        return True

    def default_params(self):
        return RatioFit(None, 0)

    def fit_params(self, train_wells):
        pairs = [(curve(w, "DTC"), curve(w, "DTS")) for w in train_wells if "DTC" in w.curves and "DTS" in w.curves]
        return RatioFit(*fit_ratio(pairs))

    def mean_from(self, params, well):
        return ratio_dts(curve(well, "DTC"), params.ratio)
