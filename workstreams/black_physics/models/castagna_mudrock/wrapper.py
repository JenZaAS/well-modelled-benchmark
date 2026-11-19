"""Station wrapper: black_castagna_dts (published mudrock line, DTS from DTC)."""
from __future__ import annotations

from workstreams.black_physics.models._shared.base import PhysicsContender
from workstreams.black_physics.models._shared.curves import curve
from workstreams.black_physics.models.castagna_mudrock.equation import (INTERCEPT_KM_S, SLOPE, VS_FLOOR_KM_S,
                                                                        mudrock_dts)


class MudrockDTS(PhysicsContender):
    """DTS from DTC on the 1985 mudrock line; fixed coefficients, uncertainty from training residuals."""
    name = "black_castagna_dts"
    inputs = ["DTC"]
    required_inputs = ["DTC"]
    targets = ["DTS"]
    card = "models/castagna_mudrock/README.md"

    @staticmethod
    def _dts(dtc):
        """Kept for existing callers: mudrock DTS (us/ft) from DTC (us/ft) with the Vs > 0.3 km/s guard."""
        return mudrock_dts(dtc)

    def default_params(self):
        return {"slope": SLOPE, "intercept_km_s": INTERCEPT_KM_S, "vs_floor_km_s": VS_FLOOR_KM_S}

    def mean_from(self, params, well):
        return mudrock_dts(curve(well, "DTC"), params["vs_floor_km_s"])

    @property
    def sigma(self) -> float | None:
        return self.calibration.scale
