"""Station wrapper: black_wyllie_nphi (Wyllie time average with measured NPHI as apparent porosity)."""
from __future__ import annotations

from workstreams.black_physics.models._shared.base import PhysicsContender
from workstreams.black_physics.models._shared.curves import curve
from workstreams.black_physics.models._shared.units import require_positive
from workstreams.black_physics.models.wyllie.equation import QUARTZ_DT_US_FT, WATER_DT_US_FT, wyllie_dtc


class WyllieNPHI(PhysicsContender):
    """DTC from NPHI. Reference configuration: quartz matrix 55.5 us/ft, water 189 us/ft, NPHI used as porosity.
    Different endmembers or an independent porosity are a different configuration: register a new ID."""
    name = "black_wyllie_nphi"
    inputs = ["NPHI"]
    required_inputs = ["NPHI"]
    targets = ["DTC"]
    card = "models/wyllie/README.md"

    def __init__(self, dt_matrix: float = QUARTZ_DT_US_FT, dt_fluid: float = WATER_DT_US_FT) -> None:
        require_positive(dt_matrix=dt_matrix, dt_fluid=dt_fluid)
        self.dt_matrix, self.dt_fluid = float(dt_matrix), float(dt_fluid)
        super().__init__()

    def default_params(self):
        return {"dt_matrix_us_ft": self.dt_matrix, "dt_fluid_us_ft": self.dt_fluid, "porosity": "NPHI (apparent, v/v)"}

    def mean_from(self, params, well):
        return wyllie_dtc(curve(well, "NPHI"), params["dt_matrix_us_ft"], params["dt_fluid_us_ft"])
