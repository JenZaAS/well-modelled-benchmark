"""Station wrapper: black_density_nphi (volumetric density mixing with measured NPHI as porosity)."""
from __future__ import annotations

from workstreams.black_physics.models._shared.base import PhysicsContender
from workstreams.black_physics.models._shared.curves import curve
from workstreams.black_physics.models._shared.units import require_positive
from workstreams.black_physics.models.volumetric_density.equation import QUARTZ_DENSITY, WATER_DENSITY, bulk_density


class DensityNPHI(PhysicsContender):
    """RHOB from NPHI. Reference configuration: quartz 2.65 g/cm3, water 1.00 g/cm3, NPHI used as porosity.
    Different endmembers or an independent porosity are a different configuration: register a new ID."""
    name = "black_density_nphi"
    inputs = ["NPHI"]
    required_inputs = ["NPHI"]
    targets = ["RHOB"]
    card = "models/volumetric_density/README.md"

    def __init__(self, rho_matrix: float = QUARTZ_DENSITY, rho_fluid: float = WATER_DENSITY) -> None:
        require_positive(rho_matrix=rho_matrix, rho_fluid=rho_fluid)
        self.rho_matrix, self.rho_fluid = float(rho_matrix), float(rho_fluid)
        super().__init__()

    def default_params(self):
        return {"rho_matrix_g_cm3": self.rho_matrix, "rho_fluid_g_cm3": self.rho_fluid,
                "porosity": "NPHI (apparent, v/v)"}

    def mean_from(self, params, well):
        return bulk_density(curve(well, "NPHI"), params["rho_matrix_g_cm3"], params["rho_fluid_g_cm3"])
