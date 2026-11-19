"""Station wrapper: black_gc_brine_gr (Greenberg-Castagna, two brine-filled lithologies from GR)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from workstreams.black_physics.models._shared.base import PhysicsContender
from workstreams.black_physics.models._shared.curves import curve
from workstreams.black_physics.models.greenberg_castagna.equation import (GR_HIGH_PERCENTILE, GR_LOW_PERCENTILE,
                                                                          gc_brine_gr_dts, gr_well_endpoints)


@dataclass(frozen=True)
class GREndpoints:
    gr_low: float | None               # gAPI at f_shale = 0; None = unsupported
    gr_high: float | None              # gAPI at f_shale = 1
    n_wells: int
    source: str

    @property
    def usable(self) -> bool:
        return self.gr_low is not None and self.gr_high is not None and self.gr_high > self.gr_low

    def as_dict(self) -> dict:
        return asdict(self) | {"usable": self.usable}


class GreenbergCastagnaBrineGR(PhysicsContender):
    """DTS from DTC and GR. Assumes every depth is a brine-filled sand/shale mixture whose shale fraction
    scales linearly with GR between two endpoints. Carbonates, radioactive sands and gas break it."""
    name = "black_gc_brine_gr"
    inputs = ["DTC", "GR"]
    required_inputs = ["DTC", "GR"]
    targets = ["DTS"]
    card = "models/greenberg_castagna/README.md"

    def __init__(self, gr_low: float | None = None, gr_high: float | None = None, endpoint_source: str | None = None,
                 low_percentile: float = GR_LOW_PERCENTILE, high_percentile: float = GR_HIGH_PERCENTILE) -> None:
        """No arguments: fit the GR endpoints on the training wells. Give both gr_low and gr_high (gAPI) to use
        explicit endpoints for a new basin, and say where they came from in endpoint_source."""
        if (gr_low is None) != (gr_high is None):
            raise ValueError("give both gr_low and gr_high, or neither")
        self._explicit = gr_low is not None
        self._explicit_source = endpoint_source or "explicit endpoints (source not recorded)"
        self._gr = (None if gr_low is None else float(gr_low), None if gr_high is None else float(gr_high))
        self.low_percentile, self.high_percentile = float(low_percentile), float(high_percentile)
        super().__init__()

    @property
    def uses_fitted_parameters(self) -> bool:
        return not self._explicit

    def default_params(self):
        if self._explicit:
            return GREndpoints(*self._gr, 0, self._explicit_source)
        return GREndpoints(None, None, 0, "not fitted")

    def fit_params(self, train_wells):
        if self._explicit:
            return self.default_params()
        got = gr_well_endpoints([curve(w, "GR") for w in train_wells if "GR" in w.curves],
                                self.low_percentile, self.high_percentile)
        if got is None:
            return GREndpoints(None, None, 0, "no training well has finite GR")
        lo, hi, n = got
        src = (f"median over {n} training wells of each well's P{self.low_percentile:g} / P{self.high_percentile:g} "
               "of finite GR")
        if hi <= lo:
            return GREndpoints(None, None, n, src + f": high {hi:.4g} <= low {lo:.4g}, unsupported")
        return GREndpoints(lo, hi, n, src)

    def mean_from(self, params, well):
        return gc_brine_gr_dts(curve(well, "DTC"), curve(well, "GR"), params.gr_low, params.gr_high)
