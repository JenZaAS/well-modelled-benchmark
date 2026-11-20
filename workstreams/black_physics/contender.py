"""Black team: physics / empirical baselines. Fast, explainable floors to beat.

Registration file for the runner (explicit imports, no auto-discovery). Each model lives in
models/<family>/ with its equation, a thin station wrapper, a model card (README.md) and LICENSE.md.

| ID                    | Target <- inputs | Equation                                                        |
|-----------------------|------------------|-----------------------------------------------------------------|
| black_gardner         | RHOB <- DTC      | Gardner power law, a and b refitted on training wells           |
| black_gardner_fixed   | RHOB <- DTC      | Gardner, published rho = 0.31 Vp^0.25 (Vp in m/s)               |
| black_castagna_dts    | DTS  <- DTC      | Castagna 1985 mudrock line, Vs = 0.8621 Vp - 1.1724 (km/s)      |
| black_gc_brine_gr     | DTS  <- DTC, GR  | Greenberg-Castagna, brine-filled sand/shale fractions from GR   |
| black_wyllie_nphi     | DTC  <- NPHI     | Wyllie time average, quartz 55.5 / water 189 us/ft              |
| black_density_nphi    | RHOB <- NPHI     | Volumetric density, quartz 2.65 / water 1.00 g/cm3              |
| black_vpvs_ratio      | DTS  <- DTC      | One fitted DTS/DTC ratio (empirical control)                    |

`Gardner` and `MudrockDTS` stay importable from this module for existing callers.

Opt-in rows of the Granite density-gap experiment (WMB_BLACK_GRANITE=1, NumPy only): black_linear_gap
(interpolation reference) and black_gardner_flank_resid (Gardner + flank residual). The experiment's
foundation-model rows (red_ttm_r3_*, red_tspulse_*, red_timestral_target_only) are registered by the red
workstream (WMB_RED_MODELS=granite,timestral_target_only); the experiment itself stays in granite/.
"""
from __future__ import annotations

import os

from workstreams.black_physics.models.castagna_mudrock.wrapper import MudrockDTS
from workstreams.black_physics.models.gardner.wrapper import Gardner, GardnerFixed
from workstreams.black_physics.models.granite.wrapper import EXTRA as GRANITE_EXTRA
from workstreams.black_physics.models.greenberg_castagna.wrapper import GreenbergCastagnaBrineGR
from workstreams.black_physics.models.volumetric_density.wrapper import DensityNPHI
from workstreams.black_physics.models.vpvs_ratio.wrapper import VpVsRatio
from workstreams.black_physics.models.wyllie.wrapper import WyllieNPHI

CONTENDERS = [Gardner, MudrockDTS, GardnerFixed, GreenbergCastagnaBrineGR, WyllieNPHI, DensityNPHI, VpVsRatio]

GRANITE_ON = os.getenv("WMB_BLACK_GRANITE", "").strip() not in ("", "0")
if GRANITE_ON:
    CONTENDERS += [*GRANITE_EXTRA]
# The rows left out above, with the flag that registers them; the viewer's Prediction tab lists them as opt-in.
OPT_IN = {} if GRANITE_ON else {cls: "WMB_BLACK_GRANITE=1" for cls in GRANITE_EXTRA}

__all__ = ["CONTENDERS", "OPT_IN", "DensityNPHI", "Gardner", "GardnerFixed", "GreenbergCastagnaBrineGR", "MudrockDTS",
           "VpVsRatio", "WyllieNPHI"]
