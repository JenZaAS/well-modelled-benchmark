"""red_timestral_target_only: red_timestral with empty covariate dictionaries (target-only control).

Context (256), rollout (blocks of 32 using predicted medians), deeper-flank reversal and inverse-distance blend
are exactly red's `DepthForecaster`; only the covariates passed to the backend are removed. This isolates how
much of red's full-input result depends on access to the other logs. Requires the Timestral environment and the
cached weights (licence: LICENSE.md); importing this module imports neither torch nor timestral.
"""
from __future__ import annotations

from ..timestral.backend import load
from ..timestral.contender import Timestral


def _target_only_backend():
    forecast = load()

    def target_only(history, past, future, horizon):
        return forecast(history, {}, {}, horizon)

    return target_only


class TimestralTargetOnly(Timestral):
    name = "red_timestral_target_only"
    row_id = "black_timestral_target_only"   # id inside the Granite experiment's artifacts
    workstream = "red_foundation_models"
    inputs: list[str] = []
    optional_inputs: list[str] = []
    load_backend = staticmethod(_target_only_backend)
