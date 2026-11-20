"""black_timestral_target_only: red_timestral with empty covariate dictionaries (target-only control).

Context (256), rollout (blocks of 32 using predicted medians), deeper-flank reversal and inverse-distance blend
are exactly red's `DepthForecaster`; only the covariates passed to the backend are removed. This isolates how
much of red's full-input result depends on access to the other logs. Requires the red Timestral environment and
authorised weights (see LICENSE.md); importing this module imports neither torch nor timestral.
"""
from __future__ import annotations

from workstreams.red_foundation_models.models.timestral.backend import load
from workstreams.red_foundation_models.models.timestral.contender import Timestral


def _target_only_backend():
    forecast = load()

    def target_only(history, past, future, horizon):
        return forecast(history, {}, {}, horizon)

    return target_only


class TimestralTargetOnly(Timestral):
    name = "black_timestral_target_only"
    workstream = "black_physics"
    inputs: list[str] = []
    optional_inputs: list[str] = []
    load_backend = staticmethod(_target_only_backend)
