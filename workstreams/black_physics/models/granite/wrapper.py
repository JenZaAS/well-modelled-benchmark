"""Station wrappers for the physics baselines of the Granite 50 m density-gap experiment (opt-in: WMB_BLACK_GRANITE=1).

    black_linear_gap              straight line between the nearest measured flank values
    black_gardner_flank_resid     fixed Gardner from DTC + linearly interpolated flank residual

The foundation-model rows of the same experiment (TTM-R3, TSPulse, Timestral target-only) live in the red
workstream: workstreams/red_foundation_models/models/granite and models/timestral_target_only.
Whole-curve tasks have no RHOB context: these rows abstain (NaN).
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction
from wmb.contract import SCENARIO_GAP, Task
from workstreams.black_physics.models._shared.curves import curve
from workstreams.black_physics.models.gardner.equation import gardner_density


def _gap_task_from_well(well, target: str) -> Task:
    """A gap task for the viewer's Prediction tab: every NaN run of the target is a gap; no target = abstain."""
    from workstreams.black_physics.granite.gaps import inputs_frame

    depth = np.asarray(well.depth, float)
    y = well.curves.get(target)
    if y is None:
        mask = np.ones(len(depth), bool)
        return Task(well.name, depth, inputs_frame(well, target), target, mask, "missing_curve", dataset=well.dataset)
    y = np.asarray(y, float)
    return Task(well.name, depth, inputs_frame(well, target), target, ~np.isfinite(y), SCENARIO_GAP, 0.0,
                y.copy(), well.dataset, well.split)


class LinearGap(BaseContender):
    """Straight line in depth between the nearest measured RHOB above and below, within one acquisition segment."""
    name = "black_linear_gap"
    workstream = "black_physics"
    targets = ["RHOB", "DTC", "DTS"]
    inputs: list[str] = []
    required_inputs: list[str] = []
    licence_name = "MIT"
    licence_url = "https://opensource.org/license/mit"
    licence_restrictions = "none (original code, MIT)"

    def fit(self, train_wells, target):
        pass

    def predict_task(self, task: Task) -> Prediction:
        from workstreams.black_physics.validate_force import linear_gap_reference

        return linear_gap_reference(task)

    def predict(self, well, target):
        return self.predict_task(_gap_task_from_well(well, target))


def gardner_flank_residual(task: Task) -> np.ndarray:
    """Fixed Gardner from DTC, plus the observed (RHOB - Gardner) residual interpolated linearly across each gap.

    Residual anchors: the nearest depths above and below the hidden run, inside its segment, where both RHOB
    (observed) and DTC are finite. NaN where DTC is missing inside the gap or an anchor is unavailable.
    """
    from workstreams.black_physics.granite import gaps as G

    depth = np.asarray(task.depth, float)
    out = np.full(len(depth), np.nan)
    if task.target_observed is None:
        return out
    obs = np.asarray(task.target_observed, float).copy()
    obs[np.asarray(task.predict_mask, bool)] = np.nan
    dtc = task.inputs["DTC"].to_numpy(float) if "DTC" in task.inputs else np.full(len(depth), np.nan)
    g = gardner_density(dtc)
    resid = obs - g                                     # finite only where both are measured
    for a, b, s, e in G.mask_pieces(depth, task.predict_mask):
        line = G.linear_fill(depth, resid, a, b, s, e)
        if line is not None:
            out[a:b] = g[a:b] + line
    return out


class GardnerFlankResidual(BaseContender):
    """Extension row: fixed Gardner (0.31 Vp^0.25) corrected by the interpolated flank residual."""
    name = "black_gardner_flank_resid"
    workstream = "black_physics"
    targets = ["RHOB"]
    inputs = ["DTC"]
    required_inputs = ["DTC"]
    licence_name = "MIT"
    licence_url = "https://opensource.org/license/mit"
    licence_restrictions = "none for this code (original, MIT); Gardner et al. (1974) is a published method"

    def fit(self, train_wells, target):
        pass

    def predict_task(self, task: Task) -> Prediction:
        return Prediction(gardner_flank_residual(task))

    def predict(self, well, target):
        t = _gap_task_from_well(well, target)
        if "DTC" in well.curves:
            t.inputs["DTC"] = curve(well, "DTC")
        return self.predict_task(t)


EXTRA = [LinearGap, GardnerFlankResidual]
