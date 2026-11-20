"""Station wrappers for the Granite 50 m density-gap experiment (opt-in: WMB_BLACK_GRANITE=1).

    black_ttm_r3_frozen / black_ttm_r3_tuned       IBM TTM-R3 lite, two directional forecasts blended by distance
    black_tspulse_frozen / black_tspulse_tuned     IBM TSPulse R1, masked reconstruction of a centred window
    black_linear_gap                               straight line between the nearest measured flank values
    black_gardner_flank_resid                      fixed Gardner from DTC + linearly interpolated flank residual

Target-only rows read only `Task.target_observed` (RHOB with the hidden interval removed) and return full-depth
arrays that are NaN outside `Task.predict_mask`. Whole-curve tasks have no RHOB context: they abstain (NaN).
The Granite rows load the frozen artifacts written by `python -m workstreams.black_physics.granite` (see
`granite/artifacts.py`); on other wells, e.g. synthetic contract checks, they run a small local fit instead.
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction
from wmb.contract import SCENARIO_GAP, Task
from workstreams.black_physics.models._shared.curves import curve
from workstreams.black_physics.models.gardner.equation import gardner_density

APACHE = "https://www.apache.org/licenses/LICENSE-2.0"


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


class _GraniteGap(BaseContender):
    workstream = "black_physics"
    targets = ["RHOB"]
    inputs: list[str] = []
    required_inputs: list[str] = []
    optional_inputs: list[str] = []
    licence_name = "Apache-2.0"
    licence_url = APACHE
    licence_restrictions = "none beyond Apache-2.0 attribution (IBM Granite weights and granite-tsfm code)"
    arm = ""
    tuned = False

    def __init__(self, device=None):
        self.device = device
        self.backend = self.calibrator = self.reconstructor = None
        self.provenance: dict = {}
        self.last_pieces: list = []

    def fit(self, train_wells, target):
        from workstreams.black_physics.granite import artifacts
        from workstreams.black_physics.granite.reconstruct import make_reconstructor

        if not self.supports(target):
            raise ValueError(f"{self.name} predicts RHOB, not {target}")
        self.backend, self.calibrator, self.provenance = artifacts.load_variant(self.name, train_wells, self.device)
        self.reconstructor = make_reconstructor(self.backend)

    def predict_with_pieces(self, tasks: list[Task]) -> tuple[list[Prediction], list[list]]:
        if self.reconstructor is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        all_pieces = self.reconstructor.run(tasks)
        preds = []
        for t, pieces in zip(tasks, all_pieces):
            n = len(t.depth)
            arr = {k: np.full(n, np.nan) for k in ("p50", "p05", "p25", "p75", "p95")}
            for p in pieces:
                arr["p50"][p.a:p.b] = p.p50
                for k, v in self.calibrator.bands(p).items():
                    arr[k][p.a:p.b] = v
            q = {0.05: arr["p05"], 0.25: arr["p25"], 0.5: arr["p50"], 0.75: arr["p75"], 0.95: arr["p95"]}
            preds.append(Prediction(mean=arr["p50"], quantiles=q, p05=arr["p05"], p95=arr["p95"]))
        return preds, all_pieces

    def predict_task(self, task: Task) -> Prediction:
        preds, pieces = self.predict_with_pieces([task])
        self.last_pieces = pieces[0]
        return preds[0]

    def predict_batch(self, tasks: list[Task]) -> list[Prediction]:
        return self.predict_with_pieces(tasks)[0]

    def predict(self, well, target):
        return self.predict_task(_gap_task_from_well(well, target))


class TTMR3Frozen(_GraniteGap):
    """TTM-R3 lite (IBM, pinned revision), pretrained heads, empirical D-calibrated bands."""
    name, arm, tuned = "black_ttm_r3_frozen", "ttm", False


class TTMR3Tuned(_GraniteGap):
    """TTM-R3 lite with its trend and residual prediction heads fine-tuned on FORCE fitting wells (A)."""
    name, arm, tuned = "black_ttm_r3_tuned", "ttm", True


class TSPulseFrozen(_GraniteGap):
    """TSPulse R1 hybrid dual-head (IBM, pinned revision), pretrained reconstruction head."""
    name, arm, tuned = "black_tspulse_frozen", "tspulse", False


class TSPulseTuned(_GraniteGap):
    """TSPulse R1 with its time-domain reconstruction head fine-tuned on FORCE fitting wells (A)."""
    name, arm, tuned = "black_tspulse_tuned", "tspulse", True


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


GRANITE = [TTMR3Frozen, TTMR3Tuned, TSPulseFrozen, TSPulseTuned]
EXTRA = [LinearGap, GardnerFlankResidual]
