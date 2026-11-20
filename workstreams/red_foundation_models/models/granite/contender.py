"""IBM Granite rows of the 50 m density-gap experiment (opt-in: WMB_RED_MODELS=granite; Granite environment).

    red_ttm_r3_frozen, red_tspulse_frozen          pretrained weights (TTM-R3 lite: two directional forecasts
                                                   blended by distance; TSPulse R1: masked reconstruction)
    red_ttm_r3_quicktune, red_tspulse_quicktune    the brief's small fine-tune (output heads, a few seconds)
    red_ttm_r3_finetune, red_tspulse_finetune      the stronger fine-tune (granite/finetune.py)

Target-only: the rows read only `Task.target_observed` (RHOB with the hidden interval removed) and return
full-depth arrays that are NaN outside `Task.predict_mask`. Whole-curve tasks have no RHOB context: they abstain.

The experiment itself (gaps, backends, training, calibration, decisions, report) is
workstreams/black_physics/granite. These rows load its frozen artifacts, refused with StaleArtifacts when the code,
split, data or gaps changed; on other wells, e.g. synthetic contract checks, they run a small local fit. Inside
the experiment's local artifacts each row keeps its original id (`row_id`, e.g. black_ttm_r3_tuned); the old
station ids are aliases in wmb.aliases, so earlier results still display. Importing this module imports no torch.
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction
from wmb.contract import Task

APACHE = "https://www.apache.org/licenses/LICENSE-2.0"


class _GraniteGap(BaseContender):
    workstream = "red_foundation_models"
    targets = ["RHOB"]
    inputs: list[str] = []
    required_inputs: list[str] = []
    optional_inputs: list[str] = []
    licence_name = "Apache-2.0"
    licence_url = APACHE
    licence_restrictions = "none beyond Apache-2.0 attribution (IBM Granite weights and granite-tsfm code)"
    arm = ""
    tuned = False
    row_id = ""          # the experiment's own id for this row (keys its frozen artifacts)

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
        self.backend, self.calibrator, self.provenance = artifacts.load_variant(self.row_id, train_wells, self.device)
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
        from workstreams.black_physics.models.granite.wrapper import _gap_task_from_well

        return self.predict_task(_gap_task_from_well(well, target))


class TTMR3Frozen(_GraniteGap):
    """TTM-R3 lite (IBM, pinned revision), pretrained heads, empirical D-calibrated bands."""
    row_id = "black_ttm_r3_frozen"
    name, arm, tuned = "red_ttm_r3_frozen", "ttm", False


class TTMR3QuickTune(_GraniteGap):
    """TTM-R3 lite with its trend and residual prediction heads fine-tuned on FORCE fitting wells (A)."""
    row_id = "black_ttm_r3_tuned"
    name, arm, tuned = "red_ttm_r3_quicktune", "ttm", True


class TSPulseFrozen(_GraniteGap):
    """TSPulse R1 hybrid dual-head (IBM, pinned revision), pretrained reconstruction head."""
    row_id = "black_tspulse_frozen"
    name, arm, tuned = "red_tspulse_frozen", "tspulse", False


class TSPulseQuickTune(_GraniteGap):
    """TSPulse R1 with its time-domain reconstruction head fine-tuned on FORCE fitting wells (A)."""
    row_id = "black_tspulse_tuned"
    name, arm, tuned = "red_tspulse_quicktune", "tspulse", True


class _GraniteFT(_GraniteGap):
    """A stronger fine-tune row (granite/finetune.py): checkpoint + D bands from manifest_ft.json."""

    tuned = True

    def fit(self, train_wells, target):
        from workstreams.black_physics.granite import artifacts
        from workstreams.black_physics.granite.reconstruct import make_reconstructor

        if not self.supports(target):
            raise ValueError(f"{self.name} predicts RHOB, not {target}")
        self.backend, self.calibrator, self.provenance = artifacts.load_ft_variant(self.row_id, train_wells, self.device)
        self.reconstructor = make_reconstructor(self.backend)


class TTMR3FineTune(_GraniteFT):
    """TTM-R3 lite after the stronger fine-tune (blended two-sided loss, 13,824 A gaps; scope picked on B)."""
    row_id = "black_ttm_r3_ft"
    name, arm = "red_ttm_r3_finetune", "ttm"


class TSPulseFineTune(_GraniteFT):
    """TSPulse R1 after the stronger fine-tune (up to whole-model training on 13,824 A gaps; scope picked on B)."""
    row_id = "black_tspulse_ft"
    name, arm = "red_tspulse_finetune", "tspulse"


GRANITE = [TTMR3Frozen, TTMR3QuickTune, TSPulseFrozen, TSPulseQuickTune]
FT = [TTMR3FineTune, TSPulseFineTune]          # separate list: the experiment's `final` stage iterates GRANITE only
