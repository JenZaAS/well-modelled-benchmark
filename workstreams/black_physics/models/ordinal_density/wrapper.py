"""Binned density forecasts for RHOB gaps (opt-in: WMB_BLACK_ORDINAL=1): a base gap fill plus a frozen 15-bin head.

    black_ordinal_timestral          red_timestral's fill (RHOB flanks + other logs), empirical errors by flank
                                     variability                                      (runs in the Timestral env)
    black_ordinal_tspulse_finetune   red_tspulse_finetune's fill (RHOB only), softmax head on the fill, gap
                                     geometry and the sonic (Gardner) difference       (runs in the Granite env)

p50 is the base model's own fill, unchanged, so MAE equals the base row's. The head adds a probability for each
density range at every hidden depth (Prediction.bin_edges / bin_probs), drawn by the viewer as a heatmap and scored
as RPS_bins. The TSPulse head is a softmax over the method's 15 fixed ranges (below 1.60, 1.60-1.70, ..., 2.80-2.90,
2.90 and above, g/cm3). The Timestral head counts past errors, so it gives the same distribution on 0.02 g/cm3 ranges
(67 bins; they sum back exactly to its 15-bin forecast), which shows its shape instead of one 70 % range.
p05/p25/p75/p95 are quantiles of the binned distribution, widened where needed to contain p50 (counted in
`provenance["band_widened"]`).

The heads were fitted on the C wells' gaps and selected on the D wells (workstreams/black_physics/ordinal_density).
They are loaded from work_root()/black_ordinal_density/frozen/ and refused (StaleHead) unless their sha256 matches
frozen_heads.yaml in this folder and the inference code is unchanged; the TSPulse row also checks the base
checkpoint. Fitted on other wells (e.g. the synthetic contract check), a row reads no FORCE-trained file, like the
Granite rows: the bins then come from a normal distribution on the base model's own band (flagged in provenance).
Whole-curve tasks have no gap flanks: these rows abstain (NaN) on them.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from wmb.contender import BaseContender, Prediction
from wmb.contract import SCENARIO_GAP, Task
from workstreams.black_physics.ordinal_density import bins as B
from workstreams.black_physics.ordinal_density.infer import Head, StaleHead
from workstreams.red_foundation_models.models.granite.contender import TSPulseFineTune
from workstreams.red_foundation_models.models.timestral.contender import Timestral

LOCK = Path(__file__).resolve().parent / "frozen_heads.yaml"
QUANTILES = (0.05, 0.25, 0.75, 0.95)
Z = {0.05: -1.6448536, 0.25: -0.6744898, 0.75: 0.6744898, 0.95: 1.6448536}
FALLBACK_SIGMA = 0.10          # g/cm3, when a locally fitted base model reports no band


def frozen_lock() -> dict:
    import yaml

    return (yaml.safe_load(LOCK.read_text(encoding="utf-8")) if LOCK.is_file() else None) or {}


def native_band(pred: Prediction) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """(lo, hi, z of hi) of the base model's own interval, chosen as for the fills the heads were fitted on:
    P10/P90 (z 1.28), else P05/P95 (z 1.64), else none."""
    from workstreams.black_physics.granite.decide_any import _band

    lo, hi, name = _band(pred)
    return lo, hi, {"P10-P90": 1.2815516, "P05-P95": 1.6448536}.get(name)


def normal_bins(mu: np.ndarray, sigma: np.ndarray, edges=B.EDGES) -> np.ndarray:
    """[n, len(edges) + 1] probabilities of N(mu, sigma) falling in each bin (end bins open)."""
    cdf = np.vectorize(lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), otypes=[float])
    c = cdf((np.asarray(edges, float)[None, :] - mu[:, None]) / sigma[:, None])
    return np.diff(np.concatenate([np.zeros((len(mu), 1)), c, np.ones((len(mu), 1))], axis=1), axis=1)


class _OrdinalGap(BaseContender):
    workstream = "black_physics"
    targets = ["RHOB"]
    base_cls: type[BaseContender] = BaseContender       # the station row whose fills the head was fitted on
    bin_step: float | None = None                       # finer nested ranges (empirical heads only); None: 15 bins
    licence_name = "MIT"
    licence_url = "https://opensource.org/licenses/MIT"

    def __init__(self):
        self.base = self.base_cls()
        self.head: Head | None = None
        self.local = False
        self.provenance: dict = {}

    def fit(self, train_wells, target):
        from workstreams.black_physics.granite.artifacts import is_official

        if not self.supports(target):
            raise ValueError(f"{self.name} predicts RHOB, not {target}")
        self.base.fit(train_wells, target)
        model = self.base_cls.name
        self.local = not is_official(train_wells)
        if self.local:                                   # never read FORCE-trained files off the FORCE wells
            self.head = None
            self.provenance = {"base": model, "head": "normal on the base band (not FORCE-trained; contract check)",
                               "base_provenance": getattr(self.base, "provenance", None), "band_widened": 0}
            return
        entry = frozen_lock().get(model)
        if not entry:
            raise StaleHead(f"{LOCK.name} lists no frozen head for {model}; run the ordinal_density `freeze` stage")
        self.head, rec = Head.load_frozen(model, entry["sha256"])
        want = rec.get("base_checkpoint_sha256")
        got = (getattr(self.base, "provenance", None) or {}).get("checkpoint_sha256")
        if want and got != want:
            raise StaleHead(f"{model}'s checkpoint ({str(got)[:12]}) is not the one its head was fitted on "
                            f"({want[:12]}); re-run decide_any predict, the ordinal_density stages and `freeze`")
        self.provenance = {"base": model, "head": rec["head"], "head_sha256": entry["sha256"], "run_id": rec["run_id"],
                           "bin_step": self.bin_step, "base_provenance": getattr(self.base, "provenance", None),
                           "band_widened": 0}

    def _with_bins(self, task: Task, base: Prediction) -> Prediction:
        n = len(task.depth)
        if task.scenario != SCENARIO_GAP or task.target_observed is None:
            return Prediction(mean=np.full(n, np.nan))
        p50 = np.asarray(base.p50 if base.p50 is not None else base.mean, float)
        lo, hi, z = native_band(base)
        edges = B.fine_edges(self.bin_step) if self.bin_step else B.EDGES
        q = {level: np.full(n, np.nan) for level in QUANTILES}
        bins = np.full((n, len(edges) + 1), np.nan)
        if self.local:
            rows = np.flatnonzero(task.predict_mask & np.isfinite(p50))
            sigma = np.full(len(rows), FALLBACK_SIGMA)
            if z is not None:
                w = (np.asarray(hi, float) - np.asarray(lo, float))[rows] / (2 * z)
                sigma = np.where(np.isfinite(w) & (w > 0), w, sigma)
            bins[rows] = normal_bins(p50[rows], sigma, edges)
            for level in QUANTILES:
                q[level][rows] = p50[rows] + Z[level] * sigma
        else:
            out = self.head.predict(task, p50, lo, hi, edges=edges if self.bin_step else None)
            idx, probs = out["index"], out["probs"]
            ok = np.isfinite(probs).all(axis=1) & np.isfinite(p50[idx])
            rows = idx[ok]
            bins[rows] = probs[ok]
            for level in QUANTILES:
                q[level][rows] = (B.quantile_on(probs[ok], edges, level, self.head.tails) if self.bin_step
                                  else B.quantile(probs[ok], level, self.head.tails))
        mid = p50[rows]
        widened = 0
        for level in QUANTILES:                                   # keep p05 <= p25 <= p50 <= p75 <= p95
            v = q[level][rows]
            fixed = np.minimum(v, mid) if level < 0.5 else np.maximum(v, mid)
            widened += int((fixed != v).sum())
            q[level][rows] = fixed
        self.provenance["band_widened"] = self.provenance.get("band_widened", 0) + widened
        p50_out = np.where(task.predict_mask, p50, np.nan)
        return Prediction(mean=p50_out, quantiles={0.05: q[0.05], 0.25: q[0.25], 0.5: p50_out, 0.75: q[0.75],
                                                   0.95: q[0.95]},
                          p05=q[0.05], p95=q[0.95], bin_edges=np.array(edges, float), bin_probs=bins)

    def predict_task(self, task: Task) -> Prediction:
        if self.head is None and not self.local:
            raise RuntimeError(f"{self.name}: call fit() first")
        return self._with_bins(task, self.base.predict_task(task))

    def predict_batch(self, tasks: list[Task]) -> list[Prediction]:
        if self.head is None and not self.local:
            raise RuntimeError(f"{self.name}: call fit() first")
        return [self._with_bins(t, p) for t, p in zip(tasks, self.base.predict_batch(tasks))]

    def predict(self, well, target):
        from workstreams.black_physics.models.granite.wrapper import _gap_task_from_well

        return self.predict_task(_gap_task_from_well(well, target))


class OrdinalTimestral(_OrdinalGap):
    """Timestral-5M's gap fill (RHOB flanks + other logs) with 0.02 g/cm3 range probabilities from its past errors."""
    name = "black_ordinal_timestral"
    base_cls = Timestral
    bin_step = 0.02
    inputs: list[str] = []
    required_inputs: list[str] = []
    optional_inputs = ["GR", "RDEP", "RMED", "CALI", "NPHI", "PEF", "DTC", "DTS"]
    licence_name = "MIT (head and wrapper); base model: Modified MIT (weights), Apache-2.0 (inference code)"
    licence_restrictions = Timestral.licence_restrictions


class OrdinalTSPulseFineTune(_OrdinalGap):
    """TSPulse R1 (stronger fine-tune) gap fill with 15-bin probabilities from a softmax head (fill, gap, sonic)."""
    name = "black_ordinal_tspulse_finetune"
    base_cls = TSPulseFineTune
    inputs: list[str] = []
    required_inputs: list[str] = []
    optional_inputs = ["DTC"]
    licence_name = "MIT (head and wrapper); base model: Apache-2.0 (IBM Granite weights and granite-tsfm code)"
    licence_restrictions = "none beyond Apache-2.0 attribution for the base model"

    def __init__(self, device=None):
        super().__init__()
        self.base = TSPulseFineTune(device)


ORDINAL = [OrdinalTimestral, OrdinalTSPulseFineTune]
