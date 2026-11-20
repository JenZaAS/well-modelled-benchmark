"""Frozen settings of the Granite 50 m density-gap experiment.

Every number that shapes a split, a mask, a training run or a decision lives here, so the saved
configuration (`as_dict`, `config_hash`) identifies a run. Change a value, and every artifact that
records the old hash is refused by the station wrappers until it is rebuilt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

TARGET = "RHOB"
TEST_NAME = "rhob_gap50"
DATASET = "force2020"

TTM_REPO = "ibm-granite/granite-timeseries-ttm-r3"
TTM_REVISION = "226cdc78f96e1304760ca141bd82d7d91be08bd8"      # branch 512-336-dec-512-lite-r3
TTM_BRANCH = "512-336-dec-512-lite-r3"
TSPULSE_REPO = "ibm-granite/granite-timeseries-tspulse-r1"
TSPULSE_REVISION = "b12164578f7b893ada0028c00d292ba10383d25a"  # branch tspulse-hybrid-dualhead-512-p8-r1
TSPULSE_BRANCH = "tspulse-hybrid-dualhead-512-p8-r1"
TSFM_COMMIT = "31fa2f01573341f17e9997280f61481de18ce28c"
DECIMA_REPO = "amyrmahdy/decima-small"
DECIMA_REVISION = "480c4ea9e54f2946b947fb3ceb3c7cc7ece77522"

ARMS = ("ttm", "tspulse")
VARIANT_IDS = {
    ("ttm", False): "black_ttm_r3_frozen",
    ("ttm", True): "black_ttm_r3_tuned",
    ("tspulse", False): "black_tspulse_frozen",
    ("tspulse", True): "black_tspulse_tuned",
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    gap_length_m: float = 50.0
    # whole-gap decision: pass when mean |reconstruction - measured| over the gap <= tolerance
    tolerance_g_cm3: float = 0.05
    accept_threshold: float = 0.80
    # well split (official FORCE training wells only; test wells stay sealed)
    split_sizes: dict = field(default_factory=lambda: {"A": 54, "B": 12, "C": 20, "D": 12})
    gaps_per_well: dict = field(default_factory=lambda: {"A": 32, "B": 8, "C": 8, "D": 8})
    # model windows
    ttm_context: int = 512
    ttm_horizon: int = 336
    tspulse_window: int = 512
    min_flank_samples: int = 16          # fewer observed samples on a side = that side is unusable (TSPulse)
    # TTM short-context rule: a flank is forecast from only if it has >= this many contiguous measured samples.
    # Padding to 512 stays unobserved (IBM's left zero padding in scaled space, mask False). Chosen on fitting
    # wells (A, 500 gaps, frozen TTM) before any B selection: with 16, 15 % of directed contexts were short and
    # some forecasts diverged (gap MAE up to 14.8 g/cm3); with 256 the worst gap MAE was 0.30. See README.
    ttm_min_context: int = 256
    # fine-tuning time box (per model)
    lr: float = 3e-4
    weight_decay: float = 0.01
    effective_batch: int = 8
    max_updates: int = 300
    max_minutes: float = 45.0
    eval_every: int = 25
    patience: int = 3
    measure_steps: int = 10
    # intervals
    ttm_scale_floor: float = 0.005       # g/cm3, floor on local P90-P10 used as the interval scale
    # decisions
    l2: float = 1.0                      # ridge penalty (standardized features) for every logistic fit
    recal_l2: float = 1.0                # penalty on the intercept/slope recalibration (pulls slope to 1)
    prevalence_smoothing: float = 1.0    # Laplace smoothing for a one-class fallback
    bootstrap: int = 2000

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(target=TARGET, test=TEST_NAME, dataset=DATASET,
                 ttm=f"{TTM_REPO}@{TTM_REVISION}", tspulse=f"{TSPULSE_REPO}@{TSPULSE_REVISION}",
                 tsfm=TSFM_COMMIT, decima=f"{DECIMA_REPO}@{DECIMA_REVISION}")
        return d


CONFIG = ExperimentConfig()


def config_hash(cfg: ExperimentConfig = CONFIG) -> str:
    return hashlib.sha256(json.dumps(cfg.as_dict(), sort_keys=True).encode()).hexdigest()[:16]


def work_dir() -> Path:
    """Local artifact folder (checkpoints, split manifests, full-precision predictions); never committed."""
    from wmb import paths

    d = Path(paths.work_root()) / "black_granite"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hf_home() -> Path:
    from wmb import paths

    return Path(paths.cache_dir()) / "hf"
