"""One Gaussian scale per model, estimated from residuals that give every well equal weight.

    scale = sqrt( mean over wells of (mean over that well's rows of residual^2) )

Rows count only where the measured target is finite and > 0 *and* the model's prediction is finite.
Fixed equations use their residuals on the training wells. Models with fitted parameters use deterministic
out-of-fold predictions grouped by whole well (up to five folds), so a well's residuals never come from
parameters it helped fit. Fewer than two contributing wells, or a scale that is not finite and > 0, gives
no scale: the model then returns point predictions with std=None.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from workstreams.black_physics.models._shared.curves import curve, has_positive_target

MAX_FOLDS = 5
FOLD_SEED = 2026
MIN_WELLS = 2


@dataclass(frozen=True)
class Calibration:
    scale: float | None          # one-sigma, target units; None = no uncertainty
    n_wells: int                 # wells that contributed residuals
    n_rows: int                  # residual rows over those wells
    method: str
    note: str = ""

    @classmethod
    def unfitted(cls) -> "Calibration":
        return cls(None, 0, 0, "none", "model not fitted")

    def std_like(self, mean: np.ndarray) -> np.ndarray | None:
        """std on exactly the support of the point estimate (NaN where mean is NaN), or None."""
        if self.scale is None:
            return None
        return np.where(np.isfinite(mean), self.scale, np.nan)

    def as_dict(self) -> dict:
        return {"scale": self.scale, "n_wells": self.n_wells, "n_rows": self.n_rows, "method": self.method,
                "note": self.note}


def residuals(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """pred - y on rows with finite, positive measured y and finite pred."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & (y > 0) & np.isfinite(pred)
    return pred[ok] - y[ok]


def summarise(per_well: Sequence[np.ndarray], method: str) -> Calibration:
    """Equal-weight RMS over wells with at least one residual."""
    used = [r for r in per_well if len(r)]
    n_rows = int(sum(len(r) for r in used))
    if len(used) < MIN_WELLS:
        return Calibration(None, len(used), n_rows, method, f"fewer than {MIN_WELLS} wells with usable residuals")
    scale = float(np.sqrt(np.mean([np.mean(r ** 2) for r in used])))
    if not np.isfinite(scale) or scale <= 0:
        return Calibration(None, len(used), n_rows, method, "residual scale not finite and positive")
    return Calibration(scale, len(used), n_rows, method)


def whole_well_folds(names: Sequence[str], max_folds: int = MAX_FOLDS, seed: int = FOLD_SEED) -> list[np.ndarray]:
    """Deterministic whole-well folds over positions 0..n-1: sort by name, permute with default_rng(seed),
    array_split into min(max_folds, n) folds."""
    order = sorted(range(len(names)), key=lambda i: (names[i], i))
    if not order:
        return []
    perm = np.asarray(order)[np.random.default_rng(seed).permutation(len(order))]
    return [f for f in np.array_split(perm, min(max_folds, len(order))) if len(f)]


def in_sample(wells: Sequence, target: str, predict: Callable) -> Calibration:
    """Fixed equation: residuals of predict(well_without_target) on each training well."""
    per_well = [residuals(curve(w, target), predict(w.without(target))) for w in wells if target in w.curves]
    return summarise(per_well, "training-well residuals (fixed equation)")


def out_of_fold(wells: Sequence, target: str, fit: Callable, predict: Callable,
                max_folds: int = MAX_FOLDS, seed: int = FOLD_SEED) -> Calibration:
    """Fitted parameters: folds over wells with a positive measured target; each fold is predicted with
    parameters fitted on every other training well (including wells without the target)."""
    wells = list(wells)
    eligible = [i for i, w in enumerate(wells) if has_positive_target(w, target)]
    folds = whole_well_folds([wells[i].name for i in eligible], max_folds, seed)
    method = f"out-of-fold residuals ({len(folds)} whole-well folds, seed {seed})"
    if len(eligible) < MIN_WELLS:
        return Calibration(None, 0, 0, method, f"fewer than {MIN_WELLS} training wells with the target")
    per_well = []
    for fold in folds:
        held = {eligible[k] for k in fold}
        params = fit([w for i, w in enumerate(wells) if i not in held])
        for i in sorted(held):
            per_well.append(residuals(curve(wells[i], target), predict(params, wells[i].without(target))))
    return summarise(per_well, method)
