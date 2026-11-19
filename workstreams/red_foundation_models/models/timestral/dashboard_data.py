"""Local dashboard analyses using the same red gap task as the CLI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wmb.contender import BaseContender
from wmb.io import Well, read_force_csv, read_las
from wmb.scoring import regression_scores
from wmb.splits import well_split
from ...experiment import depth_interpolation, masked_case


@dataclass
class Validation:
    """Split inventory, paired sample scores, and per-well depth tracks."""

    inventory: list[dict]
    scores: dict[str, dict]
    tracks: dict[str, pd.DataFrame]


def load_local(path: Path) -> list[Well]:
    """Read an existing local LAS or FORCE-format CSV; never accept URLs."""
    if not path.is_file():
        raise ValueError('Select an existing local file.')
    if path.suffix.lower() == '.las':
        return [read_las(path)]
    if path.suffix.lower() == '.csv':
        return read_force_csv(path)
    raise ValueError('Expected LAS or semicolon-separated FORCE CSV.')


def predict_tracks(model: BaseContender, well: Well, target: str) -> pd.DataFrame:
    """Predict with a fitted model; display estimates only where the input is missing."""
    observed = well.curves.get(target, np.full(well.depth.shape, np.nan))
    missing = ~np.isfinite(observed)
    pred = model.predict(well, target)
    quantiles = pred.quantiles or {}
    return pd.DataFrame({
        'Depth': well.depth, 'Observed': observed,
        'P50': np.where(missing, pred.mean, np.nan),
        'P10': np.where(missing, quantiles.get(0.1, np.nan), np.nan),
        'P90': np.where(missing, quantiles.get(0.9, np.nan), np.nan),
        'Sigma': np.where(missing, pred.std if pred.std is not None else np.nan, np.nan),
    })


def validate(model: BaseContender, train: list[Well], test: list[Well],
             target: str, seed: int = 0) -> Validation:
    """Evaluate on held-out training wells; report official test inventory, not scores."""
    if {w.name for w in train} & {w.name for w in test}:
        raise ValueError('Train and test well names overlap.')
    fit, validation = well_split(train, test_fraction=0.2, seed=seed)
    inventory = [dict(split=name, wells=len(wells), samples=sum(len(w.depth) for w in wells),
                      observed_target=sum(int(np.isfinite(w.curves[target]).sum())
                                          for w in wells if target in w.curves))
                 for name, wells in [('Fit', fit), ('Validation', validation), ('Test (unused)', test)]]
    model.fit(fit, target)
    tracks = {}
    rows = []
    for i, well in enumerate(sorted(validation, key=lambda w: w.name)):
        if not well.has(target):
            continue
        hidden, mask = masked_case(well, target, 'gap', seed + i)
        if not mask.any():
            continue
        frame = predict_tracks(model, hidden, target)
        frame['Hidden truth'] = np.where(mask, well.curves[target], np.nan)
        frame['Interpolation'] = np.where(mask, depth_interpolation(hidden, target), np.nan)
        tracks[well.name] = frame
        rows.append(frame.loc[mask, ['Hidden truth', 'P50', 'Sigma', 'Interpolation']])
    if not rows:
        raise ValueError('No validation gaps with observed ground truth for this target.')
    truth, mean, sigma, baseline = pd.concat(rows).to_numpy().T
    paired = np.isfinite(mean) & np.isfinite(baseline)
    scores = {'Timestral (paired)': regression_scores(truth[paired], mean[paired], sigma[paired]),
              'Interpolation (paired)': regression_scores(truth[paired], baseline[paired])}
    return Validation(inventory, scores, tracks)