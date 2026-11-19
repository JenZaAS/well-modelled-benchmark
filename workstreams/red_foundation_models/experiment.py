"""Paired gap, facies, and demonstration ablations on held-out wells."""
import argparse
import json

import numpy as np

from benchmark.run import load_dataset
from wmb.curves import CORE
from wmb.io import Well
from wmb.scoring import regression_scores
from wmb.splits import gap_mask, well_split
from .contender import MODELS
from .depth_model import _segments


def masked_case(well, target, scenario, seed):
    truth = well.curves[target]
    mask = np.isfinite(truth)
    if scenario == "gap":
        mask &= gap_mask(well.depth, seed=seed)
    hidden = Well(well.name, well.depth, {**well.curves, target: np.where(mask, np.nan, truth)},
                  well.dataset, well.split, dict(well.meta))
    return hidden, mask


def depth_interpolation(well, target):
    estimate = np.full(len(well.depth), np.nan)
    for start, end in _segments(well.depth):
        z, y = well.depth[start:end], well.curves[target][start:end]
        known = np.isfinite(y)
        if known.any():
            estimate[start:end] = np.interp(z, z[known], y[known])
    return estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--target", choices=CORE, default="RHOB")
    parser.add_argument("--scenario", choices=["gap", "whole"], default="gap")
    parser.add_argument("--split", choices=["test", "validation"], default="test",
                        help="validation holds out 20%% of training wells; leaves test wells unused")
    parser.add_argument("--facies", action="store_true", help="use only independently available LITH")
    parser.add_argument("--shots", type=int, default=1, help="training demonstrations when no flank exists")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wells", type=int, help="limit eligible held-out wells for a quick experiment")
    args = parser.parse_args()
    if args.max_wells is not None and args.max_wells < 1:
        parser.error("--max-wells must be positive")
    train, test = load_dataset(args.dataset, args.seed)
    if {w.name for w in train} & {w.name for w in test}:
        raise ValueError("Train and evaluation well names overlap")
    if args.split == "validation":
        train, test = well_split(train, test_fraction=0.2, seed=args.seed)
    model = MODELS[args.model](facies=args.facies, shots=args.shots)
    model.fit(train, args.target)
    rows = []
    with_lith = 0
    for i, well in enumerate(sorted(test, key=lambda w: w.name)):
        if not well.has(args.target):
            continue
        hidden, mask = masked_case(well, args.target, args.scenario, args.seed + i)
        if not mask.any():
            continue
        pred = model.predict(hidden, args.target)
        baseline = depth_interpolation(hidden, args.target) if args.scenario == "gap" else np.full(len(mask), np.nan)
        rows.append((well.curves[args.target][mask], pred.mean[mask], pred.std[mask], baseline[mask]))
        with_lith += int("LITH" in well.curves and np.isfinite(well.curves["LITH"][mask]).any())
        if len(rows) == args.max_wells:
            break
    if not rows:
        raise ValueError("No held-out wells have scorable target samples in this scenario")
    y, mean, std, baseline = map(np.concatenate, zip(*rows))
    comparisons = {}
    if args.scenario == "gap":
        paired = np.isfinite(baseline) & np.isfinite(mean)
        comparisons["paired_interpolation"] = regression_scores(y[paired], baseline[paired])
        comparisons["paired_model"] = regression_scores(y[paired], mean[paired])
    print(json.dumps({**vars(args), "wells": len(rows), "wells_with_facies": with_lith,
                      "masked_samples": len(y), **regression_scores(y, mean, std), **comparisons}))


if __name__ == "__main__":
    main()
