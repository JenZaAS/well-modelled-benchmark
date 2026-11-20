"""Train the models that predict new wells (all 118 FORCE 2020 wells) and save them.

    python -m workstreams.blue_boosted_trees.ml_study.train_models --models gbt,linear --features base,ctx,trend

Unlike ``run_study`` (98 training wells, 20 held-out wells, for honest scores), these models use every well, because
nothing is left to evaluate. Models already saved are skipped, so an interrupted run can be restarted
(``--overwrite`` to refit). Output: ``<work>/blue_ml_study/models/`` (see ``store.py``), never in git.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from wmb import scoring
from wmb.curves import CORE

from .data import load_official_split
from .models import MODEL_KINDS, MLContender, variant_name
from .run_study import FEATURE_SETS, MIN_TRAIN_WELLS
from .store import default_model_dir, model_path, save_model, write_manifest

log = logging.getLogger("ml_study")
RANGE_PERCENTILES = (0.5, 99.5)


def training_ranges(wells) -> dict[str, list[float]]:
    """Usual value range (0.5th to 99.5th percentile) of every core log, used to flag wrong units in uploaded wells."""
    out = {}
    for c in CORE:
        v = np.concatenate([w.curves[c] for w in wells if c in w.curves])
        v = v[np.isfinite(v)]
        if v.size:
            out[c] = [float(x) for x in np.percentile(v, RANGE_PERCENTILES)]
    return out


def strat_labels(wells) -> dict[str, list[str]]:
    """All GROUP and FORMATION labels of the training wells (UNKNOWN excluded)."""
    from .data import UNKNOWN
    return {level: sorted({str(x) for w in wells for x in np.unique(w.meta[level])} - {UNKNOWN})
            for level in ("group", "formation")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=",".join(CORE), help="comma-separated curves to predict (default: all 9)")
    ap.add_argument("--models", default="gbt,linear", help="comma-separated: gbt,linear,rf (default: gbt,linear)")
    ap.add_argument("--features", default="base,ctx,trend", help="comma-separated feature sets (default: all)")
    ap.add_argument("--max-train-rows", type=int, default=250_000, help="thin training rows per fit (0 = use all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="refit models that are already saved")
    ap.add_argument("--out", type=Path, default=None, help="model folder (default: <work>/blue_ml_study/models)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    targets, kinds, feature_names = a.targets.split(","), a.models.split(","), a.features.split(",")
    unknown = ([t for t in targets if t not in CORE] + [k for k in kinds if k not in MODEL_KINDS]
               + [f for f in feature_names if f not in FEATURE_SETS])
    if unknown:
        ap.error(f"unknown target/model/feature set: {unknown}")
    out = a.out or default_model_dir()

    train, test = load_official_split()
    wells = train + test
    log.info("training on %d wells; saving to %s", len(wells), out)
    write_manifest(out, {"trained_on": f"all {len(wells)} FORCE 2020 wells", "n_wells": len(wells),
                         "max_train_rows": a.max_train_rows, "seed": a.seed, "ranges": training_ranges(wells),
                         "labels": strat_labels(wells)})
    for target in targets:
        if sum(w.has(target) for w in wells) < MIN_TRAIN_WELLS:
            log.warning("skipping %s: fewer than %d wells have it", target, MIN_TRAIN_WELLS)
            continue
        for kind in kinds:
            for name in feature_names:
                context, trend = FEATURE_SETS[name]
                for use_strat in (False, True):
                    variant = variant_name(kind, use_strat, context, trend)
                    if model_path(out, target, variant).exists() and not a.overwrite:
                        continue
                    model = MLContender(kind, use_strat, a.max_train_rows, a.seed, context, trend)
                    t: dict[str, float] = {}
                    with scoring.timer(t, "fit_s"):
                        model.fit(wells, target)
                    path = save_model(out, model, target)
                    log.info("%-5s %-16s fit %3.0fs  %5.1f MB  %s", target, variant, t["fit_s"],
                             path.stat().st_size / 2**20, time.strftime("%H:%M:%S"))
    log.info("done: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
