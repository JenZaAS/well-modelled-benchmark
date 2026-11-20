"""Run the ML study: 3 models x feature sets x with/without stratigraphy x every core log, on the official FORCE 2020
split.

Feature sets: base (the other logs + depth), ctx (+ logs 5 m above/below and moving averages), trend (+ 10 m mean
above/below, their difference and local spread).

    python -m workstreams.blue_boosted_trees.ml_study.run_study                      # base feature set only
    python -m workstreams.blue_boosted_trees.ml_study.run_study --targets RHOB,DTC --models gbt,linear
    python -m workstreams.blue_boosted_trees.ml_study.run_study --features trend --models gbt,linear   # add variants

Writes to ``<results>/blue_ml_study/`` (gitignored). Results are merged per (log, variant), so a partial run
keeps everything it did not recompute:
    metrics.csv                one row per (target, variant); resistivity scored in log10
    predictions/<TARGET>.parquet   WELL, DEPTH_MD and one prediction column per variant (all 20 test wells)
    run_info.json              settings of the latest run and machine
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wmb import paths, scoring
from wmb.curves import CORE

from .data import load_official_split
from .metrics import evaluate, target_space
from .models import MODEL_KINDS, MLContender, variant_name

log = logging.getLogger("ml_study")
MIN_TRAIN_WELLS = 3
OUT_NAME = "blue_ml_study"
FEATURE_SETS = {"base": (False, False), "ctx": (True, False), "trend": (False, True)}   # (context, trend)


def default_out_dir() -> Path:
    """Where results go unless ``--out`` is given."""
    return paths.results_root() / OUT_NAME


def run_target(target: str, train, test, kinds: tuple[str, ...], feature_sets: tuple[tuple[bool, bool], ...],
               max_train_rows: int, seed: int) -> tuple[list[dict], pd.DataFrame]:
    """Fit and score every model variant for one target. Returns metric rows and the predictions table."""
    preds = pd.DataFrame({"WELL": np.concatenate([[w.name] * len(w.depth) for w in test]),
                          "DEPTH_MD": np.concatenate([w.depth for w in test])})
    truth = np.concatenate([w.curves.get(target, np.full(len(w.depth), np.nan)) for w in test])
    rows = []
    for kind in kinds:
        for context, trend in feature_sets:
            for use_strat in (False, True):
                model = MLContender(kind, use_strat, max_train_rows, seed, context, trend)
                timing: dict[str, float] = {}
                with scoring.timer(timing, "fit_s"):
                    model.fit(train, target)
                with scoring.timer(timing, "predict_s"):
                    pred = np.concatenate([model.predict(w.without(target), target).mean for w in test])
                variant = variant_name(kind, use_strat, context, trend)
                preds[variant] = pred.astype(np.float32)
                row = {"target": target, "model": kind, "context": context, "trend": trend,
                       "stratigraphy": use_strat, "variant": variant,
                       **evaluate(target_space(target, truth), target_space(target, pred)),
                       "n_features": len(model.features.names),
                       "n_train_wells": sum(w.has(target) for w in train),
                       "n_train_rows": model.n_train_rows, **{k: round(v, 2) for k, v in timing.items()}}
                rows.append(row)
                log.info("%-6s %-14s MAE=%.4f R2=%.3f bias=%+.4f fit=%.0fs", target, variant, row["MAE"], row["R2"],
                         row["bias"], timing["fit_s"])
    return rows, preds


def merge_metrics(old: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """Replace the old rows of every (target, variant) that was just re-run; keep all other rows."""
    new = pd.DataFrame(new_rows)
    if old.empty:
        return new
    redone = set(zip(new["target"], new["variant"]))
    keep = old[[(t, v) not in redone for t, v in zip(old["target"], old["variant"])]]
    merged = pd.concat([keep, new], ignore_index=True)
    for flag in ("context", "trend"):   # results from before these feature sets existed have no flag
        merged[flag] = merged[flag].fillna(False).astype(bool)
    return merged


def save_predictions(file: Path, preds: pd.DataFrame) -> None:
    """Write the predictions of one target, keeping the variant columns of earlier runs."""
    if file.exists():
        old = pd.read_parquet(file)
        if old[["WELL", "DEPTH_MD"]].equals(preds[["WELL", "DEPTH_MD"]]):
            preds = pd.concat([preds, old[[c for c in old.columns if c not in preds.columns]]], axis=1)
        else:
            log.warning("%s does not line up with the new predictions; replacing it", file.name)
    preds.to_parquet(file, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=",".join(CORE), help="comma-separated curves to predict (default: all 9)")
    ap.add_argument("--models", default=",".join(MODEL_KINDS), help="comma-separated: gbt,linear,rf")
    ap.add_argument("--features", default="base", help=f"comma-separated feature sets: {','.join(FEATURE_SETS)} "
                    "(base = other logs + depth; ctx = + context; trend = + trend). Default: base")
    ap.add_argument("--max-train-rows", type=int, default=250_000, help="thin training rows per fit (0 = use all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <results>/blue_ml_study)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    targets, kinds, feature_names = a.targets.split(","), tuple(a.models.split(",")), a.features.split(",")
    unknown = ([t for t in targets if t not in CORE] + [k for k in kinds if k not in MODEL_KINDS]
               + [f for f in feature_names if f not in FEATURE_SETS])
    if unknown:
        ap.error(f"unknown target/model/feature set: {unknown}")
    feature_sets = tuple(FEATURE_SETS[f] for f in feature_names)
    out = a.out or default_out_dir()
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    train, test = load_official_split()
    log.info("train wells=%d test wells=%d", len(train), len(test))
    (out / "run_info.json").write_text(json.dumps({
        "targets": targets, "models": kinds, "features": feature_names, "max_train_rows": a.max_train_rows,
        "seed": a.seed, "n_train_wells": len(train), "n_test_wells": len(test),
        "test_wells": [w.name for w in test],
        "machine": f"{platform.processor() or platform.machine()} / {platform.system()}",
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))

    metrics_file = out / "metrics.csv"
    table = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()
    for target in targets:
        if sum(w.has(target) for w in train) < MIN_TRAIN_WELLS:
            log.warning("skipping %s: fewer than %d training wells have it", target, MIN_TRAIN_WELLS)
            continue
        rows, preds = run_target(target, train, test, kinds, feature_sets, a.max_train_rows, a.seed)
        save_predictions(out / "predictions" / f"{target}.parquet", preds)
        table = merge_metrics(table, rows)
        table.to_csv(metrics_file, index=False)
    log.info("done: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
