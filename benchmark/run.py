"""Run contenders on a dataset, score them, write results/<contender>/<dataset>/<target>/.

Examples:
    python -m benchmark.run --list
    python -m benchmark.run --dataset synthetic --target RHOB
    python -m benchmark.run --dataset force2020 --target DTS --contenders black_gardner,blue_gbt
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

from benchmark.registry import REPO, discover  # also puts src/ on sys.path
from wmb import paths, scoring, splits, synthetic
from wmb.curves import CORE


def load_dataset(name: str, seed: int):
    if name == "synthetic":
        wells = synthetic.make_wells(seed=seed)
        return splits.well_split(wells, 0.25, seed)
    if name == "force2020":
        from wmb.io import read_force_csv
        root = paths.data_root() / "force2020"
        wells = read_force_csv(root / "train.csv", split="train")
        wells += read_force_csv(root / "leaderboard_test_features.csv", split="leaderboard")
        wells += read_force_csv(root / "hidden_test.csv", split="hidden")
        return splits.official_split(wells)
    # anything else: a folder under the data root with a wells.json manifest
    from wmb.io import iter_manifest_wells
    wells = list(iter_manifest_wells(paths.data_root() / name))
    return splits.well_split(wells, 0.25, seed)


def run_one(factory, train, test, target, dataset, out_root):
    model = factory()
    if hasattr(model, "supports") and not model.supports(target):
        return None
    t: dict = {}
    with scoring.timer(t, "fit_s"):
        model.fit(train, target)
    ys, ms, ss, n_pred = [], [], [], 0
    with scoring.timer(t, "predict_s"):
        for w in test:
            if not w.has(target):
                continue
            p = model.predict(w.without(target), target)
            ys.append(w.curves[target]); ms.append(p.mean)
            ss.append(p.std if p.std is not None else np.full_like(p.mean, np.nan))
            n_pred += len(p.mean)
    if not ys:
        return None
    y, m, s = np.concatenate(ys), np.concatenate(ms), np.concatenate(ss)
    sc = scoring.regression_scores(y, m, None if np.isnan(s).all() else s)
    sc.update({k: round(v, 4) for k, v in t.items()})
    sc["predict_s_per_1000"] = round(t["predict_s"] / max(n_pred, 1) * 1000, 6)
    sc.update({"contender": model.name, "workstream": model.workstream, "dataset": dataset, "target": target,
               "n_train_wells": len(train), "n_test_wells": len(test),
               "machine": f"{platform.processor() or platform.machine()} / {platform.system()}",
               "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    out = out_root / model.name / dataset / target
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(sc, indent=2))
    return sc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="well-modelled-benchmark runner")
    ap.add_argument("--dataset", default="synthetic", help="synthetic | force2020 | <folder under FH2026_DATA with wells.json>")
    ap.add_argument("--target", default="RHOB", choices=CORE, help="curve to hide and predict")
    ap.add_argument("--contenders", default="all", help="comma-separated names, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="list contenders and exit")
    a = ap.parse_args(argv)

    if a.list:
        print("Contenders:"); discover(verbose=True); return 0
    found = discover()
    names = list(found) if a.contenders == "all" else a.contenders.split(",")
    train, test = load_dataset(a.dataset, a.seed)
    print(f"dataset={a.dataset} target={a.target} train_wells={len(train)} test_wells={len(test)}")
    out_root = paths.results_root()
    rows = []
    for n in names:
        if n not in found:
            print(f"unknown contender: {n}", file=sys.stderr); continue
        sc = run_one(found[n], train, test, a.target, a.dataset, out_root)
        if sc:
            rows.append(sc)
            print(f"  {n:24s} MAE={sc.get('MAE', float('nan')):.4f} R2={sc.get('R2', float('nan')):.3f} "
                  f"cover1σ={sc.get('cover_1sigma', float('nan')):.2f} fit={sc['fit_s']:.2f}s pred={sc['predict_s']:.2f}s")
        else:
            print(f"  {n:24s} skipped (target not supported or no test wells with {a.target})")
    print(f"scores written under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
