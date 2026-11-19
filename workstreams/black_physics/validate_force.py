"""Aggregate-only FORCE 2020 validation of the black physics baselines on the station's own tasks.

Run from the repository root (after `python -m benchmark.make_tasks --dataset force2020`):

    python -m workstreams.black_physics.validate_force
    python -m workstreams.black_physics.validate_force --compare template_linear --check-results --json out.json

Uses the committed task spec and windows, the official 98-well fit and the 20 leaderboard + hidden wells.
For every (model, target, scenario) it reports the station's pooled scores (benchmark.run.score_tasks),
**fill fraction** = finite predictions on masked, measured target depths / all masked, measured target
depths, wells with zero fill, and the mean of per-well MAEs. Head-to-head tables restrict every model to
the common support (depths where all compared models predict) and compare intervals only where every
interval-giving model has one. Gap scenarios add a straight-line interpolation reference between the measured
flanks in the same continuous depth segment. `--check-results` verifies the runner's scores.json /
failure.json files and that they match this recomputation. Prints and writes aggregates only, never well data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

from benchmark import make_tasks as mt
from benchmark.registry import REPO, discover
from benchmark.run import score_tasks
from wmb import paths, scoring, splits
from wmb.contender import Prediction
from wmb.contract import SCENARIO_GAP, TARGET_CURVES, validate_prediction
from workstreams.black_physics.contender import CONTENDERS, GreenbergCastagnaBrineGR, VpVsRatio
from workstreams.black_physics.models._shared import calibration as cal
from workstreams.black_physics.models._shared.curves import curve, has_positive_target
from workstreams.black_physics.models.castagna_mudrock.equation import mudrock_dts
from workstreams.black_physics.models.greenberg_castagna.equation import gc_brine_gr_dts
from workstreams.black_physics.models.vpvs_ratio.equation import ratio_dts

DATASET = "force2020"
SCENARIOS = ["missing_curve", "gap_10", "gap_50", "gap_100"]
INTERP = "ref_linear_interp"


def linear_gap_reference(task) -> Prediction:
    """Gap-only reference: straight line in depth between the nearest measured target values above and below
    each hidden run, both inside the same continuous depth segment; NaN where a flank is unavailable."""
    n = len(task.depth)
    out = np.full(n, np.nan)
    if task.scenario != SCENARIO_GAP or task.target_observed is None:
        return Prediction(out)
    depth = np.asarray(task.depth, float)
    obs = np.asarray(task.target_observed, float)
    segment = np.r_[0, np.cumsum(np.diff(depth) > splits.max_gap_step(depth))]
    measured = np.isfinite(obs)
    m = np.asarray(task.predict_mask, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        for seg in np.unique(segment[i0:i1]):
            rows = np.arange(i0, i1)[segment[i0:i1] == seg]
            above = np.flatnonzero(measured[:rows[0]] & (segment[:rows[0]] == seg))
            below = np.flatnonzero(measured[rows[-1] + 1:] & (segment[rows[-1] + 1:] == seg)) + rows[-1] + 1
            if not len(above) or not len(below):
                continue
            j, k = above[-1], below[0]
            out[rows] = obs[j] + (obs[k] - obs[j]) * (depth[rows] - depth[j]) / (depth[k] - depth[j])
    return Prediction(out)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001  (provenance only)
        return "unknown"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rows(tasks, preds, truth):
    """Per task: (well id, measured mask on predict_mask, truth values, p50, std) restricted to predict_mask."""
    out = []
    for t, p in zip(tasks, preds):
        m = t.predict_mask
        y = np.asarray(truth[(t.well_id, t.target)], float)[m]
        std = None if p.std is None else np.asarray(p.std, float)[m]
        out.append((t.well_id, np.isfinite(y), y, np.asarray(p.p50, float)[m], std))
    return out


def fill_and_wells(rows) -> dict:
    measured = sum(int(meas.sum()) for _, meas, *_ in rows)
    filled = sum(int((meas & np.isfinite(p)).sum()) for _, meas, _, p, _ in rows)
    wells = {w for w, meas, *_ in rows if meas.any()}
    filled_wells = {w for w, meas, _, p, _ in rows if (meas & np.isfinite(p)).any()}
    per_well = [float(np.abs(p[ok] - y[ok]).mean()) for _, meas, y, p, _ in rows
                for ok in [meas & np.isfinite(p)] if ok.any()]
    return {"measured_rows": measured, "filled_rows": filled, "fill_fraction": filled / measured if measured else None,
            "wells": len(wells), "wells_zero_fill": len(wells - filled_wells),
            "mean_per_well_MAE": float(np.mean(per_well)) if per_well else None}


def head_to_head(rows_by_model: dict) -> dict:
    """Scores of every model on the depths where all of them predict, plus intervals where all interval models give one."""
    names = list(rows_by_model)
    first = rows_by_model[names[0]]
    common = [meas & np.logical_and.reduce([np.isfinite(rows_by_model[n][i][3]) for n in names])
              for i, (_, meas, *_) in enumerate(first)]
    n_rows = int(sum(c.sum() for c in common))
    n_wells = len({first[i][0] for i, c in enumerate(common) if c.any()})
    with_std = [n for n in names if any(r[4] is not None for r in rows_by_model[n])]
    both = [c & np.logical_and.reduce([np.isfinite(rows_by_model[n][i][4]) if rows_by_model[n][i][4] is not None
                                       else np.zeros_like(c) for n in with_std]) if with_std else np.zeros_like(c)
            for i, c in enumerate(common)]
    out = {"common_rows": n_rows, "common_wells": n_wells, "interval_rows": int(sum(b.sum() for b in both)),
           "interval_models": with_std, "models": {}}
    for n in names:
        rr = rows_by_model[n]
        y = np.concatenate([r[2][c] for r, c in zip(rr, common)])
        p = np.concatenate([r[3][c] for r, c in zip(rr, common)])
        s = scoring.regression_scores(y, p)
        wells = [float(np.abs(r[3][c] - r[2][c]).mean()) for r, c in zip(rr, common) if c.any()]
        entry = {"MAE": s.get("MAE"), "RMSE": s.get("RMSE"), "mean_per_well_MAE": float(np.mean(wells)) if wells else None}
        if n in with_std and out["interval_rows"]:
            yb = np.concatenate([r[2][b] for r, b in zip(rr, both)])
            pb = np.concatenate([r[3][b] for r, b in zip(rr, both)])
            sb = np.concatenate([r[4][b] for r, b in zip(rr, both)])
            si = scoring.regression_scores(yb, pb, sb)
            entry.update(cover_90=si.get("cover_90"), interval_width_90=si.get("interval_width_90"))
        out["models"][n] = entry
    return out


def check_results(computed: dict, ids: list[str], targets: dict) -> tuple[list[str], dict]:
    """Check the runner output for the black IDs: every per-scenario scores.json present, no failure.json, scores
    equal to this recomputation, and the latest run that covered all IDs has every combination without error.
    Returns (messages, runner timings by (id, target, scenario))."""
    root = paths.results_root()
    problems, ok, timings = [], 0, {}
    for name in ids:
        for sc in SCENARIOS:
            d = root / name / DATASET / targets[name] / sc
            if (d / "failure.json").is_file():
                problems.append(f"{name} {targets[name]} {sc}: failure.json present")
            if not (d / "scores.json").is_file():
                problems.append(f"{name} {targets[name]} {sc}: scores.json missing")
                continue
            got = json.loads((d / "scores.json").read_text())
            timings[(name, targets[name], sc)] = {k: got.get(k) for k in ("fit_s", "predict_s", "predict_s_batch",
                                                                          "samples_per_s", "n_samples")}
            mine = computed[(name, targets[name], sc)]["station"]
            if got.get("n") != mine.get("n") or not np.isclose(got.get("MAE", np.nan), mine.get("MAE", np.nan),
                                                                 rtol=1e-9, atol=1e-12):
                problems.append(f"{name} {targets[name]} {sc}: runner n/MAE {got.get('n')}/{got.get('MAE')} "
                                f"!= recomputed {mine.get('n')}/{mine.get('MAE')}")
            ok += 1
    runs = [p for p in sorted((root / "runs").glob(f"*_{DATASET}.json"))
            if set(ids) <= {r["contender"] for r in json.loads(p.read_text())}]
    rows = json.loads(runs[-1].read_text()) if runs else []
    combos = {(r["contender"], r["target"], r["scenario"]) for r in rows if r["contender"] in ids and "error" not in r}
    problems += [f"run failure: {r['contender']} {r['target']} {r['scenario']}: {r['error']}"
                 for r in rows if r["contender"] in ids and "error" in r]
    if len(combos) != len(ids) * len(SCENARIOS):
        problems.append(f"latest full run {runs[-1].name if runs else '-'} has {len(combos)} successful black "
                        f"combinations, expected {len(ids) * len(SCENARIOS)}")
    msgs = problems or [f"OK: {ok} per-scenario scores.json files match this recomputation, no failure.json; latest run "
                        f"with all {len(ids)} IDs ({runs[-1].name}) has {len(combos)} successful unique combinations"]
    return msgs, timings


def training_cv(train) -> dict:
    """Training-split DTS reference from the implementation plan (debugging only, not a benchmark score).

    The DTS-bearing training wells, sorted by name, permuted with default_rng(2026) and split into 5 folds.
    Ratio and GR endpoints are fitted on the other four folds (endpoints from each fitting well's entire finite
    GR). Scored on positive finite measured DTS where all three predictions are finite (this includes the
    mudrock velocity floor); each well has equal weight.
    """
    wells = sorted((w for w in train if has_positive_target(w, "DTS")), key=lambda w: w.name)
    folds = cal.whole_well_folds([w.name for w in wells])
    maes: dict[str, list[float]] = {"black_castagna_dts": [], "black_vpvs_ratio": [], "black_gc_brine_gr": []}
    rows = 0
    for fold in folds:
        held = set(int(i) for i in fold)
        fit_wells = [w for i, w in enumerate(wells) if i not in held]
        ratio, ends = VpVsRatio().fit_params(fit_wells), GreenbergCastagnaBrineGR().fit_params(fit_wells)
        for i in sorted(held):
            w = wells[i]
            dtc, y = curve(w, "DTC"), curve(w, "DTS")
            preds = {"black_castagna_dts": mudrock_dts(dtc), "black_vpvs_ratio": ratio_dts(dtc, ratio.ratio),
                     "black_gc_brine_gr": gc_brine_gr_dts(dtc, curve(w, "GR"), ends.gr_low, ends.gr_high)}
            ok = np.isfinite(y) & (y > 0) & np.logical_and.reduce([np.isfinite(p) for p in preds.values()])
            if ok.any():
                rows += int(ok.sum())
                for k, p in preds.items():
                    maes[k].append(float(np.abs(p[ok] - y[ok]).mean()))
    wins = int(np.sum(np.less(maes["black_gc_brine_gr"], maes["black_castagna_dts"])))
    return {"wells": len(maes["black_gc_brine_gr"]), "rows": rows, "folds": len(folds),
            "mean_per_well_MAE": {k: float(np.mean(v)) for k, v in maes.items()},
            "gc_beats_mudrock_wells": wins}


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compare", default="", help="comma-separated registered models to add to the head-to-head")
    ap.add_argument("--check-results", action="store_true", help="verify the runner's FORCE output for the black IDs")
    ap.add_argument("--json", help="also write the aggregates to this file")
    a = ap.parse_args(argv)

    spec = mt.load_spec(DATASET)
    train, test = mt.load_wells(DATASET, spec)
    tasks = mt.load_tasks(DATASET, wells=(train, test))
    index = json.loads((mt.cache_root() / DATASET / "index.json").read_text(encoding="utf-8"))
    truth = {(w.name, t): w.curves[t] for w in test for t in TARGET_CURVES if t in w.curves}
    registry = discover()
    factories = {f().name: f for f in CONTENDERS}
    factories.update({n: registry[n] for n in a.compare.split(",") if n})
    ids = [f().name for f in CONTENDERS]
    targets = {n: f().targets[0] for n, f in factories.items() if n in ids}

    report = {"commit": _git_head(), "dataset": DATASET, "spec_hash": index.get("spec_hash"),
              "windows_sha256_16": _sha(mt.spec_path(DATASET).with_suffix(".windows.yaml")),
              "n_train_wells": len(train), "n_test_wells": len(test), "n_tasks": len(tasks),
              "models": {}, "scores": {}, "head_to_head": {}}
    print(f"commit {report['commit']}; spec_hash {report['spec_hash']}; windows sha256[:16] "
          f"{report['windows_sha256_16']}; {len(train)} fit wells; {len(test)} test wells; {len(tasks)} tasks\n")

    computed = {}
    for target in TARGET_CURVES:
        models = {}
        for name, f in factories.items():
            m = f()
            if not m.supports(target):
                continue
            m.fit(train, target)
            models[name] = m
            if name in ids:
                report["models"][name] = m.describe()
        for sc in SCENARIOS:
            ts = [t for t in tasks if t.target == target and t.scenario_key == sc]
            if not ts:
                continue
            rows_by_model = {}
            preds_by_model = {name: [validate_prediction(t, m.predict_task(t)) for t in ts] for name, m in models.items()}
            if sc != "missing_curve":
                preds_by_model[INTERP] = [validate_prediction(t, linear_gap_reference(t)) for t in ts]
            for name, preds in preds_by_model.items():
                station = score_tasks(ts, preds, truth)
                rows = _rows(ts, preds, truth)
                rows_by_model[name] = rows
                entry = {"station": station, **fill_and_wells(rows), "n_tasks": len(ts)}
                computed[(name, target, sc)] = entry
                report["scores"][f"{name}|{target}|{sc}"] = entry
            black = [n for n in rows_by_model if n in ids]
            sets = {}
            for label, names in (("physics", black), ("all", list(rows_by_model))):
                if len(names) >= 2 and (label == "physics" or len(names) > len(black)):
                    sets[label] = head_to_head({n: rows_by_model[n] for n in names})
            if "physics" in sets and "all" in sets and sets["physics"]["common_rows"] == sets["all"]["common_rows"]:
                del sets["physics"]                 # same depths: the larger table already shows these numbers
            for label, h in sets.items():
                report["head_to_head"][f"{target}|{sc}|{label}"] = h

    print("## Per model: station scores and fill\n")
    print("| target | scenario | model | n | fill | wells (zero fill) | MAE | RMSE | mean per-well MAE | cover_90 | width_90 |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (name, target, sc), e in computed.items():
        s = e["station"]
        print(f"| {target} | {sc} | {name} | {s['n']} | {fmt(e['fill_fraction'], 3)} | {e['wells']} ({e['wells_zero_fill']}) "
              f"| {fmt(s.get('MAE'))} | {fmt(s.get('RMSE'))} | {fmt(e['mean_per_well_MAE'])} "
              f"| {fmt(s.get('cover_90'), 3)} | {fmt(s.get('interval_width_90'))} |")

    print("\n## Head-to-head on common support\n")
    print("| target | scenario | set | rows (wells) | model | MAE | mean per-well MAE | interval rows | cover_90 | width_90 |")
    print("|---|---|---|---:|---|---:|---:|---:|---:|---:|")
    for key, h in report["head_to_head"].items():
        target, sc, label = key.split("|")
        for name, e in h["models"].items():
            print(f"| {target} | {sc} | {label} | {h['common_rows']} ({h['common_wells']}) | {name} | {fmt(e['MAE'])} "
                  f"| {fmt(e['mean_per_well_MAE'])} | {h['interval_rows']} | {fmt(e.get('cover_90'), 3)} "
                  f"| {fmt(e.get('interval_width_90'))} |")

    print("\n## Fitted parameters and uncertainty scales\n")
    for name, d in report["models"].items():
        print(f"- `{name}`: params {json.dumps(d['params'], default=str)}; calibration {json.dumps(d['calibration'])}")

    cv = report["training_cv"] = training_cv(train)
    print(f"\n## Training-split DTS reference (plan pilot procedure)\n\n{cv['wells']} DTS wells, {cv['folds']} folds, "
          f"{cv['rows']} common depths; black_gc_brine_gr beats black_castagna_dts in {cv['gc_beats_mudrock_wells']} "
          f"of {cv['wells']} wells")
    for k, v in cv["mean_per_well_MAE"].items():
        print(f"- {k}: mean per-well MAE {v:.2f} us/ft")

    if a.check_results:
        print("\n## Runner output check\n")
        msgs, timings = check_results(computed, ids, targets)
        for line in msgs:
            print(f"- {line}")
        print("\n| model | target | scenario | fit_s | predict_s | predict_s_batch | samples_per_s |")
        print("|---|---|---|---:|---:|---:|---:|")
        for (name, target, sc), t in timings.items():
            print(f"| {name} | {target} | {sc} | {fmt(t['fit_s'], 3)} | {fmt(t['predict_s'], 3)} "
                  f"| {fmt(t['predict_s_batch'], 3)} | {fmt(t['samples_per_s'], 0)} |")
        report["runner_check"] = {"messages": msgs, "timings": {"|".join(k): v for k, v in timings.items()}}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
