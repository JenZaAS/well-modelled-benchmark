"""A trust rule for any registered model: the decision layer consumes keyed predictions ("bring your own model").

The brief's decision layer was fitted on one reconstruction (the Granite row picked on B) and read Granite
internals. This module needs only what any station contender returns: the filled curve and, if the model
reports one, an interval. Each model then gets its own rule, fitted and checked the same way:

    1. predict (in the model's own environment; writes keyed fills for the C, D and test gaps)
       <python> -m workstreams.black_physics.granite.decide_any predict --model red_timestral
    2. decide (NumPy only; any environment)
       python -m workstreams.black_physics.granite.decide_any decide

Same gaps, label and policy as the brief: one example is one whole 50 m gap; label 1 when the gap MAE is at most
0.05 g/cm3; L2 logistic rules fitted on C with equal weight per well; intercept/slope recalibration on D; accept
when p_pass >= 0.80, otherwise review. Features, all computable without the hidden values:

    flank_var          mean |value - side median| over up to 512 measured samples on each side (variability)
    flank_median_diff  |median above - median below|: a level change across the gap (a boundary inside it)
    pred_range         max - min of the fill inside the gap
    ctx_above_m/below  metres of contiguous measured density on each side
    band_width         mean width of the model's own interval inside the gap (only if the model reports one)
    gardner_mad        median |fill - fixed Gardner from sonic| where DTC exists   (Gardner and logs variants)
    dtc_fraction       share of gap depths with usable DTC                         (Gardner and logs variants)
    gr_shift           log(1 + |median GR inside the gap - median GR on the flanks| in API)   (logs variant)
    gr_iqr_gap         log(1 + interquartile range of GR inside the gap in API)                (logs variant)

The other logs are measured inside the gap, so GR can flag a lithology change that a density-only fill
cannot see. It is a feature of the trust rule, not an input of the fill.

`decide` also tries per-gap routing: for each gap, use the fill of the model whose own rule gives the highest
p_pass (no truth used), and compare the routed mean per-well MAE with each model alone, on D and on test.

Outputs (local, never committed): <work_root>/black_granite/decision_any/preds_<model>.npz, <model>.json (rules,
metrics on C/D/test, and a keyed test sidecar), summary.json.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from workstreams.black_physics.granite import config as C
from workstreams.black_physics.granite import decision as DEC
from workstreams.black_physics.granite import gaps as G

GROUPS = ("C", "D", "test")
CTX = 512
BASE = ["flank_var", "flank_median_diff", "pred_range", "ctx_above_m", "ctx_below_m"]
GARDNER = ["gardner_mad", "dtc_fraction"]
LOGS = ["gr_shift", "gr_iqr_gap"]
METHODS = ("prevalence", "width_only", "numeric", "numeric_gardner", "numeric_logs", "numeric_cal", "numeric_gardner_cal",
           "numeric_logs_cal")


def out_dir():
    p = C.work_dir() / "decision_any"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tasks_by_group() -> tuple[dict, dict]:
    """Development gaps (C, D: prepared, non-overlapping) and the station's sealed test gaps, keyed by gap id."""
    from benchmark import test_def as td
    from workstreams.black_physics.granite import pipeline as P

    tr, te = P.official()
    out = {"C": P.dev_tasks("C"), "D": P.dev_tasks("D")}
    tasks = td.build_def_tasks(td.load_def(C.TEST_NAME), tr, te)
    for t in tasks:
        idx = np.flatnonzero(t.predict_mask)
        t.meta["gap_id"] = f"{t.well_id}|{idx[0]}:{idx[-1] + 1}"
    out["test"] = tasks
    wells = {**{w.name: w for w in tr}, **{w.name: w for w in te}}
    return out, wells


def _band(p) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """The model's own interval: P10/P90 if given, else P05/P95, else none."""
    q = p.quantiles or {}
    for lo, hi in ((0.1, 0.9), (0.05, 0.95)):
        if lo in q and hi in q and q[lo] is not None and q[hi] is not None:
            return np.asarray(q[lo], float), np.asarray(q[hi], float), f"P{round(lo * 100):02d}-P{round(hi * 100)}"
    if p.p05 is not None and p.p95 is not None:
        return np.asarray(p.p05, float), np.asarray(p.p95, float), "P05-P95"
    return None, None, None


# ------------------------------------------------------------------------------------------------ step 1
def predict(model: str) -> dict:
    from benchmark.registry import discover
    from workstreams.black_physics.granite import pipeline as P

    found = discover()
    if model not in found:
        raise SystemExit(f"{model} is not registered in this environment (check the opt-in flags)")
    groups, _ = tasks_by_group()
    tr, _ = P.official()
    m = found[model]()
    t0 = time.perf_counter()
    m.fit(tr, C.TARGET)
    fit_s = time.perf_counter() - t0
    keyed, band, n = {}, None, 0
    t0 = time.perf_counter()
    for g, tasks in groups.items():
        for t, p in zip(tasks, m.predict_batch(tasks)):
            mask = t.predict_mask
            keyed[f"{g}|{t.meta['gap_id']}|p50"] = np.asarray(p.mean, float)[mask]
            lo, hi, name = _band(p)
            if lo is not None:
                keyed[f"{g}|{t.meta['gap_id']}|lo"], keyed[f"{g}|{t.meta['gap_id']}|hi"] = lo[mask], hi[mask]
                band = band or name
            n += 1
    meta = {"model": model, "band": band, "fit_s": round(fit_s, 2), "predict_s": round(time.perf_counter() - t0, 2),
            "n_gaps": n, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "provenance": getattr(m, "provenance", None)}
    np.savez_compressed(out_dir() / f"preds_{model}.npz", **keyed)
    (out_dir() / f"preds_{model}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"decide_any predict {model}: {n} gaps, band {band}, fit {fit_s:.1f} s, predict {meta['predict_s']} s")
    return meta


# ------------------------------------------------------------------------------------------------ step 2
def features(task, p50_gap, lo_gap, hi_gap, gardner_gap) -> dict:
    from workstreams.black_physics.granite.reconstruct import _span_m

    depth = np.asarray(task.depth, float)
    obs = np.asarray(task.target_observed, float)
    a, b, s, e = G.mask_pieces(depth, task.predict_mask)[0]
    up = G.flank_run(obs, a, b, s, e, CTX, "above")
    dn = G.flank_run(obs, a, b, s, e, CTX, "below")
    med = {k: float(np.median(obs[i])) if len(i) else math.nan for k, i in (("up", up), ("dn", dn))}
    dev = [np.abs(obs[i] - np.median(obs[i])) for i in (up, dn) if len(i)]
    p = np.asarray(p50_gap, float)
    fin = np.isfinite(p)
    g = np.asarray(gardner_gap, float)
    both = fin & np.isfinite(g)
    diff = abs(med["up"] - med["dn"])
    gr = np.asarray(task.inputs["GR"], float) if "GR" in task.inputs else np.full(len(depth), np.nan)
    gr_in = gr[task.predict_mask]
    gr_in = gr_in[np.isfinite(gr_in)]
    gr_fl = gr[np.r_[up, dn].astype(int)] if len(up) + len(dn) else np.array([])
    gr_fl = gr_fl[np.isfinite(gr_fl)]
    return {
        "flank_var": float(np.concatenate(dev).mean()) if dev else math.nan,
        "flank_median_diff": float(diff) if np.isfinite(diff) else math.nan,
        "pred_range": float(np.ptp(p[fin])) if fin.any() else math.nan,
        "ctx_above_m": _span_m(depth, up) if len(up) else 0.0,
        "ctx_below_m": _span_m(depth, dn) if len(dn) else 0.0,
        "band_width": float(np.nanmean(hi_gap - lo_gap)) if lo_gap is not None and np.isfinite(hi_gap - lo_gap).any()
        else math.nan,
        "gardner_mad": float(np.median(np.abs(p[both] - g[both]))) if both.any() else math.nan,
        "dtc_fraction": float(np.isfinite(g).mean()) if len(g) else 0.0,
        "gr_shift": float(np.log1p(abs(np.median(gr_in) - np.median(gr_fl)))) if len(gr_in) and len(gr_fl) else math.nan,
        "gr_iqr_gap": float(np.log1p(np.subtract(*np.percentile(gr_in, [75, 25])))) if len(gr_in) > 3 else math.nan,
    }


def auc(p, y) -> float | None:
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok].astype(int)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def records(model: str, groups: dict, wells: dict, gardner: dict) -> tuple[dict, dict]:
    from workstreams.black_physics.granite.pipeline import gap_rows

    z = dict(np.load(out_dir() / f"preds_{model}.npz"))
    meta = json.loads((out_dir() / f"preds_{model}.json").read_text(encoding="utf-8"))
    rows = {}
    for g, tasks in groups.items():
        full = []
        for t in tasks:
            a = np.full(len(t.depth), np.nan)
            a[t.predict_mask] = z[f"{g}|{t.meta['gap_id']}|p50"]
            full.append(a)
        base = gap_rows(tasks, full, [[] for _ in tasks], wells)
        out = []
        for t, r, f in zip(tasks, base, full):
            k = t.meta["gap_id"]
            lo, hi = z.get(f"{g}|{k}|lo"), z.get(f"{g}|{k}|hi")
            out.append({**r, **features(t, f[t.predict_mask], lo, hi, gardner[(g, k)]), "group": g})
        rows[g] = out
    return rows, meta


def decide(models: list[str] | None = None) -> dict:
    from workstreams.black_physics.granite.pipeline import gardner_on, log

    cfg = C.CONFIG
    groups, wells = tasks_by_group()
    gardner = {(g, t.meta["gap_id"]): gardner_on(t)[t.predict_mask] for g, ts in groups.items() for t in ts}
    models = models or sorted(p.stem[len("preds_"):] for p in out_dir().glob("preds_*.npz"))
    summary, routing = {}, {}
    for model in models:
        rows, meta = records(model, groups, wells, gardner)
        has_band = any(np.isfinite(r["band_width"]) for r in rows["C"])
        base = BASE + (["band_width"] if has_band else [])
        width = "band_width" if has_band else "flank_var"
        rules = {"prevalence": DEC.fit_prevalence(rows["C"]),
                 "width_only": DEC.fit_rule("width_only", rows["C"], [width], cfg.l2, cfg.prevalence_smoothing),
                 "numeric": DEC.fit_rule("numeric", rows["C"], base, cfg.l2, cfg.prevalence_smoothing),
                 "numeric_gardner": DEC.fit_rule("numeric_gardner", rows["C"], base + GARDNER, cfg.l2,
                                                 cfg.prevalence_smoothing),
                 "numeric_logs": DEC.fit_rule("numeric_logs", rows["C"], base + GARDNER + LOGS, cfg.l2,
                                              cfg.prevalence_smoothing)}
        D = DEC.labelled(rows["D"])
        yD = np.array([r["label"] for r in D], float)
        wD = DEC.well_weights([r["well"] for r in D])
        recal = {k: DEC.fit_recalibration(rules[k].logit(D), yD, wD, cfg.recal_l2, cfg.prevalence_smoothing)
                 for k in ("numeric", "numeric_gardner", "numeric_logs")}
        scores, sidecar, probs_by = {}, [], {}
        for g in GROUPS:
            R = rows[g]
            probs = {k: rules[k].proba(R) for k in rules}
            probs.update({f"{k}_cal": recal[k].apply(rules[k].logit(R)) for k in recal})
            probs_by[g] = probs["numeric_cal"]
            y, comp = [r["label"] for r in R], [r["complete"] for r in R]
            yy = [np.nan if v is None else v for v in y]
            scores[g] = {k: {**DEC.decision_metrics(v, y, comp, cfg.accept_threshold), "auc": auc(v, yy),
                             "risk_coverage": DEC.risk_coverage(v, y, comp)} for k, v in probs.items()}
            if g == "test":
                for i, r in enumerate(R):
                    sidecar.append({"test": C.TEST_NAME, "model": model, "well": r["well"], "gap_id": r["key"],
                                    "start_md": r["start_md"], "end_md": r["end_md"],
                                    "features": {k: r[k] for k in base + GARDNER + LOGS},
                                    "p_pass": {k: float(v[i]) for k, v in probs.items()},
                                    "decision": "accept" if probs["numeric_cal"][i] >= cfg.accept_threshold else "review",
                                    "mae": r["mae"], "label": r["label"]})
        routing[model] = {g: {r["key"]: (float(probs_by[g][i]), r) for i, r in enumerate(rows[g])} for g in ("D", "test")}
        res = {"model": model, "predictions": meta, "features": base + GARDNER + LOGS, "width_feature": width,
               "rules": {k: v.as_dict() for k, v in rules.items()},
               "recalibration": {k: v.as_dict() for k, v in recal.items()},
               "pass_rate": {g: float(np.mean([r["label"] for r in DEC.labelled(rows[g])])) for g in GROUPS},
               "mean_well_mae": {g: float(np.mean(list(_per_well(rows[g]).values()))) for g in GROUPS},
               "scores": scores, "test_sidecar": sidecar, "deployed": "numeric_cal", "threshold": cfg.accept_threshold}
        (out_dir() / f"{model}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        t, d = scores["test"]["numeric_cal"], scores["D"]["numeric_cal"]
        summary[model] = {"pass_rate": res["pass_rate"], "mean_well_mae": res["mean_well_mae"], "band": meta["band"],
                          "test": {k: t[k] for k in ("brier", "accepted", "review", "false_accepts", "n_gaps", "auc")},
                          "D": {k: d[k] for k in ("brier", "accepted", "review", "false_accepts", "n_gaps", "auc")},
                          "auc": {g: {k: scores[g][k]["auc"] for k in ("width_only", "numeric", "numeric_gardner",
                                                                        "numeric_logs")}
                                  for g in GROUPS},
                          "logs_cal": {g: {k: scores[g]["numeric_logs_cal"][k]
                                           for k in ("brier", "accepted", "false_accepts", "n_gaps")} for g in ("D", "test")},
                          "brier_test": {k: scores["test"][k]["brier"] for k in METHODS}}
        log(f"decide_any {model}: test pass rate {res['pass_rate']['test']:.2f}; numeric (recal. D) accepts "
            f"{t['accepted']}/{t['n_gaps']} with {t['false_accepts']} false (D: {d['accepted']}/{d['n_gaps']}, "
            f"{d['false_accepts']} false); AUC C/D/test {scores['C']['numeric']['auc']:.2f}/"
            f"{scores['D']['numeric']['auc']:.2f}/{(scores['test']['numeric']['auc'] or float('nan')):.2f}")
    if len(routing) > 1:
        route = route_by_trust(routing)
        (out_dir() / "routing.json").write_text(json.dumps(route, indent=1, default=float), encoding="utf-8")
        for g in ("D", "test"):
            log(f"decide_any routing over {len(routing)} models, {g}: routed {route[g]['routed']:.4f} vs best single "
                f"{route[g]['best_single'][1]:.4f} ({route[g]['best_single'][0]}); picks {route[g]['picks']}")
    prev = {}
    if (out_dir() / "summary.json").is_file():
        prev = json.loads((out_dir() / "summary.json").read_text(encoding="utf-8"))
    prev.update(summary)
    (out_dir() / "summary.json").write_text(json.dumps(prev, indent=1, default=float), encoding="utf-8")
    return summary


def route_by_trust(routing: dict) -> dict:
    """Per gap, take the fill of the model whose rule gives the highest p_pass; mean per-well MAE vs each model."""
    out = {}
    for g in ("D", "test"):
        keys = set.intersection(*(set(routing[m][g]) for m in routing))
        chosen, picks = [], {}
        for k in sorted(keys):
            m = max(routing, key=lambda m: (routing[m][g][k][0], m))
            chosen.append(routing[m][g][k][1])
            picks[m] = picks.get(m, 0) + 1
        single = {m: float(np.mean(list(_per_well([routing[m][g][k][1] for k in keys]).values()))) for m in routing}
        best = min(single, key=single.get)
        out[g] = {"n_gaps": len(keys), "routed": float(np.mean(list(_per_well(chosen).values()))),
                  "best_single": (best, single[best]), "single": single,
                  "picks": dict(sorted(picks.items(), key=lambda x: -x[1]))}
    return out


def _per_well(rows) -> dict:
    by: dict = {}
    for r in rows:
        s = by.setdefault(r["well"], [0.0, 0])
        s[0] += r["abs_err_sum"]
        s[1] += r["n_pred"]
    return {w: a / n for w, (a, n) in by.items() if n}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Trust rule for any registered model (bring your own model)")
    ap.add_argument("step", choices=["predict", "decide"])
    ap.add_argument("--model", help="comma-separated contender names (predict: required; decide: default all)")
    a = ap.parse_args(argv)
    names = [m.strip() for m in (a.model or "").split(",") if m.strip()]
    if a.step == "predict":
        if not names:
            raise SystemExit("--model is required for predict")
        for m in names:
            predict(m)
    else:
        decide(names or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
