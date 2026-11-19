"""FH2026 ML proof of concept: XGBoost baselines on FORCE 2020.

  1) Lithology: XGBoost classifier vs an always-'shale' floor.
  2) Density: RHOB with XGBoost regression, XGBoost quantile regression, and an XGBoost multiclass on
     10 / 15 uniform bins (+ one 10-bin equal-count variant); prediction = probability-weighted mean of
     bin centres, uncertainty = weighted std. A 'bin floor' shows the best any bin model could do.

CPU only, open FORCE 2020 data only, fixed inputs, no tuning. Run:
    python poc_run.py                 # everything (a few minutes on an 8-core CPU)
    python poc_run.py --only litho    # or dens
Results: results/poc-results.json next to this file. Data: $FH2026_CACHE/data/force2020.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import poc_common as C

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def timing(train_s, pred_s, n_pred, rows_per_well, load_s=None):
    per1k = pred_s / n_pred * 1000
    out = {"train_s": None if train_s is None else round(train_s, 1),
           "pred_ms_per_1000": round(per1k * 1000, 1),
           "pred_s_per_well_est": round(per1k * rows_per_well / 1000, 3)}
    if load_s is not None:
        out["load_s"] = round(load_s, 1)
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["litho", "dens"], default=None)
    ap.add_argument("--train-per-well", type=int, default=250, help="sampled training rows per training well")
    ap.add_argument("--test-per-well", type=int, default=300, help="scored samples per held-out well")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(C.CACHE / "hf"))
    C.WORK.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True)
    import xgboost as xgb

    res_path = RESULTS / "poc-results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res["hardware"] = C.hardware()
    tr, te, trw, tew = C.load_split()
    res["split"] = {"train_wells": trw, "test_wells": tew, "train_rows": len(tr), "test_rows": len(te)}
    rows_per_well = len(te) / len(tew)
    log("train wells", trw, "| held-out wells", tew)
    formations = sorted(tr.FORMATION.dropna().unique())
    rng = np.random.default_rng(0)

    def sample(df, n):
        return df.groupby("WELL", group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=0))

    # ---------------------------------------------------- 1) lithology
    if a.only in (None, "litho"):
        log("== lithology ==")
        cur = C.LITHO_CURVES
        trl = C.add_window(tr, cur); tel = C.add_window(te, cur)
        trl = trl[trl.FORCE_2020_LITHOFACIES_LITHOLOGY.isin(C.ORDER)]
        tel = tel[tel.FORCE_2020_LITHOFACIES_LITHOLOGY.isin(C.ORDER)]
        trs, tes = sample(trl, a.train_per_well), sample(tel, a.test_per_well)
        y_tr_s = trs.FORCE_2020_LITHOFACIES_LITHOLOGY.map(C.IDX).to_numpy()
        y_tr_all = trl.FORCE_2020_LITHOFACIES_LITHOLOGY.map(C.IDX).to_numpy()
        y_te = tes.FORCE_2020_LITHOFACIES_LITHOLOGY.map(C.IDX).to_numpy()
        out = {"n_test": len(tes), "test_class_counts": {C.NAMES[k]: int(v) for k, v in zip(*np.unique(y_te, return_counts=True))}}
        out["always_shale"] = C.litho_scores(y_te, np.full_like(y_te, 2))
        X_te = C.feature_matrix(tes, cur, formations)
        X_te_all = C.feature_matrix(tel, cur, formations)
        for label, df, y in [("xgb_same_sample", trs, y_tr_s), ("xgb_all_rows", trl, y_tr_all)]:
            cls = np.unique(y); remap = {c: i for i, c in enumerate(cls)}
            t0 = time.perf_counter()
            m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1)
            m.fit(C.feature_matrix(df, cur, formations), np.vectorize(remap.get)(y))
            ttr = time.perf_counter() - t0
            p = cls[m.predict(X_te)]
            t0 = time.perf_counter(); m.predict(X_te_all); tp = time.perf_counter() - t0
            out[label] = {**C.litho_scores(y_te, p), "n_train": len(df), **timing(ttr, tp, len(X_te_all), rows_per_well)}
            log(label, out[label])
        res["lithology"] = out
        res_path.write_text(json.dumps(res, indent=1))

    # ---------------------------------------------------- 2) density (binned RHOB)
    if a.only in (None, "dens"):
        log("== density ==")
        cur = C.DENS_CURVES
        trd = C.add_window(tr, cur).dropna(subset=["RHOB"]); ted = C.add_window(te, cur).dropna(subset=["RHOB"])
        trs, tes = sample(trd, a.train_per_well), sample(ted, a.test_per_well)
        y_te = tes.RHOB.to_numpy(); X_te = C.feature_matrix(tes, cur, formations)
        X_te_all = C.feature_matrix(ted, cur, formations)
        out = {"n_test": len(tes), "test_rhob_mean": round(float(y_te.mean()), 3), "test_rhob_std": round(float(y_te.std()), 3)}
        out["predict_train_mean"] = C.dist_metrics(y_te, np.full_like(y_te, trd.RHOB.mean()))
        for label, df in [("xgb_reg_same_sample", trs), ("xgb_reg_all_rows", trd)]:
            t0 = time.perf_counter()
            m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1)
            m.fit(C.feature_matrix(df, cur, formations), df.RHOB); ttr = time.perf_counter() - t0
            p = m.predict(X_te)
            t0 = time.perf_counter(); m.predict(X_te_all); tp = time.perf_counter() - t0
            out[label] = {**C.dist_metrics(y_te, p), "n_train": len(df), **timing(ttr, tp, len(X_te_all), rows_per_well)}
            log(label, out[label])
        # quantile regression: 2.3/16/50/84/97.7 % -> +-1 sigma and +-2 sigma intervals
        qs = np.array([0.023, 0.16, 0.5, 0.84, 0.977])
        t0 = time.perf_counter()
        mq = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=qs, n_estimators=300, max_depth=6,
                              learning_rate=0.1, tree_method="hist", n_jobs=-1)
        mq.fit(C.feature_matrix(trs, cur, formations), trs.RHOB); ttr = time.perf_counter() - t0
        P = np.sort(mq.predict(X_te), axis=1)
        t0 = time.perf_counter(); mq.predict(X_te_all); tp = time.perf_counter() - t0
        out["xgb_quantile_same_sample"] = {**C.dist_metrics(y_te, P[:, 2], lo1=P[:, 1], hi1=P[:, 3], lo2=P[:, 0], hi2=P[:, 4]),
                                           "n_train": len(trs), **timing(ttr, tp, len(X_te_all), rows_per_well)}
        log("xgb_quantile_same_sample", out["xgb_quantile_same_sample"])
        variants = [(10, "uniform"), (15, "uniform"), (10, "quantile")]
        for nb, kind in variants:
            edges, centres = C.bins(nb, kind=kind, y=trs.RHOB.to_numpy())
            b_tr = C.to_bin(trs.RHOB.to_numpy(), edges)
            tag = f"{nb}bins_{kind}"
            # bin floor: perfect bin classifier, mean = true bin centre
            out[f"bin_floor_{tag}"] = C.dist_metrics(y_te, centres[C.to_bin(y_te, edges)])
            # fallback classifier: XGBoost multiclass on the same bins
            cls = np.unique(b_tr); remap = {c: i for i, c in enumerate(cls)}
            t0 = time.perf_counter()
            m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1)
            m.fit(C.feature_matrix(trs, cur, formations), np.vectorize(remap.get)(b_tr)); ttr = time.perf_counter() - t0
            pr = m.predict_proba(X_te); full = np.zeros((len(pr), nb)); full[:, cls] = pr
            t0 = time.perf_counter(); m.predict_proba(X_te_all); tp = time.perf_counter() - t0
            mu = full @ centres; sd = np.sqrt(np.maximum(full @ centres ** 2 - mu ** 2, 0))
            out[f"xgb_binclass_{tag}"] = {**C.dist_metrics(y_te, mu, sd), "n_train": len(trs), **timing(ttr, tp, len(X_te_all), rows_per_well)}
            log(f"xgb_binclass_{tag}", out[f"xgb_binclass_{tag}"])
            res["density"] = out
            res_path.write_text(json.dumps(res, indent=1))
    log("results ->", res_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
