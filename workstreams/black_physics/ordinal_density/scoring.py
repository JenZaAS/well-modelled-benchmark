"""Scores for binned density distributions, averaged depths within gap, gaps within well, then wells.

The hidden truth enters only here (and as the training label on C).
"""
from __future__ import annotations

import numpy as np

from workstreams.black_physics.ordinal_density import bins as B

SEED = 2026
N_BOOT = 4000


def per_well(values, wells, gaps) -> dict:
    """{well: mean over its gaps of the mean over each gap's depths} for finite values."""
    v, wells, gaps = np.asarray(values, float), np.asarray(wells), np.asarray(gaps)
    out = {}
    for w in np.unique(wells):
        iw = (wells == w) & np.isfinite(v)
        gm = [v[iw & (gaps == g)].mean() for g in np.unique(gaps[iw])]
        if gm:
            out[str(w)] = float(np.mean(gm))
    return out


def hier_mean(values, wells, gaps) -> float:
    pw = per_well(values, wells, gaps)
    return float(np.mean(list(pw.values()))) if pw else float("nan")


def paired_bootstrap(a: dict, b: dict, seed: int = SEED, n: int = N_BOOT) -> dict:
    """Mean of a - b over common wells, a 95 % interval from resampling wells, and how many wells a is lower."""
    ws = sorted(set(a) & set(b))
    d = np.array([a[w] - b[w] for w in ws])
    if not len(d):
        return {"diff": None, "ci95": [None, None], "wells_lower": 0, "n_wells": 0}
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "wells_lower": int((d < 0).sum()), "n_wells": int(len(d))}


def reliability(prob, event, n_bins: int = 10) -> list[dict]:
    """Equal-width bins of predicted probability: count, mean prediction, observed frequency (depth-level)."""
    p, e = np.asarray(prob, float), np.asarray(event, float)
    ok = np.isfinite(p) & np.isfinite(e)
    p, e = p[ok], e[ok]
    k = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": int((k == i).sum()),
             "mean_p": float(p[k == i].mean()) if (k == i).any() else None,
             "observed": float(e[k == i].mean()) if (k == i).any() else None} for i in range(n_bins)]


def score(p, y, wells, gaps, tails: B.Tails, point=None) -> dict:
    """All metrics of one probability matrix p [n, 15] against hidden values y [n].

    point: the base model's own point prediction (for the original point MAE). Rows with NaN p are abstentions.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    wells, gaps = np.asarray(wells), np.asarray(gaps)
    have_p = np.isfinite(p).all(axis=1)
    have_y = np.isfinite(y)
    ok = have_p & have_y
    b = B.to_bin(y)
    nan = np.full(len(y), np.nan)

    def put(v):
        out = nan.copy()
        out[ok] = v
        return out

    pk, yk = p[ok], y[ok]
    r = B.readouts(pk, tails)
    rps = put(B.rps(pk, b[ok]))
    ll = put(B.log_loss(pk, b[ok]))
    below = B.prob_below(pk, B.EVENT_EDGE)
    ev = (yk < B.EVENT_EDGE).astype(float)
    brier = put((below - ev) ** 2)
    cov80 = put(((yk >= r["q10"]) & (yk <= r["q90"])).astype(float))
    cov90 = put(((yk >= r["q05"]) & (yk <= r["q95"])).astype(float))
    w80, w90 = put(r["q90"] - r["q10"]), put(r["q95"] - r["q05"])
    mae_med, mae_mean = put(np.abs(r["median"] - yk)), put(np.abs(r["mean"] - yk))
    out = {
        "n_wells": int(len(np.unique(wells[ok]))), "n_gaps": int(len(np.unique(gaps[ok]))), "n_depths": int(ok.sum()),
        "abstentions": int((~have_p & have_y).sum()), "unscored_no_truth": int((~have_y).sum()),
        "RPS": hier_mean(rps, wells, gaps), "log_loss": hier_mean(ll, wells, gaps),
        "brier_below_2.40": hier_mean(brier, wells, gaps),
        "cover80": hier_mean(cov80, wells, gaps), "width80": hier_mean(w80, wells, gaps),
        "cover90": hier_mean(cov90, wells, gaps), "width90": hier_mean(w90, wells, gaps),
        "MAE_binned_median": hier_mean(mae_med, wells, gaps), "MAE_binned_mean": hier_mean(mae_mean, wells, gaps),
        "tail_mass_mean": {"low": float(pk[:, 0].mean()) if len(pk) else None,
                           "high": float(pk[:, -1].mean()) if len(pk) else None},
        "reliability_below_2.40": reliability(below, ev),
        "_per_well_RPS": per_well(rps, wells, gaps),
        "_per_well_brier": per_well(brier, wells, gaps),
    }
    if point is not None:
        pt = np.asarray(point, float)
        out["MAE_point"] = hier_mean(np.where(ok, np.abs(pt - y), np.nan), wells, gaps)
    return out


def bin_resolution(y, wells, gaps, tails: B.Tails) -> dict:
    """Diagnostic: MAE of predicting each hidden value's own bin representative, and tail counts."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    b = B.to_bin(y)
    rep = tails.means()
    err = np.where(ok, np.abs(rep[np.clip(b, 0, B.N_BINS - 1)] - y), np.nan)
    return {"MAE_correct_bin_representative": hier_mean(err, wells, gaps),
            "n_low_tail": int((ok & (b == 0)).sum()), "n_high_tail": int((ok & (b == B.N_BINS - 1)).sum()),
            "n": int(ok.sum())}


def native_band(lo, hi, y, wells, gaps, level: float) -> dict:
    """Coverage and width of a base model's own interval at its nominal level (reference only)."""
    lo, hi, y = (np.asarray(x, float) for x in (lo, hi, y))
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    cov = np.where(ok, ((y >= lo) & (y <= hi)).astype(float), np.nan)
    wid = np.where(ok, hi - lo, np.nan)
    return {"level": level, "cover": hier_mean(cov, wells, gaps), "width": hier_mean(wid, wells, gaps),
            "n_depths": int(ok.sum())}
