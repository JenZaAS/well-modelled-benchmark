"""Probability heads over the 15 fixed bins: empirical errors, a linear softmax head, and temperature calibration.

All heads are fitted on C rows only and never see the base model's weights (the base stays frozen). Row weights
give every well the same total weight, every gap the same weight within its well, and every depth the same
weight within its gap. Hyperparameters are fixed in advance (no selection inside C):
    SMOOTH  additive mass per bin before renormalising (keeps empty bins finite for log loss)
    L2      ridge penalty of the softmax head on standardised features (intercepts unpenalised)
    T_BOUNDS  allowed temperature range
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from workstreams.black_physics.ordinal_density.bins import EDGES, N_BINS

SMOOTH = 1e-4
L2 = 1e-3                      # default; the stage `fit` picks it from L2_GRID by grouped folds over C's wells
L2_GRID = (1e-3, 1e-4, 1e-5)
N_FOLDS = 4
MAXITER = 3000
T_BOUNDS = (0.2, 10.0)


def row_weights(wells, gaps) -> np.ndarray:
    """Equal total weight per well, per gap within a well, and per depth within a gap; mean 1."""
    wells, gaps = np.asarray(wells), np.asarray(gaps)
    w = np.zeros(len(wells))
    uw = np.unique(wells)
    for well in uw:
        iw = wells == well
        ug = np.unique(gaps[iw])
        for g in ug:
            ig = iw & (gaps == g)
            w[ig] = 1.0 / (len(uw) * len(ug) * ig.sum())
    return w * len(w) / w.sum() if len(w) else w


def smooth(p) -> np.ndarray:
    return (np.asarray(p, float) + SMOOTH) / (1.0 + N_BINS * SMOOTH)


# ------------------------------------------------------------------------------------------- empirical errors
def _mass(resid_sorted, cum, base) -> np.ndarray:
    """Bin masses of (base + residual): P(y < edge) = weight of residuals strictly below edge - base."""
    cdf = cum[np.searchsorted(resid_sorted, EDGES[None, :] - np.asarray(base, float)[:, None], side="left")]
    cdf = np.pad(cdf, ((0, 0), (1, 1)), constant_values=(0.0, 1.0))
    return smooth(np.diff(cdf, axis=1))


class Residuals:
    def __init__(self, resid, weights):
        o = np.argsort(resid)
        self.r = np.asarray(resid, float)[o]
        c = np.r_[0.0, np.cumsum(np.asarray(weights, float)[o])]
        self.cum = c / c[-1]

    def mass(self, base) -> np.ndarray:
        return _mass(self.r, self.cum, base)

    def as_dict(self) -> dict:
        return {"r": self.r.tolist(), "cum": self.cum.tolist()}

    @classmethod
    def from_dict(cls, d) -> "Residuals":
        o = cls.__new__(cls)
        o.r, o.cum = np.asarray(d["r"], float), np.asarray(d["cum"], float)
        return o


class EmpiricalHead:
    """Signed residuals (measured - predicted) from C, shifted by the new prediction and counted between edges.

    by_flank=True keeps one residual set per third of observed flank variability (equal-well-weighted cut points
    on C); a depth with no flank variability uses the pooled set.
    """

    def __init__(self, by_flank: bool = False):
        self.by_flank = by_flank
        self.pooled: Residuals | None = None
        self.cuts: np.ndarray | None = None
        self.groups: dict[int, Residuals] = {}

    def fit(self, base, y, flank_var, w):
        base, y, fv, w = (np.asarray(x, float) for x in (base, y, flank_var, w))
        ok = np.isfinite(base) & np.isfinite(y)
        r = y - base
        self.pooled = Residuals(r[ok], w[ok])
        if self.by_flank:
            f = ok & np.isfinite(fv)
            o = np.argsort(fv[f])
            cw = np.cumsum(w[f][o])
            self.cuts = np.interp([1 / 3, 2 / 3], cw / cw[-1], fv[f][o])
            grp = np.searchsorted(self.cuts, fv, side="right")
            for g in range(3):
                sel = f & (grp == g)
                if sel.any():
                    self.groups[g] = Residuals(r[sel], w[sel])
        return self

    def predict(self, base, flank_var) -> np.ndarray:
        base, fv = np.asarray(base, float), np.asarray(flank_var, float)
        out = np.full((len(base), N_BINS), np.nan)
        ok = np.isfinite(base)
        out[ok] = self.pooled.mass(base[ok])
        if self.by_flank and self.cuts is not None:
            grp = np.searchsorted(self.cuts, fv, side="right")
            for g, res in self.groups.items():
                sel = ok & np.isfinite(fv) & (grp == g)
                if sel.any():
                    out[sel] = res.mass(base[sel])
        return out

    def as_dict(self) -> dict:
        return {"kind": "empirical", "by_flank": self.by_flank, "pooled": self.pooled.as_dict(),
                "cuts": None if self.cuts is None else self.cuts.tolist(),
                "groups": {str(g): r.as_dict() for g, r in self.groups.items()}}

    @classmethod
    def from_dict(cls, d) -> "EmpiricalHead":
        h = cls(d["by_flank"])
        h.pooled = Residuals.from_dict(d["pooled"])
        h.cuts = None if d["cuts"] is None else np.asarray(d["cuts"], float)
        h.groups = {int(g): Residuals.from_dict(r) for g, r in d["groups"].items()}
        return h


# ------------------------------------------------------------------------------------------- softmax head
class Scaler:
    """Train-only standardisation; NaN -> the training mean, plus a finite-value flag for columns that had NaN."""

    def fit(self, X, names):
        X = np.asarray(X, float)
        self.names = list(names)
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[~np.isfinite(self.std) | (self.std == 0)] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        self.flag = [i for i in range(X.shape[1]) if (~np.isfinite(X[:, i])).any()]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        Z = (X - self.mean) / self.std
        flags = [np.isfinite(X[:, i]).astype(float) for i in self.flag]
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return np.column_stack([np.ones(len(Z)), Z, *flags]) if len(flags) else np.column_stack([np.ones(len(Z)), Z])

    @property
    def columns(self) -> list[str]:
        return ["intercept"] + self.names + [f"finite[{self.names[i]}]" for i in self.flag]

    def as_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(), "flag": self.flag}

    @classmethod
    def from_dict(cls, d) -> "Scaler":
        s = cls()
        s.names, s.mean, s.std, s.flag = d["names"], np.asarray(d["mean"]), np.asarray(d["std"]), list(d["flag"])
        return s


class SoftmaxHead:
    """One linear layer to 15 logits, L2-regularised weighted cross-entropy (L-BFGS). All 15 columns are kept."""

    def fit(self, X, names, b, w, l2: float = L2, init: np.ndarray | None = None):
        self.scaler = Scaler().fit(X, names)
        Z = self.scaler.transform(X)
        b = np.asarray(b, int)
        w = np.asarray(w, float) / np.sum(w)
        k = Z.shape[1]
        Y = np.zeros((len(b), N_BINS))
        Y[np.arange(len(b)), b] = 1.0
        pen = np.ones((k, 1))
        pen[0] = 0.0

        def f(theta):
            W = theta.reshape(k, N_BINS)
            L = Z @ W
            lse = logsumexp(L, axis=1)
            loss = np.sum(w * (lse - L[np.arange(len(b)), b])) + 0.5 * l2 * np.sum(pen * W ** 2)
            P = np.exp(L - lse[:, None])
            grad = Z.T @ (w[:, None] * (P - Y)) + l2 * pen * W
            return loss, grad.ravel()

        x0 = np.zeros(k * N_BINS) if init is None or np.size(init) != k * N_BINS else np.ravel(init)
        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER})
        self.W = res.x.reshape(k, N_BINS)
        self.fit_info = {"converged": bool(res.success), "iterations": int(res.nit), "loss": float(res.fun),
                         "l2": l2, "columns": self.scaler.columns}
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        out = np.full((len(X), N_BINS), np.nan)
        ok = np.isfinite(X[:, 0])                     # column 0 is the base prediction: no base, no answer
        if ok.any():
            out[ok] = softmax(self.scaler.transform(X[ok]) @ self.W, axis=1)
        return out

    def as_dict(self) -> dict:
        return {"kind": "softmax", "scaler": self.scaler.as_dict(), "W": self.W.tolist(), "fit": self.fit_info}

    @classmethod
    def from_dict(cls, d) -> "SoftmaxHead":
        h = cls()
        h.scaler, h.W, h.fit_info = Scaler.from_dict(d["scaler"]), np.asarray(d["W"], float), d["fit"]
        return h


def well_folds(wells, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold number per row: wells sorted by name and dealt round-robin, so a well never spans two folds."""
    wells = np.asarray(wells)
    fold_of = {w: i % n_folds for i, w in enumerate(sorted(set(wells.tolist())))}
    return np.array([fold_of[w] for w in wells.tolist()])


def select_l2(X, names, b, wells, gaps, grid=L2_GRID, n_folds: int = N_FOLDS, log=None) -> tuple[float, dict]:
    """Pick the ridge penalty by grouped cross-validation over wells (training rows only).

    Score: held-out RPS averaged depth -> gap -> well over all folds. Larger penalties are fitted first and warm-start
    the next. Returns (best penalty, {penalty: CV RPS}).
    """
    from workstreams.black_physics.ordinal_density.bins import rps
    from workstreams.black_physics.ordinal_density.scoring import hier_mean

    X, b, wells, gaps = np.asarray(X, float), np.asarray(b, int), np.asarray(wells), np.asarray(gaps)
    folds = well_folds(wells, n_folds)
    table = {}
    preds ={l2: np.full((len(b), N_BINS), np.nan) for l2 in grid}
    for k in range(n_folds):
        tr = folds != k
        w = row_weights(wells[tr], gaps[tr])
        init = None
        for l2 in sorted(grid, reverse=True):
            h = SoftmaxHead().fit(X[tr], names, b[tr], w, l2=l2, init=init)
            init = h.W
            preds[l2][~tr] = h.predict(X[~tr])
            if log:
                log(f"  fold {k + 1}/{n_folds}, l2 {l2:g}: {h.fit_info['iterations']} it.")
    for l2 in grid:
        table[l2] = hier_mean(rps(preds[l2], b), wells, gaps)
    best = min(table, key=table.get)
    return best, table


# ------------------------------------------------------------------------------------------- temperature
def apply_temperature(p, T: float) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p).all(axis=1)
    out[ok] = softmax(np.log(np.clip(p[ok], 1e-300, 1.0)) / T, axis=1)
    return out


def fit_temperature(p, b, w) -> float:
    """One positive temperature minimising the weighted log loss of softmax(log p / T)."""
    p, b, w = np.asarray(p, float), np.asarray(b, int), np.asarray(w, float)
    ok = np.isfinite(p).all(axis=1) & (b >= 0)
    lp, b, w = np.log(np.clip(p[ok], 1e-300, 1.0)), b[ok], w[ok]

    def nll(logT):
        s = lp / np.exp(logT)
        return float(np.average(logsumexp(s, axis=1) - s[np.arange(len(b)), b], weights=w))

    r = minimize_scalar(nll, bounds=tuple(np.log(T_BOUNDS)), method="bounded")
    return float(np.exp(r.x))


def leave_one_well_out(p, b, wells, gaps) -> tuple[np.ndarray, list[float]]:
    """Calibrated probabilities for each well with the temperature fitted on the other wells (honest dev scores)."""
    p, wells = np.asarray(p, float), np.asarray(wells)
    out = np.full_like(p, np.nan)
    temps = []
    for well in np.unique(wells):
        tr = wells != well
        T = fit_temperature(p[tr], np.asarray(b)[tr], row_weights(wells[tr], np.asarray(gaps)[tr]))
        out[~tr] = apply_temperature(p[~tr], T)
        temps.append(T)
    return out, temps
