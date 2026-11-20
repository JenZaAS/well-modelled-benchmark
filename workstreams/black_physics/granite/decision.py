"""Gap-level pass/review decisions and how to score them. Model-agnostic: works on keyed records from any contender.

A record is one whole gap: {"key", "well", "label" (1 = mean |error| <= tolerance, None if incomplete),
"complete", features...}. Learned fits (C wells) use equal weight per well; recalibration uses D wells.

Methods attached to the selected reconstruction:
    prevalence        C well-weighted pass rate for every gap
    width_only        one-feature logistic on the width / variability proxy
    numeric           L2 logistic on all base features (+ the two Gardner features: numeric_gardner)
    decima            Decima-small P(yes) on the text state (raw), and its recalibrated version
An incomplete prediction is never accepted: it goes to review and stays in the denominator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def well_weights(wells) -> np.ndarray:
    """1 / (#records of the well), rescaled to mean 1."""
    wells = list(wells)
    counts: dict[str, int] = {}
    for w in wells:
        counts[w] = counts.get(w, 0) + 1
    w = np.array([1.0 / counts[x] for x in wells])
    return w * (len(w) / w.sum()) if len(w) else w


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic(X, y, w=None, l2: float = 1.0, prior=None, iters: int = 100) -> np.ndarray:
    """Penalised logistic regression by Newton's method. Column 0 must be the intercept (unpenalised).

    Penalty: l2/2 * ||beta[1:] - prior[1:]||^2 (prior defaults to 0). Returns beta.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    k = X.shape[1]
    prior = np.zeros(k) if prior is None else np.asarray(prior, float)
    P = np.eye(k) * l2
    P[0, 0] = 0.0
    def loss(b):
        z = X @ b
        return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * (b - prior) @ P @ (b - prior))

    beta = prior.copy()
    cur = loss(beta)
    for _ in range(iters):
        p = _sigmoid(X @ beta)
        g = X.T @ (w * (p - y)) + P @ (beta - prior)
        Hm = (X * (w * p * (1 - p))[:, None]).T @ X + P + 1e-9 * np.eye(k)
        step = np.linalg.solve(Hm, g)
        t = 1.0                     # step halving: a full Newton step can overshoot when some |logit| is very large
        while t > 1e-6 and loss(beta - t * step) > cur + 1e-12:
            t *= 0.5
        beta = beta - t * step
        cur = loss(beta)
        if np.abs(t * step).max() < 1e-10:
            break
    return beta


@dataclass
class Prep:
    """Missing-value fill, indicators and standardisation, all fitted on the training (C) records."""
    features: list[str]
    median: dict = field(default_factory=dict)
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    indicators: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, rows, features):
        p = cls(list(features))
        for f in features:
            v = np.array([r[f] for r in rows], float)
            ok = np.isfinite(v)
            med = float(np.median(v[ok])) if ok.any() else 0.0
            p.median[f] = med
            filled = np.where(ok, v, med)
            p.mean[f] = float(filled.mean()) if len(filled) else 0.0
            sd = float(filled.std()) if len(filled) else 0.0
            p.std[f] = sd if sd > 1e-12 else 1.0
            if (~ok).any():
                p.indicators.append(f)
        return p

    def transform(self, rows) -> np.ndarray:
        cols = [np.ones(len(rows))]
        for f in self.features:
            v = np.array([r[f] for r in rows], float)
            v = np.where(np.isfinite(v), v, self.median[f])
            cols.append((v - self.mean[f]) / self.std[f])
        for f in self.indicators:
            cols.append(np.array([0.0 if np.isfinite(r[f]) else 1.0 for r in rows]))
        return np.column_stack(cols)

    def columns(self) -> list[str]:
        return ["intercept", *self.features, *[f"{f}_missing" for f in self.indicators]]

    def as_dict(self) -> dict:
        return {"features": self.features, "median": self.median, "mean": self.mean, "std": self.std,
                "indicators": self.indicators}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def labelled(rows):
    return [r for r in rows if r.get("complete") and r.get("label") is not None]


def smoothed_rate(y, w, alpha: float = 1.0) -> float:
    y, w = np.asarray(y, float), np.asarray(w, float)
    return float((np.sum(w * y) + alpha) / (np.sum(w) + 2 * alpha))


@dataclass
class LogisticRule:
    name: str
    prep: Prep | None
    beta: list | None
    constant: float | None = None        # a constant probability (prevalence, or a one-class fallback)
    note: str = ""

    def logit(self, rows) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(rows), float(logit(self.constant)))
        return self.prep.transform(rows) @ np.asarray(self.beta)

    def proba(self, rows) -> np.ndarray:
        return _sigmoid(self.logit(rows))

    def as_dict(self):
        return {"name": self.name, "prep": None if self.prep is None else self.prep.as_dict(), "beta": self.beta,
                "constant": self.constant, "note": self.note,
                "columns": None if self.prep is None else self.prep.columns()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], None if d["prep"] is None else Prep.from_dict(d["prep"]), d["beta"], d["constant"], d["note"])


def fit_prevalence(rows, alpha: float = 0.0) -> LogisticRule:
    L = labelled(rows)
    y = np.array([r["label"] for r in L], float)
    w = well_weights([r["well"] for r in L])
    p = smoothed_rate(y, w, alpha) if alpha else float(np.sum(w * y) / np.sum(w))
    return LogisticRule("prevalence", None, None, constant=float(np.clip(p, 1e-4, 1 - 1e-4)),
                        note="C well-weighted pass rate")


def fit_rule(name: str, rows, features, l2: float, alpha: float = 1.0) -> LogisticRule:
    L = labelled(rows)
    y = np.array([r["label"] for r in L], float)
    w = well_weights([r["well"] for r in L])
    if len(set(y.tolist())) < 2:
        return LogisticRule(name, None, None, constant=smoothed_rate(y, w, alpha),
                            note=f"one observed class in C ({int(y.sum())}/{len(y)} pass): smoothed prevalence")
    prep = Prep.fit(L, features)
    beta = fit_logistic(prep.transform(L), y, w, l2)
    return LogisticRule(name, prep, [float(b) for b in beta])


@dataclass
class Recalibration:
    a: float = 0.0
    b: float = 1.0
    constant: float | None = None
    note: str = ""

    def apply(self, z) -> np.ndarray:
        z = np.asarray(z, float)
        if self.constant is not None:
            return np.full(len(z), self.constant)
        return _sigmoid(self.a + self.b * z)

    def as_dict(self):
        return {"a": self.a, "b": self.b, "constant": self.constant, "note": self.note}


def fit_recalibration(z, y, w, l2: float, alpha: float = 1.0) -> Recalibration:
    """p = sigmoid(a + b z) on D, penalty l2/2 * (b - 1)^2 (identity when D is uninformative)."""
    z, y, w = np.asarray(z, float), np.asarray(y, float), np.asarray(w, float)
    if len(set(y.tolist())) < 2:
        return Recalibration(constant=smoothed_rate(y, w, alpha),
                             note=f"one observed class in D ({int(y.sum())}/{len(y)} pass): smoothed prevalence")
    X = np.column_stack([np.ones(len(z)), z])
    beta = fit_logistic(X, y, w, l2, prior=np.array([0.0, 1.0]))
    return Recalibration(float(beta[0]), float(beta[1]))


# ------------------------------------------------------------------------------------------- scoring
def decision_metrics(p, y, complete, threshold: float, bins=(0.0, 1 / 3, 2 / 3, 1.0)) -> dict:
    """Brier on complete labelled gaps; accept = complete and p >= threshold; everything else is review."""
    p = np.asarray(p, float)
    complete = np.asarray(complete, bool)
    yv = np.array([np.nan if v is None else float(v) for v in y])
    lab = complete & np.isfinite(yv) & np.isfinite(p)
    accept = complete & np.isfinite(p) & (p >= threshold)
    fa = int((accept & (yv == 0)).sum())
    n_acc = int(accept.sum())
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        inb = lab & (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        rel.append({"bin": f"[{lo:.2f}, {hi:.2f}{']' if hi >= 1 else ')'}", "n": int(inb.sum()),
                    "mean_p": float(p[inb].mean()) if inb.any() else None,
                    "pass_rate": float(yv[inb].mean()) if inb.any() else None})
    return {
        "n_gaps": int(len(p)), "n_complete": int(complete.sum()), "n_labelled": int(lab.sum()),
        "prevalence": float(yv[lab].mean()) if lab.any() else None,
        "brier": float(np.mean((p[lab] - yv[lab]) ** 2)) if lab.any() else None,
        "accepted": n_acc, "review": int(len(p) - n_acc), "false_accepts": fa,
        "accepted_risk": (fa / n_acc) if n_acc else None,
        "bad_gaps": int((lab & (yv == 0)).sum()), "good_sent_to_review": int((lab & (yv == 1) & ~accept).sum()),
        "reliability": rel, "threshold": threshold,
    }


def risk_coverage(p, y, complete) -> list[dict]:
    """Accept every complete gap with p >= t for each distinct t (ties enter together). Coverage over all gaps."""
    p = np.asarray(p, float)
    yv = np.array([np.nan if v is None else float(v) for v in y])
    ok = np.asarray(complete, bool) & np.isfinite(p) & np.isfinite(yv)
    n = len(p)
    out = [{"threshold": math.inf, "accepted": 0, "coverage": 0.0, "false_accepts": 0, "risk": None}]
    for t in sorted(set(p[ok].tolist()), reverse=True):
        acc = ok & (p >= t)
        k, fa = int(acc.sum()), int((acc & (yv == 0)).sum())
        out.append({"threshold": float(t), "accepted": k, "coverage": k / n if n else 0.0, "false_accepts": fa,
                    "risk": fa / k if k else None})
    return out


def risk_at_coverage(curve: list[dict], coverage: float) -> float | None:
    """Risk of the smallest tie-closed acceptance set whose coverage reaches `coverage` (None if never reached)."""
    for row in curve:
        if row["coverage"] >= coverage - 1e-12 and row["accepted"] > 0:
            return row["risk"]
    return None
