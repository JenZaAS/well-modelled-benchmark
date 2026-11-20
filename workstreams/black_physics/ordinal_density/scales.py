"""Stretch: a two-scale hybrid of two fills (plan section 6).

A Gaussian smoother (sigma 2 m, truncated at +-6 m, weights from real depth differences so unequal spacing is
handled, renormalised at the gap ends; it uses values inside the gap only) splits each predicted curve into a broad
part and a fine remainder. The truth is split the same way for scoring. On C, the model with the lower broad MAE
supplies the broad part and the model with the lower fine MAE supplies the fine part; that choice is frozen. On D,
broad(A) + fine(B) is compared with each single model and the equal average. The test is scored only if the hybrid
beats both single models on D.
"""
from __future__ import annotations

import json

import numpy as np

from workstreams.black_physics.ordinal_density import cli as CL
from workstreams.black_physics.ordinal_density import scoring as S

SIGMA_M, TRUNC_M = 2.0, 6.0


def smooth(depth, x) -> np.ndarray:
    z, x = np.asarray(depth, float), np.asarray(x, float)
    dz = z[:, None] - z[None, :]
    w = np.where(np.abs(dz) <= TRUNC_M, np.exp(-0.5 * (dz / SIGMA_M) ** 2), 0.0)
    ok = np.isfinite(x)
    w = w * ok[None, :]
    s = w.sum(axis=1)
    return np.where(s > 0, (w @ np.where(ok, x, 0.0)) / np.where(s > 0, s, 1.0), np.nan)


def split(depth, x, gaps) -> tuple[np.ndarray, np.ndarray]:
    broad = np.full(len(x), np.nan)
    for g in dict.fromkeys(gaps):
        i = np.flatnonzero(gaps == g)
        broad[i] = smooth(depth[i], x[i])
    return broad, x - broad


def mae(a, b, wells, gaps) -> float:
    return S.hier_mean(np.abs(np.asarray(a) - np.asarray(b)), wells, gaps)


def run(test: str, models: list[str], run_id: str | None) -> dict:
    if len(models) != 2:
        raise SystemExit("the two-scale hybrid needs exactly two models")
    d, core = CL.run_dir(test, models, run_id)
    truth = dict(np.load(d / "truth.npz"))
    rows = {m: dict(np.load(d / f"prepared_{m}.npz")) for m in models}
    parts = {}
    for g in ("C", "D", "test"):
        gaps, wells = truth[f"{g}|gap"], truth[f"{g}|well"]
        depth = rows[models[0]][f"{g}|depth"]
        yb, yf = split(depth, truth[f"{g}|y"], gaps)
        parts[g] = {"y": (yb, yf), "wells": wells, "gaps": gaps, "depth": depth,
                    **{m: split(depth, rows[m][f"{g}|base"], gaps) for m in models}}
    C = parts["C"]
    comp = {m: {"broad": mae(C[m][0], C["y"][0], C["wells"], C["gaps"]),
                "fine": mae(C[m][1], C["y"][1], C["wells"], C["gaps"])} for m in models}
    A = min(models, key=lambda m: comp[m]["broad"])
    Bm = min(models, key=lambda m: comp[m]["fine"])
    out = {"sigma_m": SIGMA_M, "truncate_m": TRUNC_M, "C_component_MAE": comp, "broad_from": A, "fine_from": Bm}
    res = {}
    for g in ("D", "test"):
        P = parts[g]
        y = truth[f"{g}|y"]
        single = {m: P[m][0] + P[m][1] for m in models}
        cand = {**{f"single:{m}": single[m] for m in models},
                "equal_average": 0.5 * (single[models[0]] + single[models[1]]), "hybrid": P[A][0] + P[Bm][1]}
        res[g] = {k: mae(v, y, P["wells"], P["gaps"]) for k, v in cand.items()}
        pw = {k: S.per_well(np.abs(v - y), P["wells"], P["gaps"]) for k, v in cand.items()}
        res[g]["hybrid_vs"] = {k: S.paired_bootstrap(pw["hybrid"], pw[k]) for k in cand if k != "hybrid"}
    out["D"] = res["D"]
    if A == Bm:
        out["advance_to_test"] = False
        out["verdict"] = (f"{CL.label(A)} has the lower C error at both scales, so the hybrid is that model alone; "
                          "the test was not scored")
    else:
        out["advance_to_test"] = all(res["D"]["hybrid"] < res["D"][f"single:{m}"] for m in models)
        if out["advance_to_test"]:
            out["test"] = res["test"]
            out["verdict"] = "the hybrid beat both single models on D, so it was scored on the test"
        else:
            out["verdict"] = "the hybrid did not beat both single models on D; the test was not scored"
    (d / "scales.json").write_text(json.dumps(out, indent=1, default=CL._json), encoding="utf-8")
    CL.log(f"scales: broad from {A}, fine from {Bm}; {out['verdict']}")
    CL.record(d, core, "scales", {"broad_from": A, "fine_from": Bm, "advance_to_test": out.get("advance_to_test")})
    return out
