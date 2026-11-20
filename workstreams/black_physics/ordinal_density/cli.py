"""Stages of the ordinal-density experiment: prepare, fit, score, plot (and the optional `scales` stretch).

    python -m workstreams.black_physics.ordinal_density prepare --test rhob_gap50
    python -m workstreams.black_physics.ordinal_density fit --test rhob_gap50
    python -m workstreams.black_physics.ordinal_density score --test rhob_gap50 --include-decima
    python -m workstreams.black_physics.ordinal_density plot --test rhob_gap50

Outputs: <work_root>/black_ordinal_density/<run-id>/ (config.json records every stage). The run id is the test
name plus a hash of the frozen configuration and the cached base predictions, so all four stages find the same
directory without being told; --run-id overrides it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from workstreams.black_physics.ordinal_density import bins as B
from workstreams.black_physics.ordinal_density import decima as DZ
from workstreams.black_physics.ordinal_density import features as FT
from workstreams.black_physics.ordinal_density import heads as H
from workstreams.black_physics.ordinal_density import scoring as S

DEFAULT_MODELS = ["red_tspulse_finetune", "red_timestral"]
INFO = {
    "red_tspulse_finetune": {"label": "TSPulse fine-tune", "inputs": "density only (RHOB flanks)"},
    "red_timestral": {"label": "Timestral-5M + other logs", "inputs": "RHOB flanks + other logs, also inside the gap"},
}
GROUPS = ("C", "D", "test")
HEADS = {   # name: (kind, feature policy kwargs)
    "empirical_pooled": ("empirical", {"by_flank": False}),
    "empirical_flank": ("empirical", {"by_flank": True}),
    "numeric": ("softmax", {"sonic": False, "products": False}),
    "numeric_sonic": ("softmax", {"sonic": True, "products": False}),
}
HEAD_LABEL = {"empirical_pooled": "Empirical errors, pooled",
              "empirical_flank": "Empirical errors by flank variability",
              "numeric": "Numeric head", "numeric_sonic": "Numeric head + sonic check",
              "decima_small": "Decima-small (pretrained)"}
BASELINE = "empirical_pooled"
DECIMA_MODELS = ["red_tspulse_finetune"]     # the plan's budget: Decima on the TSPulse fills only (704 + 160 states)
SEED = 2026


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def label(model: str) -> str:
    return INFO.get(model, {}).get("label", model)


# --------------------------------------------------------------------------------------------- locations
def root() -> Path:
    from wmb.paths import work_root

    return work_root() / "black_ordinal_density"


def cache_files(model: str) -> tuple[Path, Path]:
    """The decide_any keyed fills for this model (station id, else an older alias such as black_tspulse_ft)."""
    from wmb.aliases import canonical_model, legacy_model_names

    from workstreams.black_physics.granite import decide_any as DA

    for name in dict.fromkeys([model, canonical_model(model), *legacy_model_names(canonical_model(model))]):
        npz, js = DA.out_dir() / f"preds_{name}.npz", DA.out_dir() / f"preds_{name}.json"
        if npz.is_file() and js.is_file():
            return npz, js
    raise FileNotFoundError(f"no cached fills for {model} in {DA.out_dir()}; run granite.decide_any predict first")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_config(test: str, models: list[str]) -> dict:
    return {"test": test, "models": models, "groups": list(GROUPS), "seed": SEED,
            "bins": {"edges": B.EDGES.tolist(), "n_bins": B.N_BINS, "edge_rule": "edge value goes to the higher bin",
                     "tails": "C labels inside each open tail; one-bin extension if empty", "event_edge": B.EVENT_EDGE},
            "heads": {k: {"kind": v[0], **v[1]} for k, v in HEADS.items()},
            "hyper": {"smooth": H.SMOOTH, "l2_grid": list(H.L2_GRID), "l2_selection": f"grouped {H.N_FOLDS}-fold CV "
                      "over C wells (sorted, round-robin), held-out RPS", "maxiter": H.MAXITER,
                      "temperature_bounds": list(H.T_BOUNDS), "flank_context": FT.CTX,
                      "weights": "equal per well, per gap within well, per depth within gap"},
            "calibration": "one temperature per head fitted on D (well-weighted log loss); D scores use "
                           "leave-one-well-out temperatures",
            "decima": {"repo": DZ.REPO, "revision": DZ.REVISION, "precision": DZ.PRECISION, "kind": DZ.KIND,
                       "question": DZ.QUESTION, "choices": DZ.CHOICES, "per_gap": DZ.PER_GAP, "groups": ["D", "test"]},
            "cache_sha256": {m: sha256(cache_files(m)[0]) for m in models}}


def run_dir(test: str, models: list[str], run_id: str | None) -> tuple[Path, dict]:
    core = core_config(test, models)
    rid = run_id or f"{test}-{hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:10]}"
    d = root() / rid
    d.mkdir(parents=True, exist_ok=True)
    return d, core


def record(d: Path, core: dict, stage: str, info: dict) -> None:
    p = d / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"run_id": d.name, "core": core, "stages": {}}
    cfg["core"] = core
    cfg["stages"][stage] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    p.write_text(json.dumps(cfg, indent=1, default=_json), encoding="utf-8")
    log(f"{stage}: done; outputs and config in {d}")


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# --------------------------------------------------------------------------------------------- prepare
def _full(task, masked) -> np.ndarray | None:
    if masked is None:
        return None
    a = np.full(len(task.depth), np.nan)
    a[task.predict_mask] = masked
    return a


def check_provenance(meta: dict) -> dict:
    """The cached fills' checkpoint must still be the one the frozen manifests name."""
    prov = meta.get("provenance") or {}
    out = {"checkpoint": prov.get("checkpoint"), "checkpoint_sha256": prov.get("checkpoint_sha256"),
           "cache_provenance": prov}
    if prov.get("checkpoint_sha256"):
        from workstreams.black_physics.granite import artifacts as ART

        p = ART.ft_manifest_path()
        man = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        shas = {v.get("checkpoint_sha256") for v in man.get("variants", {}).values()}
        if prov["checkpoint_sha256"] not in shas:
            raise RuntimeError(f"cached fills were made with checkpoint {prov['checkpoint_sha256'][:12]}, which "
                               f"{p} no longer lists; re-run granite.decide_any predict")
        out["verified_against"] = str(p)
    return out


def prepare(test: str, models: list[str], run_id: str | None) -> Path:
    from workstreams.black_physics.granite import config as C
    from workstreams.black_physics.granite import decide_any as DA
    from workstreams.black_physics.granite.pipeline import gardner_on

    if test != C.TEST_NAME:
        raise SystemExit(f"the cached fills cover {C.TEST_NAME} only (got --test {test})")
    d, core = run_dir(test, models, run_id)
    groups, wells = DA.tasks_by_group()
    truth = {}
    gap_hash = hashlib.sha256()
    for g in GROUPS:
        ys, ws, gs, ix = [], [], [], []
        for t in groups[g]:
            m = np.asarray(t.predict_mask, bool)
            ys.append(np.asarray(wells[t.well_id].curves[C.TARGET], float)[m])
            ws.append(np.full(m.sum(), t.well_id))
            gs.append(np.full(m.sum(), t.meta["gap_id"]))
            ix.append(np.flatnonzero(m))
            gap_hash.update(f"{g}|{t.meta['gap_id']}|".encode())
            gap_hash.update(np.asarray(t.depth, float)[m].tobytes())
        truth[f"{g}|y"], truth[f"{g}|well"] = np.concatenate(ys), np.concatenate(ws)
        truth[f"{g}|gap"], truth[f"{g}|index"] = np.concatenate(gs), np.concatenate(ix)
    np.savez_compressed(d / "truth.npz", **truth)
    gard = {(g, t.meta["gap_id"]): gardner_on(t) for g in GROUPS for t in groups[g]}
    info = {"gap_hash": gap_hash.hexdigest(), "models": {}}
    for g in GROUPS:
        info[g] = {"wells": int(len(np.unique(truth[f"{g}|well"]))), "gaps": int(len(groups[g])),
                   "depths": int(len(truth[f"{g}|y"])), "wells_list_sha": hashlib.sha256(
                       ",".join(sorted(set(truth[f"{g}|well"]))).encode()).hexdigest()[:16]}
    assert not set(truth["C|well"]) & set(truth["D|well"]), "C and D share a well"
    for model in models:
        npz, js = cache_files(model)
        meta = json.loads(js.read_text(encoding="utf-8"))
        prov = check_provenance(meta)
        use_width = meta.get("band") == "P10-P90"          # native band only; TSPulse's is D-calibrated: not used
        level = {"P10-P90": 0.80, "P05-P95": 0.90}.get(meta.get("band"))
        z = dict(np.load(npz))
        arr = {}
        for g in GROUPS:
            rows = []
            for t in groups[g]:
                k = f"{g}|{t.meta['gap_id']}"
                p50 = z[f"{k}|p50"]
                if len(p50) != int(np.sum(t.predict_mask)):
                    raise RuntimeError(f"{model} {k}: {len(p50)} cached values for {int(np.sum(t.predict_mask))} "
                                       "hidden depths")
                lo, hi = z.get(f"{k}|lo"), z.get(f"{k}|hi")
                r = FT.gap_rows(t, _full(t, p50), _full(t, lo), _full(t, hi), gard[(g, t.meta["gap_id"])])
                r["lo"] = (_full(t, lo) if lo is not None else np.full(len(t.depth), np.nan))[r["index"]]
                r["hi"] = (_full(t, hi) if hi is not None else np.full(len(t.depth), np.nan))[r["index"]]
                r["well"] = np.full(len(r["index"]), t.well_id)
                r["gap"] = np.full(len(r["index"]), t.meta["gap_id"])
                rows.append(r)
            for f in rows[0]:
                arr[f"{g}|{f}"] = np.concatenate([r[f] for r in rows])
            if not (np.array_equal(arr[f"{g}|index"], truth[f"{g}|index"]) and
                    np.array_equal(arr[f"{g}|gap"], truth[f"{g}|gap"])):
                raise RuntimeError(f"{model} {g}: rows do not align with the hidden-depth keys")
        np.savez_compressed(d / f"prepared_{model}.npz", **arr)
        info["models"][model] = {
            "cache": str(npz), "cache_sha256": sha256(npz), "band": meta.get("band"), "band_level": level,
            "use_native_width": use_width, "provenance": prov, "inputs": INFO.get(model, {}).get("inputs"),
            "abstentions": {g: int((~np.isfinite(arr[f"{g}|base"])).sum()) for g in GROUPS},
            "dtc_support": {g: float(arr[f"{g}|dtc"].mean()) for g in GROUPS},
            "one_sided_depths": {g: int((~np.isfinite(arr[f"{g}|flank_up"]) | ~np.isfinite(arr[f"{g}|flank_dn"])).sum())
                                 for g in GROUPS}}
        sizes = [int(len(arr[f"{g}|base"])) for g in GROUPS]
        log(f"prepare {model}: C/D/test depths {sizes}, band {meta.get('band')}, "
            f"DTC support {[round(info['models'][model]['dtc_support'][g], 3) for g in GROUPS]}")
    record(d, core, "prepare", info)
    return d


# --------------------------------------------------------------------------------------------- fit
def load(d: Path, model: str) -> tuple[dict, dict]:
    return dict(np.load(d / f"prepared_{model}.npz", allow_pickle=False)), dict(np.load(d / "truth.npz"))


def stage_info(d: Path, stage: str) -> dict:
    return json.loads((d / "config.json").read_text(encoding="utf-8"))["stages"][stage]


def build_head(name: str, use_width: bool):
    kind, kw = HEADS[name]
    if kind == "empirical":
        return H.EmpiricalHead(**kw), None
    return H.SoftmaxHead(), FT.FeaturePolicy(use_width=use_width, **kw)


def head_predict(head, policy, rows: dict, g: str) -> np.ndarray:
    if policy is None:
        return head.predict(rows[f"{g}|base"], rows[f"{g}|flank_var"])
    X, _ = policy.matrix({f: rows[f"{g}|{f}"] for f in FT.FIELDS})
    return head.predict(X)


def fit(test: str, models: list[str], run_id: str | None) -> Path:
    d, core = run_dir(test, models, run_id)
    prep = stage_info(d, "prepare")
    info = {"models": {}}
    for model in models:
        rows, truth = load(d, model)
        use_width = prep["models"][model]["use_native_width"]
        yC = truth["C|y"]
        okC = np.isfinite(rows["C|base"]) & np.isfinite(yC)
        wC = H.row_weights(truth["C|well"][okC], truth["C|gap"][okC])
        tails = B.Tails.fit(yC)
        bundle = {"model": model, "use_native_width": use_width, "tails": tails.as_dict(), "heads": {}}
        probs, timing, cv = {}, {}, {}
        for name in HEADS:
            head, policy = build_head(name, use_width)
            t0 = time.perf_counter()
            if policy is None:
                head.fit(rows["C|base"][okC], yC[okC], rows["C|flank_var"][okC], wC)
            else:
                X, names = policy.matrix({f: rows[f"C|{f}"][okC] for f in FT.FIELDS})
                l2, table = H.select_l2(X, names, B.to_bin(yC[okC]), truth["C|well"][okC], truth["C|gap"][okC])
                cv[name] = {"chosen_l2": l2, "cv_rps": {f"{k:g}": v for k, v in table.items()}}
                log(f"fit {model} {name}: grouped {H.N_FOLDS}-fold CV on C -> l2 {l2:g} "
                    f"({', '.join(f'{k:g}: {v:.4f}' for k, v in table.items())})")
                head.fit(X, names, B.to_bin(yC[okC]), wC, l2=l2)
            fit_s = time.perf_counter() - t0
            raw = {}
            t0 = time.perf_counter()
            for g in ("D", "test"):
                raw[g] = head_predict(head, policy, rows, g)
            pred_s = (time.perf_counter() - t0) / sum(len(raw[g]) for g in raw)
            bD = B.to_bin(truth["D|y"])
            okD = np.isfinite(raw["D"]).all(axis=1) & (bD >= 0)
            T = H.fit_temperature(raw["D"][okD], bD[okD], H.row_weights(truth["D|well"][okD], truth["D|gap"][okD]))
            lowo = np.full_like(raw["D"], np.nan)
            lowo[okD], temps = H.leave_one_well_out(raw["D"][okD], bD[okD], truth["D|well"][okD], truth["D|gap"][okD])
            probs[f"D|{name}|raw"], probs[f"test|{name}|raw"] = raw["D"], raw["test"]
            probs[f"D|{name}|cal"], probs[f"test|{name}|cal"] = lowo, H.apply_temperature(raw["test"], T)
            bundle["heads"][name] = {**head.as_dict(), "policy": None if policy is None else policy.as_dict(),
                                     "temperature": T, "lowo_temperatures": temps}
            timing[name] = {"fit_s": round(fit_s, 3), "predict_ms_per_depth": 1000 * pred_s}
            log(f"fit {model} {name}: fit {fit_s:.1f} s, T = {T:.3f}" +
                (f", converged {head.fit_info['converged']} in {head.fit_info['iterations']} it."
                 if policy is not None else ""))
        np.savez_compressed(d / f"probs_{model}.npz", **probs)
        (d / f"heads_{model}.json").write_text(json.dumps(bundle, default=_json), encoding="utf-8")
        info["models"][model] = {"tails": {k: v for k, v in tails.as_dict().items() if k not in ("low", "high")},
                                 "timing": timing,
                                 "temperatures": {k: v["temperature"] for k, v in bundle["heads"].items()},
                                 "C_rows": int(okC.sum()), "l2_cv": cv}
    record(d, core, "fit", info)
    return d


# --------------------------------------------------------------------------------------------- score
def _public(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def score(test: str, models: list[str], run_id: str | None, include_decima: bool) -> Path:
    from workstreams.black_physics.granite.config import hf_home

    d, core = run_dir(test, models, run_id)
    prep = stage_info(d, "prepare")
    res = {"models": {}, "matched_decima": {}}
    pw_all = {}
    for model in models:
        rows, truth = load(d, model)
        bundle = json.loads((d / f"heads_{model}.json").read_text(encoding="utf-8"))
        tails = B.Tails.from_dict(bundle["tails"])
        P = dict(np.load(d / f"probs_{model}.npz"))
        mres = {"label": label(model), "inputs": prep["models"][model]["inputs"], "groups": {}}
        for g in ("D", "test"):
            y, wl, gp = truth[f"{g}|y"], truth[f"{g}|well"], truth[f"{g}|gap"]
            gres = {"heads": {}, "native_band": None, "bin_resolution": S.bin_resolution(y, wl, gp, tails)}
            lvl = prep["models"][model]["band_level"]
            if lvl:
                gres["native_band"] = S.native_band(rows[f"{g}|lo"], rows[f"{g}|hi"], y, wl, gp, lvl)
            per_well = {}
            for name in HEADS:
                for var in ("raw", "cal"):
                    s = S.score(P[f"{g}|{name}|{var}"], y, wl, gp, tails, point=rows[f"{g}|base"])
                    per_well[(name, var)] = s["_per_well_RPS"]
                    gres["heads"][f"{name}|{var}"] = _public(s)
            for (name, var), pw in per_well.items():
                gres["heads"][f"{name}|{var}"]["RPS_vs_baseline"] = S.paired_bootstrap(pw, per_well[(BASELINE, "raw")])
            # the physics experiment: the same numeric head with and without the Gardner-from-sonic disagreement
            gres["sonic_vs_numeric"] = {var: S.paired_bootstrap(per_well[("numeric_sonic", var)],
                                                                per_well[("numeric", var)]) for var in ("raw", "cal")}
            gres["dtc_support"] = prep["models"][model]["dtc_support"][g]
            mres["groups"][g] = gres
            pw_all[(model, g)] = per_well
        # the heatmap head: lowest development (D) RPS; D 'cal' rows use leave-one-well-out temperatures
        cand = {k: v["RPS"] for k, v in mres["groups"]["D"]["heads"].items()}
        mres["selected"] = min(cand, key=cand.get)
        log(f"score {model}: selected on D -> {mres['selected']} (D RPS {cand[mres['selected']]:.4f}); test RPS "
            f"{mres['groups']['test']['heads'][mres['selected']]['RPS']:.4f}")
        mres["fallbacks"] = {g: {"flank_head_pooled": int((~np.isfinite(rows[f"{g}|flank_var"])).sum()),
                                 "one_sided_depths": int((~np.isfinite(rows[f"{g}|flank_up"]) |
                                                          ~np.isfinite(rows[f"{g}|flank_dn"])).sum())}
                             for g in ("D", "test")}
        mres["runtime"] = stage_info(d, "fit")["models"][model]["timing"]
        write_sidecar(d, model, mres["selected"], rows, truth, P, tails)
        res["models"][model] = mres
        if include_decima and model in DECIMA_MODELS:
            res["matched_decima"][model] = matched_decima(d, model, rows, truth, P, tails,
                                                          prep["models"][model]["use_native_width"], hf_home() / "hub")
    if len(models) == 2:      # the two heatmap heads against each other, on the same wells and gaps
        a, b = models
        ka, kb = (tuple(res["models"][m]["selected"].split("|")) for m in (a, b))
        res["between_models"] = {"a": a, "b": b, "a_head": "|".join(ka), "b_head": "|".join(kb),
                                 **{g: S.paired_bootstrap(pw_all[(a, g)][ka], pw_all[(b, g)][kb])
                                    for g in ("D", "test")}}
    (d / "scores.json").write_text(json.dumps(res, indent=1, default=_json), encoding="utf-8")
    write_scoreboard(d, res)
    record(d, core, "score", {"include_decima": include_decima,
                              "selected": {m: v["selected"] for m, v in res["models"].items()}})
    return d


def write_sidecar(d: Path, model: str, key: str, rows: dict, truth: dict, P: dict, tails) -> Path:
    """The heatmap head's probabilities for every hidden test depth, keyed like the station (no truth inside).

    probs [n_depths, 15]; mean = the probability-weighted pitch estimate; median = the binned p50.
    """
    p = P[f"test|{key}"]
    r = B.readouts(p, tails)
    out = d / f"sidecar_{model}_test.npz"
    np.savez_compressed(out, probs=p, edges=B.EDGES, well=truth["test|well"], gap=truth["test|gap"],
                        index=truth["test|index"], depth=rows["test|depth"], base_p50=rows["test|base"],
                        mean=r["mean"], median=r["median"], q05=r["q05"], q10=r["q10"], q90=r["q90"], q95=r["q95"],
                        p_below_2_40=B.prob_below(p), head=np.array(key), model=np.array(model))
    return out


def matched_decima(d: Path, model: str, rows: dict, truth: dict, P: dict, tails, use_width: bool, hub: Path) -> dict:
    """Decima-small on 8 sampled depths per gap (D and test), and every other head scored on exactly those depths."""
    sel, states = {}, {}
    for g in ("D", "test"):
        idx = []
        gaps = truth[f"{g}|gap"]
        for gap in dict.fromkeys(gaps):
            pos = np.flatnonzero(gaps == gap)
            idx.extend(pos[DZ.sample_positions(len(pos), model, g, str(gap))])
        sel[g] = np.array(idx)
        states[g] = [DZ.state_text({f: rows[f"{g}|{f}"][i] for f in FT.FIELDS}, use_width) for i in sel[g]]
    allstates = states["D"] + states["test"]
    t0 = time.perf_counter()
    prob, secs, meta = DZ.run(allstates, root() / "decima_cache.jsonl", Path(hub), log)
    nD = len(states["D"])
    scores = meta.pop("_scores")
    dec = {"D": prob[:nD], "test": prob[nD:]}
    bD = B.to_bin(truth["D|y"][sel["D"]])
    wD, gD = truth["D|well"][sel["D"]], truth["D|gap"][sel["D"]]
    T = H.fit_temperature(dec["D"], bD, H.row_weights(wD, gD))
    lowo, temps = H.leave_one_well_out(dec["D"], bD, wD, gD)
    dec_v = {("D", "raw"): dec["D"], ("D", "cal"): lowo, ("test", "raw"): dec["test"],
             ("test", "cal"): H.apply_temperature(dec["test"], T)}
    np.savez_compressed(d / f"decima_{model}.npz", D_index=sel["D"], test_index=sel["test"], D=dec["D"],
                        test=dec["test"],
                        D_scores=scores[:nD], test_scores=scores[nD:], seconds=secs, choices=np.array(DZ.CHOICES),
                        question=np.array(DZ.QUESTION), D_states=np.array(states["D"]),
                        test_states=np.array(states["test"]))
    out = {"n_states": {g: int(len(sel[g])) for g in sel}, "temperature": T, "lowo_temperatures": temps,
           "seconds_mean": float(secs.mean()), "seconds_median": float(np.median(secs)),
           "seconds_total": float(secs.sum()), "wall_s": time.perf_counter() - t0, "decima": meta, "groups": {}}
    for g in ("D", "test"):
        i = sel[g]
        y, wl, gp = truth[f"{g}|y"][i], truth[f"{g}|well"][i], truth[f"{g}|gap"][i]
        gres, per_well = {}, {}
        for var in ("raw", "cal"):
            s = S.score(dec_v[(g, var)], y, wl, gp, tails, point=rows[f"{g}|base"][i])
            gres[f"decima_small|{var}"], per_well[("decima_small", var)] = _public(s), s["_per_well_RPS"]
            for name in HEADS:
                s = S.score(P[f"{g}|{name}|{var}"][i], y, wl, gp, tails, point=rows[f"{g}|base"][i])
                gres[f"{name}|{var}"], per_well[(name, var)] = _public(s), s["_per_well_RPS"]
        for (name, var), pw in per_well.items():
            gres[f"{name}|{var}"]["RPS_vs_baseline"] = S.paired_bootstrap(pw, per_well[(BASELINE, "raw")])
            gres[f"{name}|{var}"]["RPS_vs_decima_raw"] = S.paired_bootstrap(pw, per_well[("decima_small", "raw")])
        out["groups"][g] = gres
    log(f"score {model} Decima-small: D RPS raw {out['groups']['D']['decima_small|raw']['RPS']:.4f} / cal "
        f"{out['groups']['D']['decima_small|cal']['RPS']:.4f}; {1000 * out['seconds_mean']:.0f} ms per state")
    return out


def _f(x, nd=4) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(b: dict) -> str:
    if not b or b.get("diff") is None:
        return "–"
    return f"{b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] ({b['wells_lower']}/{b['n_wells']})"


def write_scoreboard(d: Path, res: dict) -> None:
    L = ["# Ordinal density scoreboard", "",
         "Official `rhob_gap50` test (20 wells, one 50 m gap each, every hidden depth) unless stated. Heads fitted on "
         "C (20 wells), temperature on D (12 wells). RPS: ranked probability score over the 15 fixed bins, lower is "
         "better; Δ vs pooled errors is a paired well-bootstrap 95 % interval (wells where the head is lower).", ""]
    for model, m in res["models"].items():
        t = m["groups"]["test"]
        L.append(f"## {m['label']} ({m['inputs']})")
        L.append("")
        L.append("| Head | Test RPS | Δ RPS vs pooled errors | Cover 80 % | Width 80 % | Cover 90 % | Width 90 % | "
                 "Brier P(<2.40) | MAE median | Dev (D) RPS |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, s in t["heads"].items():
            name, var = k.split("|")
            mark = " ◆" if k == m["selected"] else ""
            L.append(f"| {HEAD_LABEL[name]} ({'calibrated' if var == 'cal' else 'raw'}){mark} | {_f(s['RPS'])} | "
                     f"{_ci(s['RPS_vs_baseline'])} | {_f(s['cover80'], 2)} | {_f(s['width80'], 3)} | "
                     f"{_f(s['cover90'], 2)} | {_f(s['width90'], 3)} | {_f(s['brier_below_2.40'], 3)} | "
                     f"{_f(s['MAE_binned_median'])} | {_f(m['groups']['D']['heads'][k]['RPS'])} |")
        nb = t["native_band"]
        if nb:
            c80, w80 = (_f(nb["cover"], 2), _f(nb["width"], 3)) if nb["level"] == 0.8 else ("–", "–")
            c90, w90 = (_f(nb["cover"], 2), _f(nb["width"], 3)) if nb["level"] == 0.9 else ("–", "–")
            L.append(f"| Native band of the model ({int(100 * nb['level'])} %), reference | – | – | {c80} | {w80} | "
                     f"{c90} | {w90} | – | – | – |")
        any_head = next(iter(t["heads"].values()))
        L.append("")
        L.append(f"◆ heatmap head, chosen by lowest D RPS. Base point MAE {_f(any_head.get('MAE_point'))} g/cm³; "
                 f"bin-resolution floor (MAE of each hidden value's own bin representative) "
                 f"{_f(t['bin_resolution']['MAE_correct_bin_representative'])}; tail depths low/high "
                 f"{t['bin_resolution']['n_low_tail']}/{t['bin_resolution']['n_high_tail']}; "
                 f"{any_head['n_wells']} wells, {any_head['n_gaps']} gaps, {any_head['n_depths']} depths, "
                 f"{any_head['abstentions']} abstentions; {m['fallbacks']['test']['flank_head_pooled']} depths without "
                 f"flank variability (pooled fallback), {m['fallbacks']['test']['one_sided_depths']} with one flank "
                 f"only. Head runtime on CPU: " + ", ".join(
                     f"{HEAD_LABEL[h]} {1000 * v['predict_ms_per_depth']:.1f} µs per depth"
                     for h, v in m["runtime"].items()) + ".")
        L.append("")
        L.append("Sonic check (numeric head + sonic − numeric head, raw; negative favours the sonic feature): test "
                 f"{_ci(t['sonic_vs_numeric']['raw'])}, D {_ci(m['groups']['D']['sonic_vs_numeric']['raw'])}. "
                 f"Depths with a usable DTC (Gardner) value: test {100 * t['dtc_support']:.0f} %, D "
                 f"{100 * m['groups']['D']['dtc_support']:.0f} %.")
        L.append("")
    if "between_models" in res:
        bm = res["between_models"]
        L.append("## The two heatmap heads against each other")
        L.append("")
        L.append(f"RPS of {label(bm['a'])} ({bm['a_head']}) − {label(bm['b'])} ({bm['b_head']}); negative favours "
                 f"{label(bm['a'])}: test {_ci(bm['test'])}, D {_ci(bm['D'])} (wells where the first is lower).")
        L.append("")
    for model, z in res.get("matched_decima", {}).items():
        L.append(f"## Matched comparison with Decima-small: {label(model)}")
        L.append("")
        L.append(f"Eight sampled depths per gap ({z['n_states']['D']} D and {z['n_states']['test']} test states); "
                 f"every head scored on exactly those depths. Decima: {1000 * z['seconds_mean']:.0f} ms per state "
                 f"(INT8 ONNX, one CPU thread), temperature {z['temperature']:.2f}.")
        L.append("")
        L.append("| Head | D RPS | Test RPS | Test Δ RPS vs Decima-small raw | Test cover 80 % | Test width 80 % |")
        L.append("|---|---|---|---|---|---|")
        for k, s in z["groups"]["test"].items():
            name, var = k.split("|")
            L.append(f"| {HEAD_LABEL[name]} ({'calibrated' if var == 'cal' else 'raw'}) | "
                     f"{_f(z['groups']['D'][k]['RPS'])} | {_f(s['RPS'])} | {_ci(s['RPS_vs_decima_raw'])} | "
                     f"{_f(s['cover80'], 2)} | {_f(s['width80'], 3)} |")
        L.append("")
    (d / "scoreboard.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m workstreams.black_physics.ordinal_density",
                                 description="Where is a reconstructed well log uncertain? 15-bin density heads.")
    ap.add_argument("stage", choices=["prepare", "fit", "score", "plot", "scales", "all"])
    ap.add_argument("--test", default="rhob_gap50")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--run-id")
    ap.add_argument("--include-decima", action="store_true", help="run the pinned Decima-small checkpoint (score)")
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    stages = ["prepare", "fit", "score", "plot"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "prepare":
            prepare(a.test, models, a.run_id)
        elif st == "fit":
            fit(a.test, models, a.run_id)
        elif st == "score":
            score(a.test, models, a.run_id, a.include_decima)
        elif st == "plot":
            from workstreams.black_physics.ordinal_density import plots

            plots.plot(a.test, models, a.run_id)
        elif st == "scales":
            from workstreams.black_physics.ordinal_density import scales

            scales.run(a.test, models, a.run_id)
    return 0
