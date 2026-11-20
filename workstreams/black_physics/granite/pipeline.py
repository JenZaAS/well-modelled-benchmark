"""The experiment stages. Run from the repository root with the Granite environment:

    python -m workstreams.black_physics.granite prepare     # A-D split + gap indices (test wells stay sealed)
    python -m workstreams.black_physics.granite train       # both head fine-tunes (GPU time box), B selection
    python -m workstreams.black_physics.granite select      # four rows on B; choose the decision reconstruction
    python -m workstreams.black_physics.granite calibrate   # 50 % / 90 % bands on D for every row
    python -m workstreams.black_physics.granite decide      # decision fits on C, recalibration on D, freeze manifest
    python -m workstreams.black_physics.granite final       # sealed test gaps: keyed predictions + decisions
    python -m workstreams.black_physics.granite report      # tables, paired bootstrap, figures (local)
    python -m workstreams.black_physics.granite all         # prepare .. decide (everything before the test)

Artifacts live in `<work_root>/black_granite/` (never committed). Per-well records stay there.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from workstreams.black_physics.granite import artifacts as ART
from workstreams.black_physics.granite import calibrate as CAL
from workstreams.black_physics.granite import config as C
from workstreams.black_physics.granite import decision as DEC
from workstreams.black_physics.granite import features as F
from workstreams.black_physics.granite import gaps as G

VARIANTS = [C.VARIANT_IDS[k] for k in (("ttm", False), ("ttm", True), ("tspulse", False), ("tspulse", True))]
_WELLS: dict = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(C.work_dir() / "pipeline.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def path(*parts) -> Path:
    p = C.work_dir().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def git_head() -> dict:
    try:
        full = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[3]).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "workstreams/black_physics"], capture_output=True,
                               text=True, cwd=Path(__file__).resolve().parents[3]).stdout.strip()
        return {"commit": full, "black_physics_dirty": bool(dirty)}
    except Exception:  # noqa: BLE001  (provenance only)
        return {"commit": None}


def official():
    if "w" not in _WELLS:
        from wmb import paths, splits
        from wmb.datasets import load_force2020

        wells = load_force2020(paths.get_dataset_path(C.DATASET))
        _WELLS["w"] = splits.official_split(wells)
    return _WELLS["w"]


def train_by_name() -> dict:
    return {w.name: w for w in official()[0]}


# ------------------------------------------------------------------------------------------------ prepare
def prepare() -> dict:
    tr, te = official()
    ident = ART.official_identity(tr)
    groups, gaps = ident["groups"], ident["gaps"]
    test_names = {w.name for w in te}
    for g, names in groups.items():
        leak = test_names & set(names)
        assert not leak, f"test wells in split {g}: {sorted(leak)}"
    G.assert_disjoint(groups)
    spacing = [g.spacing_m for v in gaps.values() for g in v]
    rec = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_head(), "config": C.CONFIG.as_dict(),
        "identity": ART.identity_record(ident), "groups": groups,
        "eligible_wells": {g: sorted({x.well for x in gaps[g]}) for g in G.GROUPS},
        "n_wells": {g: len(groups[g]) for g in G.GROUPS},
        "n_eligible": {g: len({x.well for x in gaps[g]}) for g in G.GROUPS},
        "n_gaps": {g: len(gaps[g]) for g in G.GROUPS},
        "gap_samples": {g: [int(min(x.n for x in gaps[g])), int(np.median([x.n for x in gaps[g]])),
                            int(max(x.n for x in gaps[g]))] if gaps[g] else None for g in G.GROUPS},
        "spacing_m": [float(np.min(spacing)), float(np.median(spacing)), float(np.max(spacing))],
        "n_test_wells_sealed": len(te),
        "gaps": {g: [x.as_dict() for x in gaps[g]] for g in G.GROUPS},
    }
    save_json(path("prepared.json"), rec)
    log(f"prepare: wells {rec['n_wells']}, eligible {rec['n_eligible']}, gaps {rec['n_gaps']}, "
        f"samples/gap {rec['gap_samples']}, spacing {rec['spacing_m']}")
    return rec


def prepared() -> dict:
    p = path("prepared.json")
    if not p.is_file():
        return prepare()
    return load_json(p)


def gaps_of(group: str) -> list[G.Gap]:
    return [G.Gap(**{k: v for k, v in d.items() if k not in ("n", "gap_id")}) for d in prepared()["gaps"][group]]


def dev_tasks(group: str) -> list:
    by = train_by_name()
    frames: dict = {}
    out = []
    for g in gaps_of(group):
        if g.well not in frames:
            frames[g.well] = G.inputs_frame(by[g.well], C.TARGET)
        out.append(G.gap_task(by[g.well], g, frame=frames[g.well], test=f"dev_{group}"))
    return out


def truth_for(task, wells: dict) -> np.ndarray:
    return np.asarray(wells[task.well_id].curves[C.TARGET], float)


# ------------------------------------------------------------------------------------------------ scoring
def gap_rows(tasks, preds_p50, pieces_list, wells: dict, tol: float = C.CONFIG.tolerance_g_cm3) -> list[dict]:
    """One row per task (= one gap): errors on masked, measured depths; label = mean |error| <= tol (complete only)."""
    rows = []
    for t, p50, pieces in zip(tasks, preds_p50, pieces_list):
        m = np.asarray(t.predict_mask, bool)
        y = truth_for(t, wells)[m]
        p = np.asarray(p50, float)[m]
        meas = np.isfinite(y)
        fin = meas & np.isfinite(p)
        err = np.abs(p[fin] - y[fin])
        idx = np.flatnonzero(m)
        complete = bool(fin.sum() == meas.sum() and meas.any())
        mae = float(err.mean()) if len(err) else float("nan")
        rows.append({"key": t.meta.get("gap_id") or f"{t.well_id}|{idx[0]}:{idx[-1] + 1}", "well": t.well_id,
                     "start_md": round(float(t.depth[idx[0]]), 4), "end_md": round(float(t.depth[idx[-1]]), 4),
                     "n": int(m.sum()), "n_measured": int(meas.sum()), "n_pred": int(fin.sum()),
                     "status": ",".join(sorted({pc.status for pc in pieces})) if pieces else "none",
                     "complete": complete, "mae": mae, "abs_err_sum": float(err.sum()),
                     "sq_err_sum": float((err ** 2).sum()),
                     "label": (int(mae <= tol) if complete else None)})
    return rows


def mean_well_mae(rows) -> float:
    by: dict = {}
    for r in rows:
        s = by.setdefault(r["well"], [0.0, 0])
        s[0] += r["abs_err_sum"]
        s[1] += r["n_pred"]
    vals = [a / n for a, n in by.values() if n]
    return float(np.mean(vals)) if vals else float("nan")


def summary(rows) -> dict:
    n = sum(r["n_pred"] for r in rows)
    return {"gaps": len(rows), "complete_gaps": sum(r["complete"] for r in rows),
            "completeness": sum(r["complete"] for r in rows) / max(len(rows), 1),
            "fill_fraction": n / max(sum(r["n_measured"] for r in rows), 1),
            "pooled_MAE": sum(r["abs_err_sum"] for r in rows) / n if n else float("nan"),
            "pooled_RMSE": float(np.sqrt(sum(r["sq_err_sum"] for r in rows) / n)) if n else float("nan"),
            "mean_well_MAE": mean_well_mae(rows), "wells": len({r["well"] for r in rows}),
            "status_counts": _count(r["status"] for r in rows)}


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# ------------------------------------------------------------------------------------------------ variants
def head_file(arm: str) -> Path:
    return path("checkpoints", f"{arm}_tuned_best.pt")


def make_variant(name: str, device=None):
    from workstreams.black_physics.granite.backends import make_backend

    arm, tuned = ART.NAME_TO_VARIANT[name]
    head = None
    if tuned:
        import torch

        if not head_file(arm).is_file():
            raise FileNotFoundError(f"{head_file(arm)} missing: run the train stage for {arm}")
        head = torch.load(head_file(arm), map_location="cpu", weights_only=True)
    return make_backend(arm, device, head)


def run_rows(name: str, tasks, wells, backend=None, warm: bool = True) -> tuple[list[dict], list, dict]:
    from workstreams.black_physics.granite.reconstruct import make_reconstructor

    t0 = time.perf_counter()
    backend = backend or make_variant(name)
    load_s = time.perf_counter() - t0
    rec = make_reconstructor(backend)
    t0 = time.perf_counter()
    pieces = rec.run(tasks)
    first_s = time.perf_counter() - t0
    warm_s = None
    if warm:
        t0 = time.perf_counter()
        pieces = rec.run(tasks)
        warm_s = time.perf_counter() - t0
    p50 = []
    for t, ps in zip(tasks, pieces):
        a = np.full(len(t.depth), np.nan)
        for pc in ps:
            a[pc.a:pc.b] = pc.p50
        p50.append(a)
    rows = gap_rows(tasks, p50, pieces, wells)
    timing = {"load_s": round(load_s, 3), "first_inference_s": round(first_s, 3),
              "warm_inference_s": None if warm_s is None else round(warm_s, 3), "n_tasks": len(tasks)}
    return rows, pieces, timing


# ------------------------------------------------------------------------------------------------ train
def train(arms=("ttm", "tspulse")) -> dict:
    import torch

    from workstreams.black_physics.granite import train as T
    from workstreams.black_physics.granite.backends import make_backend, versions
    from workstreams.black_physics.granite.reconstruct import make_reconstructor

    by = train_by_name()
    gaps_a = gaps_of("A")
    b_tasks = dev_tasks("B")
    out = {}
    for arm in arms:
        ex = (T.ttm_examples if arm == "ttm" else T.tspulse_examples)(by, gaps_a)

        def eval_for(backend):
            rec = make_reconstructor(backend)

            def ev():
                pieces = rec.run(b_tasks)
                p50 = []
                for t, ps in zip(b_tasks, pieces):
                    a = np.full(len(t.depth), np.nan)
                    for pc in ps:
                        a[pc.a:pc.b] = pc.p50
                    p50.append(a)
                return mean_well_mae(gap_rows(b_tasks, p50, pieces, by))
            return ev

        t_load = time.perf_counter()
        backend, info = T.train_with_fallbacks(lambda dev, arm=arm: make_backend(arm, dev), ex, eval_for, log=log)
        info["first_load_s"] = round(time.perf_counter() - t_load - info["train_minutes"] * 60, 2)
        torch.save(info["best_state"], head_file(arm))
        info.update(head_file=str(head_file(arm).relative_to(C.work_dir())), head_sha256=ART.file_sha256(head_file(arm)),
                    versions=versions(), config_hash=C.config_hash(), git=git_head())
        path(f"train_{arm}.json").write_text(T.history_json(info), encoding="utf-8")
        log(f"train {arm}: tuned checkpoint = update {info['best_update']} B {info['best_metric']:.5f} "
            f"(step 0 {info['step0_metric']:.5f}{', step 0 won on B' if info['step0_won'] else ''}); "
            f"{info['updates_run']} updates in {info['train_minutes']:.2f} min; stop: {info['stop_reason']}")
        out[arm] = {k: v for k, v in info.items() if k != "best_state"}
        del backend
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------ select
def select() -> dict:
    by = train_by_name()
    tasks = dev_tasks("B")
    res = {}
    for name in VARIANTS:
        rows, _, timing = run_rows(name, tasks, by)
        res[name] = {"summary": summary(rows), "timing": timing}
        save_json(path("dev", f"B_{name}.json"), rows)
        log(f"select B {name}: {res[name]['summary']['mean_well_MAE']:.5f} mean well MAE, "
            f"completeness {res[name]['summary']['completeness']:.3f}, warm {timing['warm_inference_s']} s")
    full = [n for n in VARIANTS if res[n]["summary"]["completeness"] >= 1.0]
    rule = "lowest B mean well-level MAE among rows with full eligible-gap predictions"
    pool = full
    if not full:
        best_c = max(res[n]["summary"]["completeness"] for n in VARIANTS)
        pool = [n for n in VARIANTS if res[n]["summary"]["completeness"] == best_c]
        rule = "no row covered every B gap: highest completeness, then lowest B mean well-level MAE"

    def order(n):
        arm, tuned = ART.NAME_TO_VARIANT[n]
        return (round(res[n]["summary"]["mean_well_MAE"], 12), tuned, res[n]["timing"]["warm_inference_s"] or 0.0)

    chosen = min(pool, key=order)
    sel = {"selected": chosen, "rule": rule + "; exact ties: frozen first, then faster warmed inference",
           "rows": res, "git": git_head()}
    save_json(path("selection.json"), sel)
    log(f"select: {chosen} ({rule})")
    return sel


# ------------------------------------------------------------------------------------------------ calibrate
def calibrate() -> dict:
    by = train_by_name()
    tasks = dev_tasks("D")
    out = {}
    for name in VARIANTS:
        arm, _ = ART.NAME_TO_VARIANT[name]
        rows, pieces, timing = run_rows(name, tasks, by, warm=False)
        recs = [(t.well_id, ps[0], truth_for(t, by)[ps[0].a:ps[0].b]) for t, ps in zip(tasks, pieces) if ps]
        cal = CAL.fit(arm, recs, "D")
        raw = raw_coverage(arm, recs)
        out[name] = {"calibrator": cal.as_dict(), "raw_80": raw, "summary": summary(rows)}
        save_json(path("dev", f"D_{name}.json"), rows)
        log(f"calibrate D {name}: k50 {cal.k50:.4f} k90 {cal.k90:.4f} on {cal.n_wells} wells / {cal.n_samples} samples; "
            f"raw 80% coverage {raw}")
    save_json(path("calibration.json"), {"variants": out, "git": git_head()})
    return out


def raw_coverage(arm: str, recs) -> dict | None:
    """Coverage of the raw TTM P10-P90 band (nominal 80 %), pooled and equal-well-weighted."""
    if arm != "ttm":
        return None
    hit, per = [], {}
    for well, pc, y in recs:
        if pc.q_raw is None:
            continue
        ok = np.isfinite(y)
        h = (y[ok] >= pc.raw_lo[ok]) & (y[ok] <= pc.raw_hi[ok])
        hit.append(h)
        per.setdefault(well, []).append(h)
    if not hit:
        return None
    pooled = float(np.concatenate(hit).mean())
    by_well = float(np.mean([np.concatenate(v).mean() for v in per.values()]))
    width = float(np.mean(np.concatenate([pc.raw_hi - pc.raw_lo for _, pc, _ in recs if pc.q_raw is not None])))
    return {"nominal": 0.8, "pooled": pooled, "well_weighted": by_well, "mean_width": width}


# ------------------------------------------------------------------------------------------------ decide
def decima_python() -> Path:
    return Path(os.environ.get("WMB_DECIMA_PYTHON") or
                Path(os.environ.get("LOCALAPPDATA", "")) / "wmb-envs" / "decima-cpu" / "Scripts" / "python.exe")


def run_decima(rows: list[dict], tag: str) -> tuple[dict, dict | None]:
    """{key: {p_pass, logit}} from the Decima environment, or ({}, None) with a logged reason if unavailable."""
    py = decima_python()
    if not py.is_file():
        log(f"decima: environment not found at {py}; Decima rows marked unavailable")
        return {}, None
    inp, out = path("decision", f"decima_in_{tag}.jsonl"), path("decision", f"decima_out_{tag}.jsonl")
    inp.write_text("\n".join(json.dumps({"key": r["key"], "state": r["state"]}) for r in rows) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parent / "decima_score.py"
    cmd = [str(py), "-I", str(script), str(inp), str(out), "--cache", str(C.hf_home() / "hub")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"decima: failed ({res.stderr.strip()[-400:]}); Decima rows marked unavailable")
        return {}, None
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    meta = lines[0]["_meta"]
    return {x["key"]: x for x in lines[1:]}, meta


def gardner_on(task) -> np.ndarray:
    """black_gardner_fixed's estimate on the task's depths (published equation; no fitted parameter used)."""
    from workstreams.black_physics.models.gardner.wrapper import GardnerFixed

    return np.asarray(GardnerFixed().predict(task.to_well(), C.TARGET).mean, float)


def decision_records(name: str, tasks, wells: dict, pieces=None) -> list[dict]:
    arm, _ = ART.NAME_TO_VARIANT[name]
    if pieces is None:
        _, pieces, _ = run_rows(name, tasks, wells, warm=False)
    p50 = []
    for t, ps in zip(tasks, pieces):
        a = np.full(len(t.depth), np.nan)
        for pc in ps:
            a[pc.a:pc.b] = pc.p50
        p50.append(a)
    rows = gap_rows(tasks, p50, pieces, wells)
    out = []
    for t, ps, r in zip(tasks, pieces, rows):
        pc = ps[0]
        g = gardner_on(t)[pc.a:pc.b]
        f = F.gap_features(arm, pc, g)
        f["complete"] = f["complete"] and r["complete"]
        out.append({**r, **f, "state": F.state_text(arm, f, C.CONFIG.gap_length_m)})
    return out


METHODS_RAW = ("prevalence", "width_only", "numeric", "numeric_gardner", "decima_raw")
METHODS_CAL = ("numeric_cal", "numeric_gardner_cal", "decima_cal")


def apply_methods(rows: list[dict], rules: dict, recal: dict, decima: dict) -> dict[str, np.ndarray]:
    out = {}
    for k in ("prevalence", "width_only", "numeric", "numeric_gardner"):
        out[k] = rules[k].proba(rows)
    for k in ("numeric", "numeric_gardner"):
        out[f"{k}_cal"] = recal[k].apply(rules[k].logit(rows))
    if decima:
        z = np.array([decima[r["key"]]["logit"] if r["key"] in decima else np.nan for r in rows])
        out["decima_raw"] = np.array([decima[r["key"]]["p_pass"] if r["key"] in decima else np.nan for r in rows])
        if "decima" in recal:
            out["decima_cal"] = recal["decima"].apply(z)
    return out


def decide() -> dict:
    cfg = C.CONFIG
    by = train_by_name()
    sel = load_json(path("selection.json"))
    name = sel["selected"]
    arm, _ = ART.NAME_TO_VARIANT[name]
    rows = {g: decision_records(name, dev_tasks(g), by) for g in ("C", "D")}
    width_feat = ["width"]
    rules = {"prevalence": DEC.fit_prevalence(rows["C"]),
             "width_only": DEC.fit_rule("width_only", rows["C"], width_feat, cfg.l2, cfg.prevalence_smoothing),
             "numeric": DEC.fit_rule("numeric", rows["C"], F.BASE, cfg.l2, cfg.prevalence_smoothing),
             "numeric_gardner": DEC.fit_rule("numeric_gardner", rows["C"], F.WITH_GARDNER, cfg.l2,
                                             cfg.prevalence_smoothing)}
    decima = {}
    meta = None
    for g in ("C", "D"):
        got, meta_g = run_decima(rows[g], f"dev_{g}")
        decima.update(got)
        meta = meta or meta_g
    D = DEC.labelled(rows["D"])
    yD = np.array([r["label"] for r in D], float)
    wD = DEC.well_weights([r["well"] for r in D])
    recal = {k: DEC.fit_recalibration(rules[k].logit(D), yD, wD, cfg.recal_l2, cfg.prevalence_smoothing)
             for k in ("numeric", "numeric_gardner")}
    if decima:
        zD = np.array([decima[r["key"]]["logit"] for r in D])
        recal["decima"] = DEC.fit_recalibration(zD, yD, wD, cfg.recal_l2, cfg.prevalence_smoothing)
    dev_scores = {}
    for g in ("C", "D"):
        probs = apply_methods(rows[g], rules, recal, decima)
        dev_scores[g] = {k: DEC.decision_metrics(v, [r["label"] for r in rows[g]], [r["complete"] for r in rows[g]],
                                                 cfg.accept_threshold) for k, v in probs.items()}
        for i, r in enumerate(rows[g]):
            r["p"] = {k: float(v[i]) for k, v in probs.items()}
    for g in ("C", "D"):
        save_json(path("decision", f"rows_{g}.json"), rows[g])
    dec = {"selected": name, "arm": arm, "schema": F.schema_doc(arm), "rules": {k: v.as_dict() for k, v in rules.items()},
           "recalibration": {k: v.as_dict() for k, v in recal.items()}, "decima": meta,
           "decima_question": None if meta is None else meta.get("question"),
           "dev_scores": dev_scores, "n": {g: len(rows[g]) for g in rows},
           "n_labelled": {g: len(DEC.labelled(rows[g])) for g in rows},
           "pass_rate": {g: float(np.mean([r["label"] for r in DEC.labelled(rows[g])])) if DEC.labelled(rows[g]) else None
                         for g in rows}}
    save_json(path("decision.json"), dec)
    for k, v in rules.items():
        log(f"decide: {k} {v.note or ''} cols={None if v.prep is None else v.prep.columns()} beta={v.beta}")
    log(f"decide: C pass rate {dec['pass_rate']['C']}, D pass rate {dec['pass_rate']['D']}; recal {dec['recalibration']}")
    freeze()
    return dec


# ------------------------------------------------------------------------------------------------ freeze
def freeze() -> dict:
    from workstreams.black_physics.granite.backends import versions

    tr, _ = official()
    ident = ART.official_identity(tr)
    cal = load_json(path("calibration.json"))["variants"]
    sel = load_json(path("selection.json"))
    variants = {}
    for name in VARIANTS:
        arm, tuned = ART.NAME_TO_VARIANT[name]
        rel = str(head_file(arm).relative_to(C.work_dir())) if tuned else None
        variants[name] = {"arm": arm, "tuned": tuned,
                          "weights": f"{C.TTM_REPO}@{C.TTM_REVISION}" if arm == "ttm" else f"{C.TSPULSE_REPO}@{C.TSPULSE_REVISION}",
                          "head": rel, "head_sha256": ART.file_sha256(head_file(arm)) if tuned else None,
                          "calibrator": cal[name]["calibrator"]}
    m = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "identity": ART.identity_record(ident),
         "config": C.CONFIG.as_dict(), "variants": variants, "selected": sel["selected"],
         "decision_file": "decision.json", "versions": versions(), "git": git_head()}
    save_json(ART.manifest_path(), m)
    log(f"freeze: manifest written ({ART.manifest_path()}), selected {sel['selected']}")
    return m


# ------------------------------------------------------------------------------------------------ final
def final() -> dict:
    """Score the sealed test gaps once with frozen choices: all rows + decisions, keyed, full precision."""
    from benchmark import test_def as td
    from workstreams.black_physics.models.gardner.wrapper import GardnerFixed
    from workstreams.black_physics.models.granite import wrapper as W

    cfg = C.CONFIG
    tr, te = official()
    tdef = td.load_def(C.TEST_NAME)
    tasks = td.build_def_tasks(tdef, tr, te)
    wells = {w.name: w for w in te}
    for t in tasks:
        idx = np.flatnonzero(t.predict_mask)
        t.meta["gap_id"] = f"{t.well_id}|{idx[0]}:{idx[-1] + 1}"
    manifest = ART.read_manifest()
    preds: dict[str, dict] = {}
    rows: dict[str, list] = {}
    timing: dict[str, dict] = {}
    pieces_by: dict = {}
    for cls in W.GRANITE:
        m = cls()
        t0 = time.perf_counter()
        m.fit(tr, C.TARGET)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ps, pcs = m.predict_with_pieces(tasks)
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        ps, pcs = m.predict_with_pieces(tasks)
        warm = time.perf_counter() - t0
        preds[m.name] = {"p50": [p.p50 for p in ps], "p05": [p.p05 for p in ps], "p95": [p.p95 for p in ps],
                         "p25": [p.quantiles[0.25] for p in ps], "p75": [p.quantiles[0.75] for p in ps],
                         "raw_lo": [_piece_arr(t, pc, "raw_lo") for t, pc in zip(tasks, pcs)],
                         "raw_hi": [_piece_arr(t, pc, "raw_hi") for t, pc in zip(tasks, pcs)]}
        rows[m.name] = gap_rows(tasks, preds[m.name]["p50"], pcs, wells)
        timing[m.name] = {"fit_s": round(fit_s, 3), "first_inference_s": round(first, 3), "warm_inference_s": round(warm, 3),
                          "provenance": m.provenance}
        pieces_by[m.name] = pcs
        log(f"final {m.name}: {summary(rows[m.name])['pooled_MAE']:.5f} pooled MAE, warm {warm:.3f} s")
        del m
    others = {"black_linear_gap": W.LinearGap(), "black_gardner_fixed": GardnerFixed(),
              "black_gardner_flank_resid": W.GardnerFlankResidual()}
    for name, m in others.items():
        m.fit(tr, C.TARGET)
        t0 = time.perf_counter()
        ps = [m.predict_task(t) for t in tasks]
        timing[name] = {"warm_inference_s": round(time.perf_counter() - t0, 4)}
        preds[name] = {"p50": [np.asarray(p.mean, float) for p in ps], "p05": [None] * len(ps), "p95": [None] * len(ps)}
        if getattr(ps[0], "std", None) is not None:
            from wmb.scoring import gaussian_interval
            lo, hi = zip(*[gaussian_interval(p.mean, p.std, 0.90) for p in ps])
            preds[name]["p05"], preds[name]["p95"] = list(lo), list(hi)
            lo, hi = zip(*[gaussian_interval(p.mean, p.std, 0.50) for p in ps])
            preds[name]["p25"], preds[name]["p75"] = list(lo), list(hi)
        rows[name] = gap_rows(tasks, preds[name]["p50"], [[] for _ in tasks], wells)
        log(f"final {name}: {summary(rows[name])['pooled_MAE']:.5f} pooled MAE")
    # full-precision keyed predictions (gap depths only)
    keyed = {}
    for name, d in preds.items():
        for i, t in enumerate(tasks):
            m = t.predict_mask
            for k, v in d.items():
                if v[i] is not None:
                    keyed[f"{name}|{t.meta['gap_id']}|{k}"] = np.asarray(v[i], float)[m]
    for i, t in enumerate(tasks):
        keyed[f"truth|{t.meta['gap_id']}"] = truth_for(t, wells)[t.predict_mask]
        keyed[f"depth|{t.meta['gap_id']}"] = np.asarray(t.depth, float)[t.predict_mask]
    np.savez_compressed(path("final", "predictions.npz"), **keyed)
    # decisions on the selected reconstruction
    dec = load_json(path("decision.json"))
    name = dec["selected"]
    drows = decision_records(name, tasks, wells, pieces_by[name])
    rules = {k: DEC.LogisticRule.from_dict(v) for k, v in dec["rules"].items()}
    recal = {k: DEC.Recalibration(**{kk: vv for kk, vv in v.items()}) for k, v in dec["recalibration"].items()}
    decima, meta = run_decima(drows, "test")
    probs = apply_methods(drows, rules, recal, decima)
    head_sha = manifest["variants"][name]["head_sha256"]
    sidecar = []
    for i, r in enumerate(drows):
        sidecar.append({"test": C.TEST_NAME, "well": r["well"], "start_md": r["start_md"], "end_md": r["end_md"],
                        "gap_id": r["key"], "reconstruction": name,
                        "checkpoint": head_sha or manifest["variants"][name]["weights"],
                        "features": {k: r[k] for k, *_ in F.SCHEMA}, "state": r["state"], "complete": r["complete"],
                        "p_pass": {k: float(v[i]) for k, v in probs.items()},
                        "mae": r["mae"], "label": r["label"], "status": r["status"]})
    save_json(path("final", "decision_sidecar.json"), sidecar)
    scores = {k: DEC.decision_metrics(v, [r["label"] for r in drows], [r["complete"] for r in drows],
                                      cfg.accept_threshold) for k, v in probs.items()}
    curves = {k: DEC.risk_coverage(v, [r["label"] for r in drows], [r["complete"] for r in drows]) for k, v in probs.items()}
    res = {"tasks": len(tasks), "rows": rows, "timing": timing, "decision_scores": scores, "risk_coverage": curves,
           "decima_meta": meta, "selected": name, "git": git_head(), "manifest_frozen_at": manifest["frozen_at"]}
    save_json(path("final", "final.json"), res)
    for k, s in scores.items():
        log(f"final decision {k}: brier {s['brier']}, accepted {s['accepted']}/{s['n_gaps']}, false accepts {s['false_accepts']}")
    return res


def _piece_arr(task, pieces, attr) -> np.ndarray | None:
    a = np.full(len(task.depth), np.nan)
    any_ = False
    for pc in pieces:
        v = getattr(pc, attr)
        if v is not None:
            a[pc.a:pc.b] = v
            any_ = True
    return a if any_ else None


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Granite 50 m density-gap experiment")
    ap.add_argument("stage", choices=["prepare", "train", "select", "calibrate", "decide", "freeze", "final", "report",
                                      "all", "fetch"])
    ap.add_argument("--arm", choices=["ttm", "tspulse"], help="train only this arm")
    a = ap.parse_args(argv)
    if a.stage == "fetch":
        from workstreams.black_physics.granite.backends import fetch_weights
        print(json.dumps(fetch_weights(), indent=2))
        return 0
    if a.stage in ("prepare", "all"):
        prepare()
    if a.stage in ("train", "all"):
        train((a.arm,) if a.arm else ("ttm", "tspulse"))
    if a.stage in ("select", "all"):
        select()
    if a.stage in ("calibrate", "all"):
        calibrate()
    if a.stage in ("decide", "all"):
        decide()
    if a.stage == "freeze":
        freeze()
    if a.stage == "final":
        final()
    if a.stage == "report":
        from workstreams.black_physics.granite import report
        report.build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
