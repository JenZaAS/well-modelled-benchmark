"""Stronger fine-tune (step-8 extension): broader unfreezing, more A gaps, a real schedule. Selected on B only.

The brief's head-only run (`train`) stopped after 75-125 updates (<5 s): patience 3 at 25-update evaluations
reacted to B noise, the TSPulse time head has 200 parameters, and TTM was trained per direction while it is
scored on the blended two-sided median. This stage fixes those, keeping everything else identical (A wells
for fitting, B for selection, D for bands, the same reconstructors and min-context rule):

- **Scope.** TTM: prediction heads (as before) or the whole model (3.16 M). TSPulse: decoder + heads or the whole
  model (1.07 M). Both fit easily on a 4 GB laptop GPU.
- **Data.** Up to 256 overlapping 50 m gaps per A well (the brief's 32 are a prefix of these), same placer.
- **TTM objective.** Both flanks go through the model in one batch; the deeper forecast is reversed and blended by
  depth fraction u exactly as at inference; pinball over the nine quantiles on the *blended* output (one-sided
  gaps use their single direction). TSPulse keeps the external MAE on the hidden samples.
- **Schedule.** AdamW, weight decay 0.01, batch 32, 100-update linear warm-up then cosine to 10 %, gradient
  clipping 1.0, evaluation on B every 100 updates, patience 10 evaluations, <= 4000 updates or 12 minutes.
- **Sweep.** Three (scope, learning rate) settings per model; the best B evaluation after >= 1 update wins.

    python -m workstreams.black_physics.granite finetune [--arm ttm|tspulse]     # sweep + checkpoint per arm
    python -m workstreams.black_physics.granite finetune_score                   # D bands, B and test rows

These rows were designed after the test results of the head-only run were known. Every choice here uses A, B
and D only; the test gaps are scored once, after the checkpoints and bands are saved.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from workstreams.black_physics.granite import calibrate as CAL
from workstreams.black_physics.granite import config as C
from workstreams.black_physics.granite import gaps as G
from workstreams.black_physics.granite.backends import TSPULSE_HEAD_KEYS, TTM_HEAD_KEYS, TTM_QUANTILES

SCOPES = {"ttm": {"head": TTM_HEAD_KEYS, "full": ("",)},
          "tspulse": {"head": TSPULSE_HEAD_KEYS, "decoder": ("decoder_with_head.",), "full": ("",)}}
SWEEP = {"ttm": [("head", 1e-3), ("full", 1e-4), ("full", 3e-5)],
         "tspulse": [("decoder", 1e-4), ("full", 1e-4), ("full", 3e-5)]}
FT_NAMES = {"ttm": "black_ttm_r3_ft", "tspulse": "black_tspulse_ft"}


@dataclass(frozen=True)
class FTConfig:
    gaps_per_well_A: int = 256
    batch: int = 32
    max_updates: int = 4000
    max_minutes: float = 12.0
    eval_every: int = 100
    patience: int = 10
    warmup: int = 100
    min_lr_frac: float = 0.10
    clip: float = 1.0
    weight_decay: float = 0.01
    seed: int = C.CONFIG.seed


FT = FTConfig()


# ------------------------------------------------------------------------------------------------ examples
@dataclass
class PairExample:
    """One A gap for TTM: either or both directed contexts, labels in increasing depth, blend weight u."""
    well: str
    up: np.ndarray | None          # shallow flank, oldest first
    dn: np.ndarray | None          # deeper flank, reversed (forecast runs upward)
    y: np.ndarray
    u: np.ndarray | None           # weight of the deeper forecast (two-sided only)


def a_gaps(by: dict, groups: dict, n_per_well: int = FT.gaps_per_well_A) -> list[G.Gap]:
    out = []
    for name in groups["A"]:
        w = by[name]
        if w.has(C.TARGET):
            out += G.index_gaps(name, w.depth, w.curves[C.TARGET], group="A", n_max=n_per_well, overlap=True,
                                seed=C.CONFIG.seed, length_m=C.CONFIG.gap_length_m)
    return out


def ttm_pair_examples(by: dict, gaps, cfg=C.CONFIG) -> list[PairExample]:
    from workstreams.black_physics.granite.reconstruct import _distance_weight

    out = []
    for g in gaps:
        if g.n > cfg.ttm_horizon:
            continue
        w = by[g.well]
        y = np.asarray(w.curves[C.TARGET], float)
        obs = y.copy()
        obs[g.start:g.end] = np.nan
        up = G.flank_run(obs, g.start, g.end, g.seg_start, g.seg_end, cfg.ttm_context, "above")
        dn = G.flank_run(obs, g.start, g.end, g.seg_start, g.seg_end, cfg.ttm_context, "below")
        use_up, use_dn = len(up) >= cfg.ttm_min_context, len(dn) >= cfg.ttm_min_context
        if not (use_up or use_dn):
            continue
        u = _distance_weight(w.depth, g.start, g.end, int(up[-1]), int(dn[0])) if use_up and use_dn else None
        out.append(PairExample(g.well, obs[up] if use_up else None, obs[dn][::-1].copy() if use_dn else None,
                               y[g.start:g.end].copy(), u))
    return out


def ttm_blended_loss(backend, batch: list[PairExample]):
    """Pinball over the nine quantiles on the blended two-sided forecast, equal weight per gap."""
    import torch

    ctx, where = [], []
    for e in batch:
        iu = id_ = None
        if e.up is not None:
            iu = len(ctx)
            ctx.append(e.up)
        if e.dn is not None:
            id_ = len(ctx)
            ctx.append(e.dn)
        where.append((iu, id_))
    past, mask = backend.batch(ctx)
    q = backend.forward(past, mask)                                      # [N, 9, 336]
    tau = torch.tensor(TTM_QUANTILES, dtype=q.dtype, device=q.device).view(-1, 1)
    losses = []
    for e, (iu, id_) in zip(batch, where):
        H = len(e.y)
        y = torch.as_tensor(e.y, dtype=q.dtype, device=q.device).view(1, -1)
        if iu is not None and id_ is not None:
            u = torch.as_tensor(e.u, dtype=q.dtype, device=q.device).view(1, -1)
            qq = (1 - u) * q[iu, :, :H] + u * q[id_, :, :H].flip(-1)
        elif iu is not None:
            qq = q[iu, :, :H]
        else:
            qq = q[id_, :, :H].flip(-1)
        d = y - qq
        losses.append(torch.maximum(tau * d, (tau - 1) * d).mean())
    return torch.stack(losses).mean()


def tspulse_examples(by: dict, gaps):
    from workstreams.black_physics.granite import train as T

    return T.tspulse_examples(by, gaps)


def tspulse_loss(backend, batch):
    from workstreams.black_physics.granite import train as T

    return T.tspulse_loss(backend, batch)


# ------------------------------------------------------------------------------------------------ training
def full_state(backend) -> dict:
    return {k: v.detach().cpu().clone() for k, v in backend.model.state_dict().items()}


def load_ft(arm: str, device=None, path=None):
    """Backend with a saved fine-tuned state (whole state_dict, buffers included)."""
    import torch

    from workstreams.black_physics.granite.backends import make_backend

    ck = torch.load(path or checkpoint(arm), map_location="cpu", weights_only=True)
    b = make_backend(arm, device)
    b.model.load_state_dict(ck["state"])
    b.model.eval()
    return b, ck["meta"]


def checkpoint(arm: str):
    return C.work_dir() / "checkpoints" / f"{arm}_ft_best.pt"


def _lr_factor(step: int, ftc: FTConfig) -> float:
    if step < ftc.warmup:
        return (step + 1) / ftc.warmup
    t = min(1.0, (step - ftc.warmup) / max(1, ftc.max_updates - ftc.warmup))
    return ftc.min_lr_frac + (1 - ftc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t))


def train_one(backend, examples, loss_fn, eval_fn, scope: str, lr: float, step0: float, ftc: FTConfig = FT,
              log=print) -> dict:
    import torch

    from workstreams.black_physics.granite import train as T

    torch.manual_seed(ftc.seed)
    backend.head_keys = SCOPES[backend.arm][scope]          # instance override: what set_trainable unfreezes
    names = backend.set_trainable()
    params = [p for p in backend.model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=ftc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, ftc))
    sampler = T.WellSampler(examples, ftc.seed)
    init = full_state(backend)

    def evaluate() -> float:
        backend.model.eval()
        try:
            with torch.no_grad():
                return float(eval_fn())
        finally:
            backend.set_trainable()

    best = {"update": None, "metric": float("inf"), "state": None}
    history = [{"update": 0, "minutes": 0.0, "train_loss": None, "B_metric": step0, "lr": 0.0}]
    stale, losses, stop = 0, [], "max_updates"
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    u = 0
    for u in range(1, ftc.max_updates + 1):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(backend, sampler.draw(ftc.batch))
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, ftc.clip))
        if not math.isfinite(gn):
            stop = "non-finite gradient"
            break
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        minutes = (time.perf_counter() - t0) / 60
        if u % ftc.eval_every == 0 or u == ftc.max_updates or minutes >= ftc.max_minutes:
            metric = evaluate()
            improved = metric < best["metric"] - 1e-9
            if improved:
                best = {"update": u, "metric": metric, "state": full_state(backend)}
                stale = 0
            else:
                stale += 1
            history.append({"update": u, "minutes": round(minutes, 3), "B_metric": metric,
                            "train_loss": float(np.mean(losses[-ftc.eval_every:])),
                            "lr": float(sched.get_last_lr()[0])})
            log(f"[ft {backend.arm} {scope} lr {lr:g}] update {u}: loss {history[-1]['train_loss']:.5f} "
                f"B {metric:.5f}{' *' if improved else ''} (step 0 {step0:.5f}; {minutes:.2f} min)")
            if stale >= ftc.patience:
                stop = f"no improvement in {ftc.patience} evaluations"
                break
            if minutes >= ftc.max_minutes:
                stop = f"time box {ftc.max_minutes} min"
                break
    minutes = (time.perf_counter() - t0) / 60
    info = {"scope": scope, "lr": lr, "trainable": len(names), "n_trainable": int(n_train),
            "n_examples": len(examples), "updates_run": u, "train_minutes": round(minutes, 3), "stop_reason": stop,
            "step0_metric": step0, "best_update": best["update"], "best_metric": best["metric"],
            "delta_vs_step0": best["metric"] - step0, "history": history,
            "peak_vram_reserved_mb": (round(torch.cuda.max_memory_reserved() / 2**20, 1)
                                      if backend.device.type == "cuda" else None)}
    if best["state"] is not None:
        info["max_abs_change"] = max(float((best["state"][k].float() - init[k].float()).abs().max())
                                     for k in init if init[k].is_floating_point())
    info["best_state"] = best["state"]
    return info


def b_eval_for(b_tasks, by):
    from workstreams.black_physics.granite.pipeline import gap_rows, mean_well_mae
    from workstreams.black_physics.granite.reconstruct import make_reconstructor

    def for_backend(backend):
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
    return for_backend


def sweep(arms=("ttm", "tspulse")) -> dict:
    import torch

    from workstreams.black_physics.granite import pipeline as P
    from workstreams.black_physics.granite.backends import make_backend, versions

    by = P.train_by_name()
    groups = P.prepared()["groups"]
    t = time.perf_counter()
    gaps = a_gaps(by, groups)
    P.log(f"finetune: {len(gaps)} A gaps ({FT.gaps_per_well_A} per well max) in {time.perf_counter() - t:.1f} s")
    b_tasks = P.dev_tasks("B")
    ev_for = b_eval_for(b_tasks, by)
    out = {}
    for arm in arms:
        examples = ttm_pair_examples(by, gaps) if arm == "ttm" else tspulse_examples(by, gaps)
        loss_fn = ttm_blended_loss if arm == "ttm" else tspulse_loss
        two = sum(1 for e in examples if getattr(e, "u", None) is not None) if arm == "ttm" else None
        P.log(f"finetune {arm}: {len(examples)} examples from {len({e.well for e in examples})} wells"
              + (f" ({two} two-sided)" if two is not None else ""))
        base = make_backend(arm, None)
        base.model.eval()
        step0 = float(ev_for(base)())
        del base
        runs, best = [], None
        for scope, lr in SWEEP[arm]:
            backend = make_backend(arm, None)
            info = train_one(backend, examples, loss_fn, ev_for(backend), scope, lr, step0, log=P.log)
            st = info.pop("best_state")
            runs.append(info)
            P.log(f"finetune {arm} {scope} lr {lr:g}: best B {info['best_metric']:.5f} at update "
                  f"{info['best_update']} (step 0 {step0:.5f}, delta {info['delta_vs_step0']:+.5f}); "
                  f"{info['updates_run']} updates, {info['train_minutes']:.1f} min, stop: {info['stop_reason']}")
            if st is not None and (best is None or info["best_metric"] < best[0]["best_metric"] - 1e-12):
                best = (info, st)
            del backend
            torch.cuda.empty_cache()
        info, st = best
        meta = {"arm": arm, "scope": info["scope"], "lr": info["lr"], "best_update": info["best_update"],
                "B_metric": info["best_metric"], "step0_metric": step0, "ft_config": asdict(FT),
                "config_hash": C.config_hash(), "weights": (f"{C.TTM_REPO}@{C.TTM_REVISION}" if arm == "ttm"
                                                           else f"{C.TSPULSE_REPO}@{C.TSPULSE_REVISION}")}
        checkpoint(arm).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": st, "meta": meta}, checkpoint(arm))
        from workstreams.black_physics.granite.artifacts import file_sha256

        rec = {"arm": arm, "selected": meta, "runs": runs, "n_examples": len(examples), "n_two_sided": two,
               "n_gaps_A": len(gaps), "checkpoint": str(checkpoint(arm).relative_to(C.work_dir())),
               "checkpoint_sha256": file_sha256(checkpoint(arm)), "versions": versions(), "git": P.git_head()}
        P.save_json(P.path(f"finetune_{arm}.json"), rec)
        P.log(f"finetune {arm}: selected {meta['scope']} lr {meta['lr']:g} (B {meta['B_metric']:.5f} vs step 0 "
              f"{step0:.5f})")
        out[arm] = rec
    return out


# ------------------------------------------------------------------------------------------------ scoring
def _arrays(tasks, pieces, cal):
    p = {k: [] for k in ("p50", "p05", "p25", "p75", "p95", "raw_lo", "raw_hi")}
    for t, ps in zip(tasks, pieces):
        n = len(t.depth)
        a = {k: np.full(n, np.nan) for k in p}
        has_raw = False
        for pc in ps:
            a["p50"][pc.a:pc.b] = pc.p50
            for k, v in cal.bands(pc).items():
                a[k][pc.a:pc.b] = v
            if pc.raw_lo is not None:
                a["raw_lo"][pc.a:pc.b], a["raw_hi"][pc.a:pc.b] = pc.raw_lo, pc.raw_hi
                has_raw = True
        for k in p:
            p[k].append(a[k] if (has_raw or k not in ("raw_lo", "raw_hi")) else None)
    return p


def score() -> dict:
    """For each saved checkpoint: D bands, B and D rows, then the sealed test gaps (keyed, full precision)."""
    import torch

    from benchmark import test_def as td
    from workstreams.black_physics.granite import pipeline as P
    from workstreams.black_physics.granite.reconstruct import make_reconstructor

    by = P.train_by_name()
    d_tasks, b_tasks = P.dev_tasks("D"), P.dev_tasks("B")
    tr, te = P.official()
    tasks = td.build_def_tasks(td.load_def(C.TEST_NAME), tr, te)
    wells = {w.name: w for w in te}
    for t in tasks:
        idx = np.flatnonzero(t.predict_mask)
        t.meta["gap_id"] = f"{t.well_id}|{idx[0]}:{idx[-1] + 1}"
    keyed, rows, info = {}, {}, {}
    for arm in ("ttm", "tspulse"):
        if not checkpoint(arm).is_file():
            continue
        name = FT_NAMES[arm]
        backend, meta = load_ft(arm)
        rec = make_reconstructor(backend)
        d_pieces = rec.run(d_tasks)
        cal = CAL.fit(arm, [(t.well_id, ps[0], P.truth_for(t, by)[ps[0].a:ps[0].b])
                            for t, ps in zip(d_tasks, d_pieces) if ps], "D")
        d_rows = P.gap_rows(d_tasks, _arrays(d_tasks, d_pieces, cal)["p50"], d_pieces, by)
        b_pieces = rec.run(b_tasks)
        b_rows = P.gap_rows(b_tasks, _arrays(b_tasks, b_pieces, cal)["p50"], b_pieces, by)
        P.save_json(P.path("dev", f"B_{name}.json"), b_rows)
        P.save_json(P.path("dev", f"D_{name}.json"), d_rows)
        t0 = time.perf_counter()
        pieces = rec.run(tasks)
        inf_s = time.perf_counter() - t0
        arr = _arrays(tasks, pieces, cal)
        rows[name] = P.gap_rows(tasks, arr["p50"], pieces, wells)
        for i, t in enumerate(tasks):
            for k, v in arr.items():
                if v[i] is not None:
                    keyed[f"{name}|{t.meta['gap_id']}|{k}"] = np.asarray(v[i], float)[t.predict_mask]
        info[name] = {"meta": meta, "calibrator": cal.as_dict(), "B": P.summary(b_rows), "D": P.summary(d_rows),
                      "inference_s": round(inf_s, 3)}
        P.log(f"finetune_score {name}: B {P.mean_well_mae(b_rows):.5f}, D {P.mean_well_mae(d_rows):.5f}, "
              f"test {P.mean_well_mae(rows[name]):.5f} mean well MAE (k50 {cal.k50:.3f}, k90 {cal.k90:.3f})")
        del backend
        torch.cuda.empty_cache()
    np.savez_compressed(P.path("final", "finetune_predictions.npz"), **keyed)
    from workstreams.black_physics.granite import artifacts as ART

    man = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "identity": ART.identity_record(ART.official_identity(tr)),
           "ft_config": asdict(FT), "git": P.git_head(),
           "variants": {FT_NAMES[a]: {"checkpoint": str(checkpoint(a).relative_to(C.work_dir())),
                                      "checkpoint_sha256": ART.file_sha256(checkpoint(a)),
                                      "calibrator": info[FT_NAMES[a]]["calibrator"], "meta": info[FT_NAMES[a]]["meta"]}
                        for a in ("ttm", "tspulse") if FT_NAMES[a] in info}}
    P.save_json(ART.ft_manifest_path(), man)
    res = {"rows": rows, "variants": info, "git": P.git_head(), "ft_config": asdict(FT)}
    P.save_json(P.path("final", "finetune.json"), res)
    return res


def history_json(rec: dict) -> str:
    return json.dumps(rec, indent=2, default=str)
