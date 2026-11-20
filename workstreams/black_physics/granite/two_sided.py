"""Two-sided TTM: a learned bridge between the two directional forecasts (step-8 extension).

Why. TTM fills a gap with two one-sided forecasts (down from the shallow flank, up from the reversed deep flank)
and a fixed blend by depth fraction u. Neither forecast sees the other flank, so fine-tuning TTM can only make it a
better extrapolator: on the D and test gaps the stronger fine-tune left two-sided gaps unchanged (+0.5 %) and
helped only one-sided ones (-7 %). TSPulse, which sees both flanks in one window, improved 7 %.

What. A small network (the *bridge*) replaces the fixed blend on two-sided gaps. At each depth of the gap it reads:
- u, the fixed blend weight (position in the gap);
- both forecasts' medians and P10-P90 widths;
- the nearest measured density on each side;
- each forecast's miss at the far flank: the downward forecast's last value against the first measured value
  below, and the upward forecast's last value against the last measured value above.

It returns a correction to the mixing weight (on the logit scale) and an additive shift, applied to all nine
quantiles. Density inputs are differences from the fixed blend, so the bridge is level-free. Its last layer
starts at zero, so at step 0 the output is exactly the fixed blend. One-sided gaps keep their single direction.

Training uses the same A gaps, pinball loss, schedule, B selection and D bands as `finetune.py`. Settings:
- `bridge`: the bridge alone, on pretrained TTM;
- `bridge_on_ft`: the bridge alone, on the stronger fine-tune's TTM checkpoint;
- `heads+bridge`: TTM's prediction heads and the bridge, trained together from pretrained.

The best B evaluation after at least one update wins; the test gaps are scored once, after the checkpoint and
bands are saved. Nothing here changes the files hashed into the frozen manifest; the existing rows are untouched.

    <granite-python> -m workstreams.black_physics.granite.two_sided train     # sweep on A, select on B
    <granite-python> -m workstreams.black_physics.granite.two_sided score     # D bands, B/D rows, test once

Outputs (local): checkpoints/ttm_twosided_best.pt, twosided_ttm.json, dev/{B,D}_black_ttm_r3_twosided.json,
final/twosided.json, final/twosided_predictions.npz, manifest_twosided.json.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np

from workstreams.black_physics.granite import calibrate as CAL
from workstreams.black_physics.granite import config as C
from workstreams.black_physics.granite.backends import TTM_HEAD_KEYS, TTM_QUANTILES

NAME = "black_ttm_r3_twosided"          # id inside the experiment's artifacts; station id red_ttm_r3_twosided
N_FEAT = 11
SCALE = 0.1                             # g/cm3: density differences are divided by this before the network
MED = TTM_QUANTILES.index(0.5)
SWEEP = [("bridge", 1e-3, 0.0), ("bridge_on_ft", 1e-3, 0.0), ("heads+bridge", 1e-3, 1e-3)]   # (setting, bridge lr, head lr)


def checkpoint():
    return C.work_dir() / "checkpoints" / "ttm_twosided_best.pt"


def manifest_path():
    return C.work_dir() / "manifest_twosided.json"


# ------------------------------------------------------------------------------------------------ the bridge
def make_bridge(hidden: int = 64):
    import torch
    from torch import nn

    class Bridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, 2))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, qa, qb, u, ya, yb):
            """qa, qb: [B, 9, H] in increasing depth; u: [B, H]; ya, yb: [B] nearest measured value above / below."""
            ma, mb = qa[:, MED], qb[:, MED]
            base = (1 - u) * ma + u * mb
            ra = (yb[:, None] - ma[:, -1:]).expand_as(u)        # downward forecast's miss at the deep flank
            rb = (ya[:, None] - mb[:, :1]).expand_as(u)         # upward forecast's miss at the shallow flank
            f = torch.stack([u, (ma - base) / SCALE, (mb - base) / SCALE, (ya[:, None] - base) / SCALE,
                             (yb[:, None] - base) / SCALE, ra / SCALE, rb / SCALE, ra * u / SCALE,
                             rb * (1 - u) / SCALE, (qa[:, -1] - qa[:, 0]) / SCALE, (qb[:, -1] - qb[:, 0]) / SCALE], -1)
            out = self.net(f)
            w = torch.sigmoid(torch.logit(u.clamp(1e-4, 1 - 1e-4)) + out[..., 0])
            return (1 - w)[:, None] * qa + w[:, None] * qb + SCALE * out[..., 1][:, None]

    return Bridge()


def apply_bridge(bridge, qa: np.ndarray, qb: np.ndarray, u: np.ndarray, ya: float, yb: float) -> np.ndarray:
    """NumPy front end for inference: [9, H] arrays in increasing depth -> bridged [9, H]."""
    import torch

    p = next(bridge.parameters())
    t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=p.device)   # noqa: E731
    with torch.no_grad():
        q = bridge(t(qa)[None], t(qb)[None], t(u)[None], t([ya]), t([yb]))
    return q[0].float().cpu().numpy().astype(float)


# ------------------------------------------------------------------------------------------------ inference
class TwoSidedReconstructor:
    """TTMReconstructor's geometry and one-sided fallbacks; two-sided pieces re-blended by the bridge."""

    arm = "ttm"

    def __init__(self, backend, bridge):
        from workstreams.black_physics.granite.reconstruct import TTMReconstructor

        self.backend, self.bridge = backend, bridge
        self.base = TTMReconstructor(backend.forecast)

    def run(self, tasks):
        from workstreams.black_physics.granite.reconstruct import _blank, _distance_weight

        results = self.base.run(tasks)
        jobs, contexts = [], []
        for ti, (t, pieces) in enumerate(zip(tasks, results)):
            obs = _blank(t)
            for pi, p in enumerate(pieces):
                if p.status == "two_sided":
                    jobs.append((ti, pi, obs))
                    contexts += [obs[p.ctx_above], obs[p.ctx_below][::-1]]
        if not jobs:
            return results
        q = self.backend.forecast(contexts)
        for k, (ti, pi, obs) in enumerate(jobs):
            p = results[ti][pi]
            depth = np.asarray(tasks[ti].depth, float)
            qa = np.asarray(q[2 * k][:, :p.n], float)
            qb = np.asarray(q[2 * k + 1][:, :p.n], float)[:, ::-1]
            u = _distance_weight(depth, p.a, p.b, int(p.ctx_above[-1]), int(p.ctx_below[0]))
            qq = apply_bridge(self.bridge, qa, qb, u, float(obs[p.ctx_above[-1]]), float(obs[p.ctx_below[0]]))
            p.q_raw = qq
            p.p50 = qq[MED].copy()
        return results


# ------------------------------------------------------------------------------------------------ training
def bridged_loss(backend, bridge, batch, train_ttm: bool):
    """Pinball over the nine quantiles on the bridged output; one-sided gaps use their single direction."""
    import contextlib

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
    with (contextlib.nullcontext() if train_ttm else torch.no_grad()):
        q = backend.forward(past, mask)                                    # [N, 9, 336]
    tau = torch.tensor(TTM_QUANTILES, dtype=q.dtype, device=q.device).view(-1, 1)
    losses = []
    for e, (iu, id_) in zip(batch, where):
        H = len(e.y)
        y = torch.as_tensor(e.y, dtype=q.dtype, device=q.device).view(1, -1)
        if iu is not None and id_ is not None:
            u = torch.as_tensor(e.u, dtype=q.dtype, device=q.device).view(1, -1)
            ya = torch.as_tensor([e.up[-1]], dtype=q.dtype, device=q.device)
            yb = torch.as_tensor([e.dn[-1]], dtype=q.dtype, device=q.device)
            qq = bridge(q[iu:iu + 1, :, :H], q[id_:id_ + 1, :, :H].flip(-1), u, ya, yb)[0]
        elif iu is not None:
            qq = q[iu, :, :H]
        else:
            qq = q[id_, :, :H].flip(-1)
        d = y - qq
        losses.append(torch.maximum(tau * d, (tau - 1) * d).mean())
    return torch.stack(losses).mean()


def b_eval(b_tasks, by, backend, bridge):
    from workstreams.black_physics.granite.pipeline import gap_rows, mean_well_mae

    pieces = TwoSidedReconstructor(backend, bridge).run(b_tasks)
    p50 = []
    for t, ps in zip(b_tasks, pieces):
        a = np.full(len(t.depth), np.nan)
        for pc in ps:
            a[pc.a:pc.b] = pc.p50
        p50.append(a)
    return mean_well_mae(gap_rows(b_tasks, p50, pieces, by))


def train_setting(setting: str, lr_bridge: float, lr_heads: float, examples, b_tasks, by, step0: float, log) -> dict:
    import torch

    from workstreams.black_physics.granite import finetune as FTN
    from workstreams.black_physics.granite import train as T
    from workstreams.black_physics.granite.backends import make_backend

    ftc = FTN.FT
    torch.manual_seed(ftc.seed)
    if setting == "bridge_on_ft":
        backend, _ = FTN.load_ft("ttm")
    else:
        backend = make_backend("ttm", None)
    train_ttm = setting == "heads+bridge"
    if train_ttm:
        backend.head_keys = TTM_HEAD_KEYS
        backend.set_trainable()
    else:
        backend.model.eval()
        for p in backend.model.parameters():
            p.requires_grad_(False)
        examples = [e for e in examples if e.u is not None]          # only two-sided gaps reach the bridge
    bridge = make_bridge().to(backend.device)
    groups = [{"params": list(bridge.parameters()), "lr": lr_bridge}]
    if train_ttm:
        groups.append({"params": [p for p in backend.model.parameters() if p.requires_grad], "lr": lr_heads})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=ftc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: FTN._lr_factor(s, ftc))
    sampler = T.WellSampler(examples, ftc.seed)

    def state():
        return {"ttm": FTN.full_state(backend), "bridge": {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}}

    def evaluate():
        backend.model.eval()
        bridge.eval()
        try:
            with torch.no_grad():
                return float(b_eval(b_tasks, by, backend, bridge))
        finally:
            bridge.train()
            if train_ttm:
                backend.set_trainable()

    s0 = evaluate()
    best = {"update": None, "metric": float("inf"), "state": None}
    history = [{"update": 0, "minutes": 0.0, "train_loss": None, "B_metric": s0, "lr": 0.0}]
    stale, losses, stop, u = 0, [], "max_updates", 0
    t0 = time.perf_counter()
    for u in range(1, ftc.max_updates + 1):
        opt.zero_grad(set_to_none=True)
        loss = bridged_loss(backend, bridge, sampler.draw(ftc.batch), train_ttm)
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
                best = {"update": u, "metric": metric, "state": state()}
                stale = 0
            else:
                stale += 1
            history.append({"update": u, "minutes": round(minutes, 3), "B_metric": metric,
                            "train_loss": float(np.mean(losses[-ftc.eval_every:])), "lr": float(sched.get_last_lr()[0])})
            log(f"[twosided {setting}] update {u}: loss {history[-1]['train_loss']:.5f} B {metric:.5f}"
                f"{' *' if improved else ''} (start {s0:.5f}, pretrained {step0:.5f}; {minutes:.2f} min)")
            if stale >= ftc.patience:
                stop = f"no improvement in {ftc.patience} evaluations"
                break
            if minutes >= ftc.max_minutes:
                stop = f"time box {ftc.max_minutes} min"
                break
    info = {"setting": setting, "lr_bridge": lr_bridge, "lr_heads": lr_heads, "n_examples": len(examples),
            "n_bridge_params": int(sum(p.numel() for p in bridge.parameters())),
            "n_ttm_trained": int(sum(p.numel() for p in backend.model.parameters() if p.requires_grad)) if train_ttm else 0,
            "updates_run": u, "train_minutes": round((time.perf_counter() - t0) / 60, 3), "stop_reason": stop,
            "start_metric": s0, "pretrained_metric": step0, "best_update": best["update"], "best_metric": best["metric"],
            "delta_vs_start": best["metric"] - s0, "delta_vs_pretrained": best["metric"] - step0, "history": history,
            "best_state": best["state"]}
    del backend, bridge
    torch.cuda.empty_cache()
    return info


def train() -> dict:
    import torch

    from workstreams.black_physics.granite import finetune as FTN
    from workstreams.black_physics.granite import pipeline as P
    from workstreams.black_physics.granite.artifacts import file_sha256
    from workstreams.black_physics.granite.backends import make_backend, versions

    by = P.train_by_name()
    groups = P.prepared()["groups"]
    gaps = FTN.a_gaps(by, groups)
    examples = FTN.ttm_pair_examples(by, gaps)
    two = sum(e.u is not None for e in examples)
    P.log(f"twosided: {len(examples)} TTM examples from {len(gaps)} A gaps ({two} two-sided)")
    b_tasks = P.dev_tasks("B")
    base = make_backend("ttm", None)
    base.model.eval()
    step0 = float(b_eval(b_tasks, by, base, make_bridge().to(base.device)))
    del base
    runs, best = [], None
    for setting, lr_b, lr_h in SWEEP:
        info = train_setting(setting, lr_b, lr_h, examples, b_tasks, by, step0, P.log)
        st = info.pop("best_state")
        runs.append(info)
        P.log(f"twosided {setting}: best B {info['best_metric']:.5f} at update {info['best_update']} (start "
              f"{info['start_metric']:.5f}, pretrained {step0:.5f}); {info['updates_run']} updates, "
              f"{info['train_minutes']:.1f} min, stop: {info['stop_reason']}")
        if st is not None and (best is None or info["best_metric"] < best[0]["best_metric"] - 1e-12):
            best = (info, st)
    info, st = best
    meta = {"setting": info["setting"], "lr_bridge": info["lr_bridge"], "lr_heads": info["lr_heads"],
            "best_update": info["best_update"], "B_metric": info["best_metric"], "pretrained_B": step0,
            "ft_config": asdict(FTN.FT), "config_hash": C.config_hash(), "weights": f"{C.TTM_REPO}@{C.TTM_REVISION}",
            "n_feat": N_FEAT, "scale": SCALE}
    checkpoint().parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": st["ttm"], "bridge": st["bridge"], "meta": meta}, checkpoint())
    rec = {"selected": meta, "runs": runs, "n_examples": len(examples), "n_two_sided": two, "n_gaps_A": len(gaps),
           "checkpoint": str(checkpoint().relative_to(C.work_dir())), "checkpoint_sha256": file_sha256(checkpoint()),
           "versions": versions(), "git": P.git_head()}
    P.save_json(P.path("twosided_ttm.json"), rec)
    P.log(f"twosided: selected {meta['setting']} (B {meta['B_metric']:.5f} vs pretrained {step0:.5f})")
    return rec


# ------------------------------------------------------------------------------------------------ loading
def load_checkpoint(device=None, path=None):
    """(backend, bridge, meta) from a saved two-sided checkpoint."""
    import torch

    from workstreams.black_physics.granite.backends import make_backend

    ck = torch.load(path or checkpoint(), map_location="cpu", weights_only=True)
    backend = make_backend("ttm", device)
    backend.model.load_state_dict(ck["state"])
    backend.model.eval()
    bridge = make_bridge().to(backend.device)
    bridge.load_state_dict(ck["bridge"])
    bridge.eval()
    return backend, bridge, ck["meta"]


def load_variant(train_wells, device=None, log=None) -> tuple:
    """(reconstructor, calibrator, provenance) for the station row; refused when the artifacts are stale."""
    from workstreams.black_physics.granite import artifacts as ART

    if not ART.is_official(train_wells):          # contract check on other wells: local fit, identity bridge
        backend, cal, prov = ART.local_fit(C.VARIANT_IDS[("ttm", True)], train_wells, device, log)
        bridge = make_bridge().to(backend.device)
        bridge.eval()
        return TwoSidedReconstructor(backend, bridge), cal, {**prov, "bridge": "identity (local fit)"}
    import json

    p = manifest_path()
    if not p.is_file():
        raise ART.StaleArtifacts(f"no two-sided manifest at {p}; run `two_sided train` and `two_sided score`")
    m = json.loads(p.read_text(encoding="utf-8"))
    want = ART.identity_record(ART.official_identity(train_wells))
    problems = [f"{k}: artifacts {m['identity'].get(k)!r} != current {v!r}" for k, v in want.items()
                if m["identity"].get(k) != v]
    ck = C.work_dir() / m["checkpoint"]
    if not ck.is_file():
        problems.append(f"checkpoint {ck} is missing")
    elif ART.file_sha256(ck) != m["checkpoint_sha256"]:
        problems.append(f"checkpoint {ck} changed (sha256 mismatch)")
    if problems:
        raise ART.StaleArtifacts("two-sided artifacts do not match: " + "; ".join(problems))
    backend, bridge, meta = load_checkpoint(device, ck)
    cal = CAL.IntervalCalibrator.from_dict(m["calibrator"])
    prov = {"source": "frozen two-sided fine-tune", "manifest_frozen_at": m["frozen_at"], "checkpoint": m["checkpoint"],
            "checkpoint_sha256": m["checkpoint_sha256"], "setting": meta["setting"], "calibrator": m["calibrator"], **want}
    return TwoSidedReconstructor(backend, bridge), cal, prov


# ------------------------------------------------------------------------------------------------ scoring
def score() -> dict:
    """D bands, B and D rows, then the sealed test gaps once (keyed, full precision), and the manifest."""
    import torch

    from benchmark import test_def as td
    from workstreams.black_physics.granite import artifacts as ART
    from workstreams.black_physics.granite import pipeline as P
    from workstreams.black_physics.granite.finetune import _arrays

    by = P.train_by_name()
    d_tasks, b_tasks = P.dev_tasks("D"), P.dev_tasks("B")
    tr, te = P.official()
    tasks = td.build_def_tasks(td.load_def(C.TEST_NAME), tr, te)
    wells = {w.name: w for w in te}
    for t in tasks:
        idx = np.flatnonzero(t.predict_mask)
        t.meta["gap_id"] = f"{t.well_id}|{idx[0]}:{idx[-1] + 1}"
    backend, bridge, meta = load_checkpoint()
    rec = TwoSidedReconstructor(backend, bridge)
    d_pieces = rec.run(d_tasks)
    cal = CAL.fit("ttm", [(t.well_id, ps[0], P.truth_for(t, by)[ps[0].a:ps[0].b])
                          for t, ps in zip(d_tasks, d_pieces) if ps], "D")
    d_rows = P.gap_rows(d_tasks, _arrays(d_tasks, d_pieces, cal)["p50"], d_pieces, by)
    b_pieces = rec.run(b_tasks)
    b_rows = P.gap_rows(b_tasks, _arrays(b_tasks, b_pieces, cal)["p50"], b_pieces, by)
    P.save_json(P.path("dev", f"B_{NAME}.json"), b_rows)
    P.save_json(P.path("dev", f"D_{NAME}.json"), d_rows)
    t0 = time.perf_counter()
    pieces = rec.run(tasks)
    inf_s = time.perf_counter() - t0
    arr = _arrays(tasks, pieces, cal)
    rows = P.gap_rows(tasks, arr["p50"], pieces, wells)
    keyed = {f"{NAME}|{t.meta['gap_id']}|{k}": np.asarray(v[i], float)[t.predict_mask]
             for i, t in enumerate(tasks) for k, v in arr.items() if v[i] is not None}
    np.savez_compressed(P.path("final", "twosided_predictions.npz"), **keyed)
    man = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "identity": ART.identity_record(ART.official_identity(tr)),
           "checkpoint": str(checkpoint().relative_to(C.work_dir())), "checkpoint_sha256": ART.file_sha256(checkpoint()),
           "calibrator": cal.as_dict(), "meta": meta, "git": P.git_head()}
    P.save_json(manifest_path(), man)
    res = {"rows": {NAME: rows}, "meta": meta, "calibrator": cal.as_dict(), "B": P.summary(b_rows),
           "D": P.summary(d_rows), "inference_s": round(inf_s, 3), "git": P.git_head()}
    P.save_json(P.path("final", "twosided.json"), res)
    P.log(f"twosided score: B {P.mean_well_mae(b_rows):.5f}, D {P.mean_well_mae(d_rows):.5f}, test "
          f"{P.mean_well_mae(rows):.5f} mean well MAE (k50 {cal.k50:.3f}, k90 {cal.k90:.3f})")
    del backend, bridge
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------------ comparison
COMPARE = {"black_ttm_r3_frozen": "TTM-R3 frozen", "black_ttm_r3_ft": "TTM-R3 fine-tune (heads)",
           "black_tspulse_ft": "TSPulse fine-tune (whole model)", "black_timestral_target_only": "Timestral target-only"}


def _per_well(rows) -> dict:
    by = {}
    for r in rows:
        if r["n_pred"]:
            a = by.setdefault(r["well"], [0.0, 0])
            a[0] += r["abs_err_sum"]
            a[1] += r["n_pred"]
    return {w: s / n for w, (s, n) in by.items()}


def _paired(a: dict, b: dict, n: int = 2000, seed: int = 2026) -> dict:
    ws = sorted(set(a) & set(b))
    d = np.array([a[w] - b[w] for w in ws])
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"delta": float(d.mean()), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            "wells": len(ws), "better_in": int((d < 0).sum())}


def compare() -> dict:
    """Two-sided row against the other density-only rows: B, D and test wells; two-sided vs one-sided gaps."""
    import json

    from workstreams.black_physics.granite import pipeline as P

    W = C.work_dir()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))   # noqa: E731
    test = {**load(W / "final" / "final.json")["rows"], **load(W / "final" / "finetune.json")["rows"],
            **load(W / "final" / "twosided.json")["rows"]}
    if (W / "final" / "timestral.json").is_file():
        test.update(load(W / "final" / "timestral.json")["rows"])
    sets = {"test": test}
    for g in ("B", "D"):
        sets[g] = {n: load(W / "dev" / f"{g}_{n}.json") for n in [NAME, *COMPARE] if (W / "dev" / f"{g}_{n}.json").is_file()}
    out = {"mean_well_mae": {}, "paired": {}, "by_status": {}}
    for g, rows in sets.items():
        pw = {n: _per_well(r) for n, r in rows.items()}
        out["mean_well_mae"][g] = {n: float(np.mean(list(v.values()))) for n, v in pw.items()}
        out["paired"][g] = {n: _paired(pw[NAME], pw[n]) for n in COMPARE if n in pw}
    for n in (NAME, "black_ttm_r3_frozen", "black_ttm_r3_ft"):
        acc = {}
        for g in ("D", "test"):
            ref = {r["key"]: r for r in sets[g]["black_ttm_r3_frozen"]}
            for r in sets[g][n]:
                f = ref.get(r["key"])
                if f is None or not (r["n_pred"] and f["n_pred"]):
                    continue
                k = "two_sided" if f["status"] == "two_sided" else "one_sided_or_fallback"
                acc.setdefault(k, []).append(r["mae"])
        out["by_status"][n] = {k: {"gaps": len(v), "mean_gap_mae": float(np.mean(v))} for k, v in acc.items()}
    P.save_json(P.path("final", "twosided_compare.json"), out)
    for g in ("B", "D", "test"):
        m = out["mean_well_mae"][g]
        P.log(f"twosided compare {g}: " + ", ".join(f"{n} {v:.4f}" for n, v in m.items()))
        for n, x in out["paired"][g].items():
            P.log(f"   two-sided - {n}: {x['delta']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}], "
                  f"better in {x['better_in']}/{x['wells']} wells")
    for n, v in out["by_status"].items():
        P.log(f"twosided compare by status (D + test) {n}: " + "; ".join(
            f"{k} {x['gaps']} gaps {x['mean_gap_mae']:.4f}" for k, x in v.items()))
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Two-sided TTM: learned bridge between the two directional forecasts")
    ap.add_argument("step", choices=["train", "score", "compare"])
    a = ap.parse_args(argv)
    {"train": train, "score": score, "compare": compare}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
