"""Head-only fine-tuning of the two Granite models on fitting wells (A); checkpoints selected on B.

TTM-R3:  trend + residual prediction heads (`*.head.*`, `*.multi_quantile_head_block.*`) train; mixer backbones,
         decoders and scaler stay frozen. Each A gap gives two directed examples (shallow flank forward, deeper
         flank reversed with its labels reversed). Loss: pinball over the nine provided quantiles on valid
         horizon positions, averaged per example, equal weight per example.
TSPulse: only `decoder_with_head.time_head` trains; backbone, decoder and FFT path stay frozen. Forward on the
         masked window with return_loss=False, then external MAE against the separately held labels at the
         hidden positions only (the built-in loss would use the masked input as its label).

Time box (both): seed, AdamW(lr 3e-4, wd 0.01), effective batch 8 via accumulation, <= 300 updates or 45 minutes,
evaluate every 25 updates on B, stop after 3 evaluations without improvement, keep the best evaluation
(step 0 included). A tiny overfit-and-reload test and 10 measured optimizer steps per microbatch size run first;
their weights are discarded.
"""
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass

import numpy as np

from workstreams.black_physics.granite import gaps as G
from workstreams.black_physics.granite.backends import TTM_QUANTILES
from workstreams.black_physics.granite.config import CONFIG, ExperimentConfig
from workstreams.black_physics.granite.reconstruct import TSPulseReconstructor


@dataclass
class Example:
    well: str
    gap_id: str
    x: np.ndarray            # TTM: finite context (oldest first); TSPulse: 512-sample window, NaN = unobserved
    y: np.ndarray            # labels in model output order (TTM: forecast steps; TSPulse: hidden positions)
    off: int = 0             # TSPulse: index of the first hidden sample inside the window


def ttm_examples(wells: dict, gaps, cfg: ExperimentConfig = CONFIG, target: str = "RHOB") -> list[Example]:
    out = []
    for g in gaps:
        w = wells[g.well]
        y = np.asarray(w.curves[target], float)
        obs = y.copy()
        obs[g.start:g.end] = np.nan                      # hide first; everything below reads obs only
        if g.n > cfg.ttm_horizon:
            continue
        lab = y[g.start:g.end].copy()
        up = G.flank_run(obs, g.start, g.end, g.seg_start, g.seg_end, cfg.ttm_context, "above")
        dn = G.flank_run(obs, g.start, g.end, g.seg_start, g.seg_end, cfg.ttm_context, "below")
        if len(up) >= cfg.ttm_min_context:
            out.append(Example(g.well, g.gap_id, obs[up], lab))
        if len(dn) >= cfg.ttm_min_context:
            out.append(Example(g.well, g.gap_id, obs[dn][::-1].copy(), lab[::-1].copy()))
    return out


def tspulse_examples(wells: dict, gaps, cfg: ExperimentConfig = CONFIG, target: str = "RHOB") -> list[Example]:
    rec = TSPulseReconstructor(None, cfg.tspulse_window, cfg.min_flank_samples)
    out = []
    for g in gaps:
        if g.n >= cfg.tspulse_window:
            continue
        y = np.asarray(wells[g.well].curves[target], float)
        obs = y.copy()
        obs[g.start:g.end] = np.nan
        vals, w0 = rec.window_values(obs, g.start, g.end, g.seg_start, g.seg_end)
        idx = np.arange(w0, w0 + cfg.tspulse_window)
        seen = np.isfinite(vals)
        if max(int((seen & (idx < g.start)).sum()), int((seen & (idx >= g.end)).sum())) < cfg.min_flank_samples:
            continue
        out.append(Example(g.well, g.gap_id, vals, y[g.start:g.end].copy(), g.start - w0))
    return out


class WellSampler:
    """Wells sampled evenly, then one of that well's examples uniformly (deterministic generator)."""

    def __init__(self, examples: list[Example], seed: int):
        self.by_well: dict[str, list[Example]] = {}
        for e in examples:
            self.by_well.setdefault(e.well, []).append(e)
        self.wells = sorted(self.by_well)
        self.rng = np.random.default_rng(seed)

    def draw(self, n: int) -> list[Example]:
        out = []
        for _ in range(n):
            w = self.wells[int(self.rng.integers(len(self.wells)))]
            ex = self.by_well[w]
            out.append(ex[int(self.rng.integers(len(ex)))])
        return out


def ttm_loss(backend, batch: list[Example]):
    import torch

    past, mask = backend.batch([e.x for e in batch])
    q = backend.forward(past, mask)                                         # [B, 9, 336]
    B, _, H = q.shape
    y = np.zeros((B, H), np.float32)
    v = np.zeros((B, H), np.float32)
    for i, e in enumerate(batch):
        y[i, :len(e.y)] = e.y
        v[i, :len(e.y)] = 1.0
    Y, V = torch.from_numpy(y).to(q.device), torch.from_numpy(v).to(q.device)
    tau = torch.tensor(TTM_QUANTILES, dtype=q.dtype, device=q.device).view(1, -1, 1)
    d = Y[:, None, :] - q
    pin = torch.maximum(tau * d, (tau - 1) * d) * V[:, None, :]
    per_example = pin.sum(dim=(1, 2)) / (V.sum(dim=1) * len(TTM_QUANTILES))
    return per_example.mean()


def tspulse_loss(backend, batch: list[Example]):
    import torch

    past, mask = backend.batch(np.stack([e.x for e in batch]))
    r = backend.forward(past, mask)                                         # [B, 512]
    B, W = r.shape
    y = np.zeros((B, W), np.float32)
    v = np.zeros((B, W), np.float32)
    for i, e in enumerate(batch):
        y[i, e.off:e.off + len(e.y)] = e.y
        v[i, e.off:e.off + len(e.y)] = 1.0
    Y, V = torch.from_numpy(y).to(r.device), torch.from_numpy(v).to(r.device)
    per_example = ((r - Y).abs() * V).sum(dim=1) / V.sum(dim=1)
    return per_example.mean()


LOSSES = {"ttm": ttm_loss, "tspulse": tspulse_loss}


def _sync(dev) -> None:
    import torch

    if dev.type == "cuda":
        torch.cuda.synchronize()


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:  # noqa: BLE001  (resource note only)
        return None


def _grad_report(backend) -> dict:
    """Per head group: number of tensors, finite?, total |grad|; groups = trend/residual (TTM) or time head."""
    groups: dict[str, dict] = {}
    for n, p in backend.model.named_parameters():
        if not p.requires_grad:
            continue
        key = n.split(".")[0] if backend.arm == "ttm" else "time_head"
        g = groups.setdefault(key, {"tensors": 0, "finite": True, "abs_sum": 0.0})
        g["tensors"] += 1
        if p.grad is None:
            g["finite"] = False
            continue
        gg = p.grad.detach()
        g["finite"] &= bool(gg.isfinite().all())
        g["abs_sum"] += float(gg.abs().sum())
    return groups


def _optimizer(params, cfg: ExperimentConfig, lr: float | None = None):
    import torch

    return torch.optim.AdamW(params, lr=lr or cfg.lr, weight_decay=cfg.weight_decay)


def _update(backend, loss_fn, sampler: WellSampler, opt, micro: int, accum: int) -> float:
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(accum):
        loss = loss_fn(backend, sampler.draw(micro)) / accum
        loss.backward()
        total += float(loss.detach())
    opt.step()
    return total


def overfit_and_reload(backend, examples: list[Example], cfg: ExperimentConfig, steps: int = 40) -> dict:
    """Two examples, a larger learning rate: the head must fit them; a saved head must reload exactly."""
    import torch

    loss_fn = LOSSES[backend.arm]
    init = backend.head_state()
    params = [p for p in backend.model.parameters() if p.requires_grad]
    opt = _optimizer(params, cfg, lr=3e-3)
    few = examples[:2]
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(backend, few)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    backend.model.eval()
    with torch.no_grad():
        before = loss_fn(backend, few).item()
    buf = io.BytesIO()
    torch.save(backend.head_state(), buf)
    backend.load_head(init)
    with torch.no_grad():
        reset = loss_fn(backend, few).item()
    buf.seek(0)
    backend.load_head(torch.load(buf, weights_only=True))
    with torch.no_grad():
        after = loss_fn(backend, few).item()
    backend.load_head(init)
    backend.set_trainable()
    return {"steps": steps, "first_loss": losses[0], "last_loss": losses[-1],
            "loss_dropped": losses[-1] < 0.8 * losses[0], "eval_loss_trained": before, "eval_loss_reset": reset,
            "eval_loss_reloaded": after, "reload_exact": abs(after - before) < 1e-7}


def measure(backend, sampler: WellSampler, cfg: ExperimentConfig, micro_sizes=(1, 2, 4, 8)) -> list[dict]:
    """cfg.measure_steps optimizer updates per microbatch size (effective batch fixed); weights discarded."""
    import torch

    loss_fn = LOSSES[backend.arm]
    init = backend.head_state()
    rows = []
    for m in micro_sizes:
        accum = max(1, cfg.effective_batch // m)
        params = [p for p in backend.model.parameters() if p.requires_grad]
        opt = _optimizer(params, cfg)
        dev = backend.device
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            _update(backend, loss_fn, sampler, opt, m, accum)          # warm-up, not timed
            _sync(dev)
            t0 = time.perf_counter()
            for _ in range(cfg.measure_steps):
                _update(backend, loss_fn, sampler, opt, m, accum)
            _sync(dev)
            sec = (time.perf_counter() - t0) / cfg.measure_steps
            row = {"microbatch": m, "accumulation": accum, "s_per_update": round(sec, 4), "ok": True}
        except torch.cuda.OutOfMemoryError as exc:                  # pragma: no cover (hardware-specific)
            row = {"microbatch": m, "accumulation": accum, "ok": False, "error": f"OOM: {exc}"[:200]}
            torch.cuda.empty_cache()
        if dev.type == "cuda":
            row.update(peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                       peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1))
        row["process_rss_mb"] = _rss_mb()
        rows.append(row)
        backend.load_head(init)
        if not row["ok"]:
            break
    return rows


def choose_microbatch(rows: list[dict], vram_budget_mb: float = 2500.0) -> int:
    ok = [r for r in rows if r.get("ok") and r.get("peak_reserved_mb", 0.0) <= vram_budget_mb]
    return max((r["microbatch"] for r in ok), default=1)


def train_head(backend, examples: list[Example], eval_fn, cfg: ExperimentConfig = CONFIG,
               max_updates: int | None = None, max_minutes: float | None = None, log=print) -> dict:
    """Run the time-boxed head fine-tune. eval_fn() -> float (lower is better) on B, computed in eval mode.

    Early stopping compares with the best evaluation so far, step 0 included. The returned `best_state` is the
    best *actually tuned* head (an evaluation after >= 1 update), kept even when step 0 scored better on B;
    `step0_won` records that case.
    """
    import torch

    max_updates = cfg.max_updates if max_updates is None else max_updates
    max_minutes = cfg.max_minutes if max_minutes is None else max_minutes
    torch.manual_seed(cfg.seed)
    names = backend.set_trainable()
    info: dict = {"arm": backend.arm, "trainable": names, "n_trainable": backend.n_parameters()["head"],
                  "n_parameters": backend.n_parameters()["total"], "n_examples": len(examples),
                  "n_wells": len({e.well for e in examples}), "device": str(backend.device)}
    log(f"[{backend.arm}] {info['n_trainable']} trainable of {info['n_parameters']} parameters, "
        f"{len(examples)} examples from {info['n_wells']} wells on {backend.device}")
    info["overfit_reload"] = overfit_and_reload(backend, examples, cfg)
    log(f"[{backend.arm}] overfit/reload: {info['overfit_reload']}")
    sampler = WellSampler(examples, cfg.seed)
    info["measure"] = measure(backend, sampler, cfg)
    micro = choose_microbatch(info["measure"])
    accum = max(1, cfg.effective_batch // micro)
    info.update(microbatch=micro, accumulation=accum)
    log(f"[{backend.arm}] measured {info['measure']}; using microbatch {micro} x {accum}")
    sampler = WellSampler(examples, cfg.seed)          # fresh stream for the real run
    loss_fn = LOSSES[backend.arm]
    params = [p for p in backend.model.parameters() if p.requires_grad]
    opt = _optimizer(params, cfg)
    init = backend.head_state()

    def evaluate() -> float:
        backend.model.eval()
        try:
            return float(eval_fn())
        finally:
            backend.set_trainable()

    t_eval = time.perf_counter()
    best = {"update": 0, "metric": evaluate(), "state": init}
    tuned = {"update": None, "metric": float("inf"), "state": None}
    info["eval_s_first"] = round(time.perf_counter() - t_eval, 2)
    history = [{"update": 0, "minutes": 0.0, "train_loss": None, "B_metric": best["metric"], "best": True}]
    log(f"[{backend.arm}] update 0: B mean well MAE {best['metric']:.5f}")
    stale, losses, truncated, stop_reason = 0, [], False, "max_updates"
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for u in range(1, max_updates + 1):
        losses.append(_update(backend, loss_fn, sampler, opt, micro, accum))
        if u == 1:
            info["first_update_grads"] = _grad_report(backend)
        minutes = (time.perf_counter() - t0) / 60
        if u % cfg.eval_every == 0 or u == max_updates or minutes >= max_minutes:
            metric = evaluate()
            improved = metric < best["metric"] - 1e-9
            if improved:
                best = {"update": u, "metric": metric, "state": backend.head_state()}
                stale = 0
            else:
                stale += 1
            if metric < tuned["metric"] - 1e-9:
                tuned = {"update": u, "metric": metric, "state": best["state"] if improved else backend.head_state()}
            history.append({"update": u, "minutes": round(minutes, 3),
                            "train_loss": float(np.mean(losses[-cfg.eval_every:])), "B_metric": metric,
                            "best": improved})
            log(f"[{backend.arm}] update {u}: loss {history[-1]['train_loss']:.5f} B {metric:.5f}"
                f"{' *' if improved else ''} ({minutes:.2f} min)")
            if stale >= cfg.patience:
                stop_reason = f"no improvement in {cfg.patience} evaluations"
                break
            if minutes >= max_minutes:
                truncated, stop_reason = True, f"time box {max_minutes} min"
                break
    _sync(backend.device)
    info["train_minutes"] = round((time.perf_counter() - t0) / 60, 3)
    info["updates_run"] = u
    if backend.device.type == "cuda":
        info["train_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        info["train_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    info["process_rss_mb"] = _rss_mb()
    info.update(history=history, best_overall_update=best["update"], best_overall_metric=best["metric"],
                best_update=tuned["update"], best_metric=tuned["metric"], step0_won=best["update"] == 0,
                step0_metric=history[0]["B_metric"], stop_reason=stop_reason, truncated=truncated,
                caps={"max_updates": max_updates, "max_minutes": max_minutes})
    changed = {n: float((tuned["state"][n] - init[n]).abs().max()) for n in init}
    info["best_head_max_abs_change"] = max(changed.values()) if changed else 0.0
    info["best_state"] = tuned["state"]
    backend.load_head(tuned["state"])
    backend.model.eval()
    return info


def train_with_fallbacks(make_backend, examples, eval_fn_for, cfg: ExperimentConfig = CONFIG, log=print) -> tuple:
    """GPU first; on CUDA OOM retry at microbatch 1, then CPU head-only with the same caps. Returns (backend, info)."""
    import torch

    backend = make_backend(None)
    try:
        info = train_head(backend, examples, eval_fn_for(backend), cfg, log=log)
        return backend, info
    except torch.cuda.OutOfMemoryError as exc:                       # pragma: no cover (hardware-specific)
        log(f"CUDA OOM ({exc}); retrying on CPU with the same caps")
        torch.cuda.empty_cache()
        backend = make_backend("cpu")
        info = train_head(backend, examples, eval_fn_for(backend), cfg, log=log)
        info["fallback"] = "cpu after CUDA OOM"
        return backend, info


def history_json(info: dict) -> str:
    return json.dumps({k: v for k, v in info.items() if k != "best_state"}, indent=2, default=str)


def mean_finite(x) -> float:
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else math.nan
