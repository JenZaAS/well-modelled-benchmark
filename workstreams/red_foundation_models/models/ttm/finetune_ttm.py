"""CPU fine-tune of IBM Granite TTM R2 on FORCE 2020 train wells.

Writes weights to ``workstreams/red_foundation_models/weights/ttm/<TARGET>/``
(or ``$WMB_RED_CACHE/ttm/<TARGET>``). With ``--probabilistic``, writes a
quantile-head checkpoint to ``.../ttm/<TARGET>-probabilistic/`` (q05/q50/q95,
pinball loss, optional split-conformal widening on held-out train wells).

If the output path is inside the repo, it must be git-ignored
(``git check-ignore``) before anything is written.

Modest defaults for a short smoke: 1 epoch, 8 wells, 256 windows, context 512,
horizon 96, batch 8. Example:

    python -m workstreams.red_foundation_models.models.ttm.finetune_ttm --target RHOB --epochs 1
    python -m workstreams.red_foundation_models.models.ttm.finetune_ttm --target RHOB --probabilistic
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Repo root on sys.path when run as a module or as a file.
REPO = Path(__file__).resolve().parents[4]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workstreams.red_foundation_models.cache import (  # noqa: E402
    assert_gitignored,
    red_hf_home,
    ttm_finetuned_dir,
    ttm_probabilistic_dir,
)
from workstreams.red_foundation_models.models.ttm.backend import (  # noqa: E402
    CONTEXT,
    HORIZON,
    pretrained_path,
)
from workstreams.red_foundation_models.probabilistic import (  # noqa: E402
    QUANTILE_LEVELS,
    conformal_width as cqr_width,
    write_meta,
)


class WindowDataset(Dataset):
    def __init__(self, windows: list[tuple[np.ndarray, np.ndarray]]):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        past, future = self.windows[i]
        return {
            "past_values": torch.from_numpy(past).unsqueeze(-1),
            "future_values": torch.from_numpy(future).unsqueeze(-1),
        }


def _windows(series: np.ndarray, context: int, horizon: int, stride: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(series, dtype=np.float32)
    out = []
    last = y.size - context - horizon
    if last < 0:
        return out
    for start in range(0, last + 1, max(stride, 1)):
        past = y[start:start + context]
        future = y[start + context:start + context + horizon]
        if np.isfinite(past).all() and np.isfinite(future).all():
            out.append((past, future))
    return out


def _collect(train_wells, target: str, context: int, horizon: int, max_wells: int, max_windows: int, stride: int):
    wells = [w for w in train_wells if target in w.curves]
    wells = sorted(wells, key=lambda w: w.name)[: max_wells or None]
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    used = []
    for w in wells:
        y = np.asarray(w.curves[target], dtype=np.float32)
        got = _windows(y, context, horizon, stride)
        if got:
            used.append(w.name)
            windows.extend(got)
        if max_windows and len(windows) >= max_windows:
            windows = windows[:max_windows]
            break
    return used, windows


def _split_wells(wells, calib_frac: float, seed: int):
    """Hold out a fraction of training wells for conformal calibration (never test wells)."""
    eligible = [w for w in wells]
    eligible = sorted(eligible, key=lambda w: w.name)
    if calib_frac <= 0 or len(eligible) < 2:
        return eligible, []
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(eligible))
    n_cal = int(round(len(eligible) * calib_frac))
    n_cal = max(1, min(n_cal, len(eligible) - 1))
    cal_idx = set(order[:n_cal].tolist())
    calib = [eligible[i] for i in range(len(eligible)) if i in cal_idx]
    train = [eligible[i] for i in range(len(eligible)) if i not in cal_idx]
    return train, calib


def _loss(model, batch, device):
    past = batch["past_values"].to(device)
    future = batch["future_values"].to(device)
    out = model(past_values=past, future_values=future)
    if getattr(out, "loss", None) is not None:
        return out.loss
    pred = out.prediction_outputs
    if pred.ndim == 3:
        pred = pred[..., 0]
        truth = future[..., 0]
    else:
        truth = future.reshape(pred.shape)
    return torch.mean((pred - truth) ** 2)


def attach_quantile_head(model, levels=QUANTILE_LEVELS):
    """Attach TTM's native MultiQuantileHead (monotonic q05/q50/q95, pinball)."""
    from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import MultiQuantileHead

    levels = [float(x) for x in levels]
    model.config.multi_quantile_head = True
    model.config.loss = "pinball"
    model.config.quantile_levels = levels
    model.multi_quantile_head_block = MultiQuantileHead(model.config)
    return model


def _calibrate(model, windows, device, batch_size: int, alpha: float) -> float:
    """CQR width from held-out training windows (model in eval mode)."""
    if not windows:
        return 0.0
    loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False, num_workers=0)
    ys, lo, hi = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            past = batch["past_values"].to(device)
            future = batch["future_values"].to(device)
            out = model(past_values=past, future_values=None, return_loss=False)
            q = out.quantile_outputs  # [B, Q, T, C]
            if q is None:
                raise RuntimeError("probabilistic TTM returned no quantile_outputs during calibration")
            ys.append(future[..., 0].cpu().numpy().reshape(-1))
            lo.append(q[:, 0, :, 0].cpu().numpy().reshape(-1))
            hi.append(q[:, -1, :, 0].cpu().numpy().reshape(-1))
    return cqr_width(np.concatenate(ys), np.concatenate(lo), np.concatenate(hi), alpha=alpha)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CPU fine-tune TTM R2 on FORCE 2020 train wells")
    p.add_argument("--target", default="RHOB", choices=["RHOB", "DTC", "DTS"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--context", type=int, default=CONTEXT)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--max-wells", type=int, default=8, help="0 = all train wells")
    p.add_argument("--max-windows", type=int, default=256, help="0 = no cap")
    p.add_argument("--stride", type=int, default=64, help="window stride along depth")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probabilistic", action="store_true",
                   help="Train a q05/q50/q95 quantile head with pinball loss")
    p.add_argument("--conformal", action=argparse.BooleanOptionalAction, default=None,
                   help="Split-conformal widening on held-out train wells "
                        "(default: on when --probabilistic)")
    p.add_argument("--calib-frac", type=float, default=0.2,
                   help="Fraction of training wells held out for conformal calibration")
    p.add_argument("--max-seconds", type=float, default=0,
                   help="Stop after this many seconds of wall time (0 = no cap)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.epochs < 1:
        raise SystemExit("epochs must be >= 1")
    use_conformal = args.conformal if args.conformal is not None else bool(args.probabilistic)
    if use_conformal and not args.probabilistic:
        raise SystemExit("--conformal requires --probabilistic")
    if args.calib_frac < 0 or args.calib_frac >= 1:
        raise SystemExit("--calib-frac must be in [0, 1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    from wmb.paths import get_dataset_path
    from benchmark.run import load_dataset

    out_dir = ttm_probabilistic_dir(args.target) if args.probabilistic else ttm_finetuned_dir(args.target)
    out_dir = assert_gitignored(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(red_hf_home())

    t0 = time.perf_counter()
    print(f"dataset=force2020 target={args.target} context={args.context} horizon={args.horizon}", flush=True)
    print(f"probabilistic={args.probabilistic} conformal={use_conformal} cache={out_dir}", flush=True)
    print(f"force={get_dataset_path('force2020')}", flush=True)
    train, _test = load_dataset("force2020", args.seed)
    eligible = [w for w in train if args.target in w.curves]
    calib_wells = []
    if args.probabilistic and use_conformal:
        eligible, calib_wells = _split_wells(eligible, args.calib_frac, args.seed)
        print(f"split train_wells={len(eligible)} calib_wells={len(calib_wells)} frac={args.calib_frac}", flush=True)
    used, windows = _collect(eligible, args.target, args.context, args.horizon, args.max_wells, args.max_windows, args.stride)
    if not windows:
        raise SystemExit("no finite training windows — check FORCE data and --context/--horizon")
    print(f"wells={len(used)} windows={len(windows)} names={used[:12]}{'...' if len(used) > 12 else ''}", flush=True)

    device = torch.device("cpu")
    source = pretrained_path()
    if Path(source).is_dir():
        model = TinyTimeMixerForPrediction.from_pretrained(source, local_files_only=True)
    else:
        model = TinyTimeMixerForPrediction.from_pretrained(source)
    if args.probabilistic:
        attach_quantile_head(model, QUANTILE_LEVELS)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loader = DataLoader(WindowDataset(windows), batch_size=args.batch_size, shuffle=True, num_workers=0)

    epoch_times = []
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        te = time.perf_counter()
        losses = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, batch, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        dt = time.perf_counter() - te
        epoch_times.append(dt)
        mean = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch={epoch}/{args.epochs} loss={mean:.5f} time={dt:.2f}s windows/s={len(windows)/dt:.1f}", flush=True)
        if args.max_seconds and (time.perf_counter() - t0) >= args.max_seconds:
            print(f"stopping early at {time.perf_counter() - t0:.1f}s (max-seconds={args.max_seconds})", flush=True)
            stopped_early = True
            break

    model.eval()
    width = 0.0
    calib_used, calib_windows = [], []
    if args.probabilistic and use_conformal:
        calib_used, calib_windows = _collect(
            calib_wells, args.target, args.context, args.horizon, 0, args.max_windows or 0, args.stride
        )
        if calib_windows:
            tc = time.perf_counter()
            width = _calibrate(model, calib_windows, device, args.batch_size, alpha=0.10)
            print(
                f"conformal width={width:.6f} calib_wells={len(calib_used)} "
                f"calib_windows={len(calib_windows)} time={time.perf_counter()-tc:.2f}s",
                flush=True,
            )
        else:
            print("conformal skipped: no finite calibration windows", flush=True)

    model.save_pretrained(out_dir)
    if args.probabilistic:
        write_meta(out_dir, {
            "probabilistic": True,
            "quantile_levels": list(QUANTILE_LEVELS),
            "conformal": bool(use_conformal),
            "conformal_width": float(width),
            "conformal_alpha": 0.10,
            "target": args.target,
            "context": args.context,
            "horizon": args.horizon,
            "epochs": args.epochs,
            "epochs_run": len(epoch_times),
            "stopped_early": stopped_early,
            "train_wells": used,
            "calib_wells": calib_used,
            "train_windows": len(windows),
            "calib_windows": len(calib_windows),
            "base": source,
        })
    elapsed = time.perf_counter() - t0
    print(
        f"saved={out_dir} total={elapsed:.2f}s epochs={len(epoch_times)} "
        f"epoch_mean={float(np.mean(epoch_times)):.2f}s hf_home={os.environ.get('HF_HOME')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
