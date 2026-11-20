"""Zero-shot (or fine-tuned) CPU inference for IBM Granite TTM R2.

Install ``granite-tsfm`` without downgrading the shared ``torch`` / ``transformers``
pins used by other teams (needs ``transformers>=4.57.6,<6`` and ``torch>=2.10,<2.12``).
Prefetch ``ibm-granite/granite-timeseries-ttm-r2`` into the workstream weight
folder (``weights/hf`` or ``$WMB_RED_CACHE/hf``). Importing this module does not load TTM.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os

import numpy as np

from workstreams.red_foundation_models.probabilistic import apply_conformal, read_meta

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
CONTEXT = 512
HORIZON = 96


def pretrained_path(weights_dir: Path | None = None) -> str:
    """Fine-tuned folder, then a local snapshot under the red weight dir, then the hub id."""
    if weights_dir is not None and (Path(weights_dir) / "config.json").is_file():
        return str(Path(weights_dir))
    env = os.environ.get("WMB_TTM_PATH")
    if env and (Path(env) / "config.json").is_file():
        return env
    from workstreams.red_foundation_models.cache import DEFAULT_CACHE, red_cache_dir, red_hf_home
    # Always also search the workstream default folder so a WMB_RED_CACHE
    # override still finds the prefetched base snapshot.
    roots = [red_cache_dir(), red_hf_home(), DEFAULT_CACHE, DEFAULT_CACHE / "hf"]
    seen = set()
    for root in roots:
        for local in (root / "local" / "granite-timeseries-ttm-r2", root / "granite-timeseries-ttm-r2"):
            local = local.resolve()
            if local in seen:
                continue
            seen.add(local)
            if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
                return str(local)
    return MODEL_ID


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{name} must be a nonempty 1D array")
    if values.dtype.kind not in "iuf" and values.dtype != np.float32:
        raise TypeError(f"{name} must contain real numeric values")
    return np.ascontiguousarray(values, dtype=np.float32)


def is_probabilistic_model(model) -> bool:
    if getattr(model, "multi_quantile_head_block", None) is not None:
        return True
    return bool(getattr(getattr(model, "config", None), "multi_quantile_head", False))


def _quantiles(out) -> np.ndarray | None:
    q = getattr(out, "quantile_outputs", None)
    if q is None:
        return None
    arr = q.detach().cpu().numpy()  # [B, Q, T, C]
    return np.asarray(arr[0, :, :, 0], dtype=np.float32)


def load(context: int = CONTEXT, horizon_cap: int = HORIZON, weights_dir: Path | None = None) -> Callable:
    """Load cached (or fine-tuned) weights once; the callable takes only local arrays.

    Point checkpoints return a 1-D forecast of length ``horizon`` (the depth
    adapter adds a Gaussian band). A quantile-head checkpoint returns a
    ``(3, horizon)`` array of q05/q50/q95, with any saved conformal width applied.
    """
    import torch
    # Import the model class only. The top-level ``tsfm_public`` package pulls
    # ``datasets`` → ``pyarrow.dataset``, which on Windows tries optional
    # plugins (arrow_substrait.dll, _hdfs, _csv) and can trip Smart App Control.
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    ctx = int(context or CONTEXT)
    cap = int(horizon_cap or HORIZON)
    device = torch.device("cpu")
    source = pretrained_path(weights_dir)
    if Path(source).is_dir():
        # from_pretrained on a local folder; do not use get_model() (hub-only, and
        # it imports tsfm_public.toolkit → datasets → pyarrow).
        model = TinyTimeMixerForPrediction.from_pretrained(source, local_files_only=True)
    else:
        from workstreams.red_foundation_models.cache import red_hf_home
        os.environ["HF_HOME"] = str(red_hf_home())
        red_hf_home().mkdir(parents=True, exist_ok=True)
        model = TinyTimeMixerForPrediction.from_pretrained(source)
    model.to(device)
    model.eval()
    model_ctx = int(getattr(model.config, "context_length", ctx) or ctx)
    model_hor = int(getattr(model.config, "prediction_length", cap) or cap)
    probabilistic = is_probabilistic_model(model)
    meta = read_meta(source) if Path(source).is_dir() else {}
    conformal = float(meta.get("conformal_width") or 0.0) if probabilistic else 0.0

    def forecast(history, past, future, horizon):
        # past/future covariates are accepted (same contract as red) but unused in
        # the published univariate zero-shot path; fine-tuned checkpoints may still
        # be univariate forecasters.
        del past, future
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        target = _vector(history, "history")
        if target.size >= model_ctx:
            window = target[-model_ctx:]
        else:
            window = np.pad(target, (model_ctx - target.size, 0), mode="edge")
        remaining = int(horizon)
        chunks = []
        hist = window
        with torch.inference_mode():
            while remaining > 0:
                out = model(past_values=torch.from_numpy(hist).view(1, model_ctx, 1))
                q = _quantiles(out) if probabilistic else None
                if q is not None:
                    if conformal:
                        lo, mid, hi = apply_conformal(q[0], q[q.shape[0] // 2], q[-1], conformal)
                        q = np.stack([lo, mid, hi])
                    take_n = min(q.shape[1], remaining, model_hor)
                    take = q[:, :take_n]
                    if not take_n:
                        raise ValueError("TTM returned an empty forecast")
                    chunks.append(take)
                    remaining -= take_n
                    hist = np.concatenate([hist, take[1]])[-model_ctx:]
                else:
                    pred = out.prediction_outputs
                    if pred.ndim == 3:
                        step = pred[0, :, 0]
                    else:
                        step = pred.reshape(-1)
                    y = step.detach().cpu().numpy().astype(np.float32, copy=False)
                    take = y[: min(len(y), remaining, model_hor)]
                    if not take.size:
                        raise ValueError("TTM returned an empty forecast")
                    chunks.append(take)
                    remaining -= take.size
                    hist = np.concatenate([hist, take])[-model_ctx:]
        if probabilistic:
            return np.concatenate(chunks, axis=1)
        return np.concatenate(chunks)

    return forecast
