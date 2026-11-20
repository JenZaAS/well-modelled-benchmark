"""Zero-shot CPU inference for AutonLab MOMENT-1-small.

Install ``momentfm`` with ``python -m pip install --no-deps momentfm==0.1.4`` so its declared pins
(``transformers==4.33.3``, ``huggingface-hub==0.24.0``, ``numpy==1.25.2``) are not
applied. Verified on the box with ``transformers 4.57.6`` and ``torch 2.11.0+cpu``.
The published head is reconstruction/imputation (the forecasting head is not
pre-trained). Importing this module does not import MOMENT or access the cache.
"""
from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import numpy as np

MODEL_ID = "AutonLab/MOMENT-1-small"
CONTEXT = 512  # MOMENT seq_len
HORIZON = 96   # gap chunk; leaves 416 samples of context in the 512 window


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{name} must be a nonempty 1D array")
    return np.ascontiguousarray(values, dtype=np.float32)


def _search_roots() -> list[Path]:
    from wmb.paths import cache_dir
    from workstreams.red_foundation_models.cache import red_hf_home
    return [Path(red_hf_home()), Path(cache_dir()) / "hf"]


def _find_snapshot(roots: list[Path] | None = None) -> Path:
    roots = _search_roots() if roots is None else roots
    repo = "models--" + MODEL_ID.replace("/", "--")
    for root in roots:
        snaps = root / "hub" / repo / "snapshots"
        for cfg in sorted(snaps.glob("*/config.json")) if snaps.is_dir() else []:
            if (cfg.parent / "model.safetensors").is_file():
                return cfg.parent
    searched = ", ".join(str(r) for r in roots)
    raise FileNotFoundError(
        f"MOMENT weights for {MODEL_ID} not found in: {searched}. Prefetch {MODEL_ID} "
        "into the red weights HF cache or %USERPROFILE%\\fh2026-cache\\hf "
        "(no runtime download)."
    )


def _load_weights(model, snapshot: Path) -> None:
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(snapshot / "model.safetensors")))


def load(context: int = CONTEXT, horizon_cap: int = HORIZON) -> Callable:
    try:
        from momentfm import MOMENTPipeline
    except ImportError as exc:
        raise ImportError(
            "Could not import momentfm. Install it with `python -m pip install --no-deps momentfm==0.1.4` "
            "so it does not downgrade transformers/torch/numpy/huggingface_hub."
        ) from exc
    import torch

    seq = int(context or CONTEXT)
    cap = int(horizon_cap or HORIZON)
    from workstreams.red_foundation_models.cache import red_hf_home
    red_hf_home().mkdir(parents=True, exist_ok=True)
    snapshot = _find_snapshot()
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    # Explicit config: newer hub mixins drop it when constructing the pipeline.
    model = MOMENTPipeline(config=config, model_kwargs={"task_name": "reconstruction"})
    _load_weights(model, snapshot)
    model.init()
    model.eval()
    seq = int(getattr(model, "seq_len", seq) or seq)

    def forecast(history, past, future, horizon):
        # past/future covariates are accepted for the shared depth contract but unused.
        del past, future
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        target = _vector(history, "history")
        remaining = int(horizon)
        chunks = []
        hist = target
        with torch.inference_mode():
            while remaining > 0:
                take = min(remaining, cap, seq - 1)
                ctx_max = seq - take
                ctx = hist[-ctx_max:] if hist.size > ctx_max else hist
                left = seq - ctx.size - take
                window = np.zeros(seq, dtype=np.float32)
                if ctx.size:
                    window[left:left + ctx.size] = ctx
                # input_mask: 1 = real timestep (context + gap). mask: 1 = observed, 0 = impute.
                input_mask = np.zeros(seq, dtype=np.float32)
                input_mask[left:left + ctx.size + take] = 1.0
                mask = np.zeros(seq, dtype=np.float32)
                if ctx.size:
                    mask[left:left + ctx.size] = 1.0
                x = torch.from_numpy(window).view(1, 1, seq)
                out = model(
                    x_enc=x,
                    input_mask=torch.from_numpy(input_mask).view(1, seq),
                    mask=torch.from_numpy(mask).view(1, seq),
                )
                rec = np.asarray(out.reconstruction.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                if rec.size != seq or not np.isfinite(rec).all():
                    raise ValueError(f"MOMENT reconstruction was invalid: shape={rec.shape}")
                pred = rec[left + ctx.size:left + ctx.size + take]
                if pred.size != take:
                    raise ValueError("MOMENT returned a short imputation")
                chunks.append(pred)
                remaining -= take
                hist = np.concatenate([hist, pred])
        return np.concatenate(chunks)

    return forecast
