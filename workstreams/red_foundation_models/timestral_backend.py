r"""Local CPU inference with Mistral's Timestral-5M (21 MB).

Install in a Python 3.11 venv::

    python -m pip install torch==2.8.0 numpy==2.2.6 safetensors==0.6.2 huggingface_hub==0.35.3 \
        git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff

The inference code is Apache-2.0. The weights are separately licensed:
https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE
The Modified MIT license excludes companies/employers with global consolidated
monthly revenue over USD 20 million in the preceding month. Such users need
separate authorization from Mistral for local inference.

If authorized, explicitly prefetch before running the benchmark::

    export HF_HOME="${FH2026_CACHE:?}/hf"
    HF_HUB_OFFLINE=0 hf download mistralai/Timestral-5M config.json model.safetensors LICENSE \
        --revision bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f

Keep the same HF_HOME for inference. Neither import nor load downloads weights.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def load() -> Callable[
    [np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], int], tuple[np.ndarray, ...]
]:
    from timestral import TimestralForecaster

    model = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-5M",
        revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
        device="cpu",
        local_files_only=True,
    )

    def forecast(history, past, future, horizon):
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 1 or history.size == 0:
            raise ValueError("history must be a nonempty 1D target")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if past.keys() != future.keys():
            raise ValueError("past and future must contain the same covariates")

        rows = []
        for key in sorted(past):
            left = np.asarray(past[key], dtype=np.float32)
            right = np.asarray(future[key], dtype=np.float32)
            if left.shape != history.shape or right.shape != (horizon,):
                raise ValueError(f"{key}: expected past {history.shape} and future ({horizon},)")
            rows.append(np.concatenate((left, right)))
        # A known covariate is one row spanning both context and horizon.
        known = np.stack(rows)[None, :, :] if rows else None
        result = model.forecast(np.ascontiguousarray(history), int(horizon), known_covariates=known)
        return tuple(result.quantile(q)[0, 0].numpy() for q in (0.1, 0.5, 0.9))

    return forecast
