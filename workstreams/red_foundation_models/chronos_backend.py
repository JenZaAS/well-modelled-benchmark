"""Local CPU inference using chronos-forecasting and a cached amazon/chronos-2.

Install ``chronos-forecasting==2.3.2`` with ``transformers>=4.57.6,<5``.
Provision the checkpoint with ``hf download amazon/chronos-2`` in the same
Hugging Face cache as the benchmark (see the laptop setup's HF_HOME).
Importing this module does not import Chronos or access the cache.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _vector(values: np.ndarray, name: str, length: int | None = None) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{name} must be a nonempty 1D array")
    if length is not None and values.size != length:
        raise ValueError(f"{name} must have length {length}, got {values.size}")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    return np.ascontiguousarray(values, dtype=np.float32)


def load() -> Callable[
    [np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Load cached weights once; the returned callable takes only local arrays."""
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download("amazon/chronos-2", local_files_only=True)
    pipeline = Chronos2Pipeline.from_pretrained(
        snapshot, local_files_only=True, device_map="cpu", dtype=torch.float32,
    )

    def forecast(history, past, future, horizon):
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if past.keys() != future.keys():
            raise ValueError("past and future must contain the same covariate names")
        target = _vector(history, "history")
        task = {
            "target": target,
            "past_covariates": {k: _vector(v, f"past[{k!r}]", target.size) for k, v in past.items()},
            "future_covariates": {k: _vector(v, f"future[{k!r}]", horizon) for k, v in future.items()},
        }
        with torch.inference_mode():
            quantiles, _ = pipeline.predict_quantiles(
                [task], prediction_length=int(horizon), quantile_levels=[0.1, 0.5, 0.9],
            )
        # One task, one target: [task][target, forecast step, quantile].
        q10, q50, q90 = quantiles[0][0].cpu().numpy().T
        return q10, q50, q90

    return forecast
