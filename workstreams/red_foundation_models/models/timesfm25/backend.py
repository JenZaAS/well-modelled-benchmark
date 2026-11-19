"""Offline TimesFM 2.5 inference; no model dependencies at discovery time."""
from collections.abc import Callable

import numpy as np

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def load(context: int, max_horizon: int) -> Callable:
    """Load the checkpoint once; each call fits XReg only on its visible history."""
    import timesfm
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        snapshot, local_files_only=True, torch_compile=False,
    )
    model.compile(timesfm.ForecastConfig(
        max_context=context, max_horizon=max_horizon, return_backcast=True,
        use_continuous_quantile_head=True, fix_quantile_crossing=True, infer_is_positive=False,
    ))

    def forecast(history, past, future, horizon):
        target = _vector(history, "history")
        if target.size > context:
            raise ValueError(f"history exceeds the configured context ({context})")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= max_horizon:
            raise ValueError(f"horizon must be an integer in [1, {max_horizon}]")
        if past.keys() != future.keys():
            raise ValueError("past and future must contain the same covariate names")
        covariates = {}
        for key in sorted(past):
            left, right = _vector(past[key], key), _vector(future[key], key)
            if left.shape != target.shape or right.shape != (horizon,):
                raise ValueError(f"{key}: expected past {target.shape} and future ({horizon},)")
            covariates[key] = [np.concatenate((left, right))]
        if covariates:
            _, quantiles = model.forecast_with_covariates(
                inputs=[target], dynamic_numerical_covariates=covariates,
                xreg_mode="xreg + timesfm", ridge=1.0, force_on_cpu=True,
            )
        else:
            _, quantiles = model.forecast(horizon=horizon, inputs=[target])
        # Channel zero is not P10; the next nine are deciles. Discard any backcast.
        return np.asarray(quantiles[0])[-horizon:, [1, 5, 9]].T

    return forecast
