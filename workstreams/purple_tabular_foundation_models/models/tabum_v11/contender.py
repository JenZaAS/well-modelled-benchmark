"""Frozen tabUM v1.1 regressor with the workstream's matched context."""

import numpy as np

from ...depth_rows import DepthRowContender


class TabUMContender(DepthRowContender):
    """Predict native conditional quantiles without adapting model weights."""

    name = "purple_tabum_v11"
    licence_name = "MIT"
    licence_url = "https://github.com/heldernoid/tabum/blob/d119a5d3e11d144cd9c75b8609586a076c32db1e/LICENSE"
    licence_restrictions = "Retain the upstream copyright and permission notice."
    checkpoint = "helmo/tabum-v1.1"
    revision = "819c5927aeb8e574facc3d339a91e37bdec4bb9a"

    def _fit_backend(self, x: np.ndarray, y: np.ndarray) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from tabum.inference import TabUMRegressor
            from tabum.model import TabUM
        except ImportError as exc:
            raise ImportError("Install the tabUM model's dependencies in a Python 3.13 environment") from exc

        path = snapshot_download(self.checkpoint, revision=self.revision,
                                 allow_patterns=["config.json", "model.safetensors"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TabUM.from_pretrained(path, device=device)
        self._reg = TabUMRegressor(model=model, device=device, n_ensemble=1).fit(x, y.ravel())

    def _predict_batch(self, x: np.ndarray) -> np.ndarray:
        import torch

        # Pinned estimator's predict_quantile repeats _out for each probability.
        # Reuse the same distribution and its native target-unit decoder instead.
        with torch.inference_mode():
            distribution = self._reg._out(x)
            return np.column_stack([
                self._reg.model.predict_quantile(distribution, q)[0].float().cpu().numpy()
                for q in (0.05, 0.50, 0.95)
            ])
