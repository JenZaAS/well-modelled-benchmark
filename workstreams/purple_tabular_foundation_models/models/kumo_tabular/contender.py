"""Depth-row in-context regression with NVIDIA Kumo-Tabular."""

import numpy as np

from ...depth_rows import DepthRowContender


class KumoTabularContender(DepthRowContender):
    """Use the frozen Kumo small regressor."""

    name = "purple_kumo_tabular"
    licence_name = "OpenMDW-1.1"
    licence_url = "https://huggingface.co/nvidia/Kumo-Tabular"
    licence_restrictions = "Preserve the weight licence and notices when redistributing."

    def _fit_backend(self, x: np.ndarray, y: np.ndarray) -> None:
        try:
            import sdm
            import torch
        except ImportError as exc:
            raise ImportError("Install the purple tabular workstream dependencies before fitting Kumo") from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = sdm.models.KumoTabular(task="regression", size="small", device=self._device)
        self._model.fit(torch.as_tensor(x, device=self._device), torch.as_tensor(y, device=self._device),
                        num_estimators=1, generator=torch.Generator(device=self._device).manual_seed(2020))

    def _predict_batch(self, x: np.ndarray) -> np.ndarray:
        import torch

        table = self._model.predict(torch.as_tensor(x, device=self._device))
        return table[:, ["q050", "q500", "q950"]].to_pandas().to_numpy()
