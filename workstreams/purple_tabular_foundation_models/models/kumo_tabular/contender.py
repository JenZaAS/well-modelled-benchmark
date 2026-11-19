"""Depth-row in-context regression with NVIDIA Kumo-Tabular."""

from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction
from wmb.contract import INPUT_CURVES, TARGET_CURVES
from wmb.io import Well


def _sample(train: list[Well], target: str, budget: int) -> list[tuple[Well, np.ndarray]]:
    eligible = []
    for well in sorted(train, key=lambda w: (w.dataset, w.name)):
        if target in well.curves:
            rows = np.flatnonzero(np.isfinite(well.curves[target]) & np.isfinite(well.depth))
            if rows.size:
                eligible.append((well, rows))
    if not eligible:
        raise ValueError(f"No finite training labels with depth for {target}")
    # Allocate round-robin so short wells give unused capacity back to longer wells.
    counts = np.zeros(len(eligible), dtype=int)
    sizes = np.array([rows.size for _, rows in eligible])
    remaining = min(budget, int(sizes.sum()))
    while remaining:
        active = np.flatnonzero(counts < sizes)[:remaining]
        counts[active] += 1
        remaining -= len(active)
    return [(well, rows[np.linspace(0, rows.size - 1, n, dtype=int)])
            for (well, rows), n in zip(eligible, counts) if n]


def _features(well: Well, rows: np.ndarray | slice, columns: tuple[str, ...]) -> np.ndarray:
    depth = well.depth[rows]
    data = np.column_stack([depth, *(well.curves[c][rows] if c in well.curves else
                                     np.full(depth.shape, np.nan) for c in columns)]).astype(np.float32)
    data[~np.isfinite(data)] = np.nan
    return data


class KumoTabularContender(BaseContender):
    """Use a frozen pretrained regressor; fit caches examples, not gradients."""

    name = "purple_kumo_tabular"
    workstream = "purple_tabular_foundation_models"
    targets = list(TARGET_CURVES)
    optional_inputs = list(INPUT_CURVES)
    licence_name = "OpenMDW-1.1"
    licence_url = "https://huggingface.co/nvidia/Kumo-Tabular"
    licence_restrictions = "Preserve the weight licence and notices when redistributing."

    def __init__(self, context_rows: int = 1024, batch_rows: int = 256) -> None:
        if min(context_rows, batch_rows) < 1:
            raise ValueError("Row limits must be positive")
        self.context_rows = context_rows
        self.batch_rows = batch_rows
        self._target: str | None = None
        self._columns: tuple[str, ...] = ()

    def fit(self, train_wells: list[Well], target: str) -> None:
        """Cache a reproducible, well-balanced sample of finite training labels."""
        self._target = None
        if not self.supports(target):
            raise ValueError(f"Unsupported target: {target}")
        selected = _sample(train_wells, target, self.context_rows)
        self._columns = tuple(c for c in INPUT_CURVES if c != target and any(
            c in w.curves and np.isfinite(w.curves[c][rows]).any() for w, rows in selected))
        x = np.concatenate([_features(w, rows, self._columns) for w, rows in selected])
        y = np.concatenate([w.curves[target][rows] for w, rows in selected]).astype(np.float32)
        self._fit_backend(x, y[:, None])
        self._target = target

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

    def predict(self, well: Well, target: str) -> Prediction:
        """Predict native 5th, 50th and 95th percentiles in original row order."""
        if target != self._target:
            raise ValueError(f"Fit on {target} before predicting")
        result = np.empty((len(well.depth), 3), dtype=np.float32)
        for start in range(0, len(result), self.batch_rows):
            rows = slice(start, start + self.batch_rows)
            result[rows] = self._predict_batch(_features(well, rows, self._columns))
        return Prediction(p05=result[:, 0], p50=result[:, 1], p95=result[:, 2])
