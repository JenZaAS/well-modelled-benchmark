"""Pinball / split-conformal helpers for the TTM quantile fine-tune."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 90% central band used by the benchmark (cover_90 / interval_width_90).
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
META_NAME = "white_meta.json"  # name used by existing checkpoints; do not rename


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss; same definition as ``wmb.scoring._pinball``."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = np.isfinite(y) & np.isfinite(q)
    if not ok.any():
        return float("nan")
    d = y[ok] - q[ok]
    return float(np.maximum(tau * d, (tau - 1) * d).mean())


def conformal_width(y: np.ndarray, q05: np.ndarray, q95: np.ndarray, alpha: float = 0.10) -> float:
    """Split-conformal residual for conformalized quantile regression (CQR).

    Scores are ``max(q05 - y, y - q95)``. The returned width ``q_hat`` is the
    finite-sample ``(1-alpha)`` conformal quantile, applied as
    ``q05' = q05 - q_hat``, ``q95' = q95 + q_hat``. Calibration must use only
    held-out *training* wells — never test wells.
    """
    y, q05, q95 = (np.asarray(a, dtype=float).reshape(-1) for a in (y, q05, q95))
    s = np.maximum(q05 - y, y - q95)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    n = int(s.size)
    # Standard split-conformal index: ceil((n+1)(1-alpha))th order statistic.
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    if k >= n:
        return float(np.max(s))
    k = max(k, 0)
    return float(np.sort(s)[k])


def apply_conformal(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray, width: float):
    """Widen (or shrink) a 90% band by ``width``, clipped so q05 <= q50 <= q95."""
    q05 = np.asarray(q05, dtype=np.float32)
    q50 = np.asarray(q50, dtype=np.float32)
    q95 = np.asarray(q95, dtype=np.float32)
    lo = np.minimum(q05 - width, q50)
    hi = np.maximum(q95 + width, q50)
    return lo, q50, hi


def meta_path(folder: Path) -> Path:
    return Path(folder) / META_NAME


def write_meta(folder: Path, payload: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = meta_path(folder)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_meta(folder: Path | str | None) -> dict:
    if not folder:
        return {}
    path = meta_path(Path(folder))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
