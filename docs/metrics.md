# Metrics (implemented in `src/wmb/scoring.py`)

| Score | Definition | Better |
|---|---|---|
| **Accuracy** | MAE, RMSE, R² of the predicted mean vs the true log | lower MAE/RMSE, higher R² |
| **Coverage** | share of true values inside mean ±1σ (ideal 0.683) and ±2σ (ideal 0.954); older runs, still computed when `std` is given | close to ideal |
| **Interval coverage** | `cover_90`, `cover_50`: share of truth inside the 90% / 50% interval (ideals 0.90, 0.50). 90% is `[p05, p95]`, else `p50 ± 1.6449σ`; 50% is `[p25, p75]` if given, else `p50 ± 0.6745σ`, else omitted | close to ideal |
| **Sharpness** | `interval_width_90` (mean `p95 − p05`, or Gaussian equivalent), and mean predicted σ; only meaningful when coverage is right | lower |
| **Pinball** | `pinball_05`, `pinball_50`, `pinball_95` when `p05`/`p95` are given | lower |
| **Uncertainty flag** | `has_uncertainty` is false for point-only predictions, which get no coverage or sharpness | n/a |
| **CRPS** | Gaussian continuous ranked probability score (needs `std`): accuracy and calibration in one number | lower |
| **Speed** | wall-clock `load_s`, `fit_s`, `predict_s`, `predict_s_per_well`, `predict_s_batch`, `samples_per_s`; final run on the same laptop CPU | lower |
| **Lithology** | accuracy and FORCE 2020 penalty score (matrix from the data folder) | higher |

Resistivity curves are modelled in log10 (`wmb.curves.LOG_SCALE`).
Calibration and sharpness follow Gneiting & Raftery (JASA 2007).
