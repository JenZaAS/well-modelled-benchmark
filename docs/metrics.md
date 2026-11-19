# Metrics (implemented in `src/wmb/scoring.py`)

| Score | Definition | Better |
|---|---|---|
| **Accuracy** | MAE, RMSE, R² of the predicted mean vs the true log | lower MAE/RMSE, higher R² |
| **Coverage** | share of true values inside mean ±1σ (ideal 0.683) and ±2σ (ideal 0.954) | close to ideal |
| **Sharpness** | mean predicted σ (narrow bands), only meaningful when coverage is right | lower |
| **CRPS** | Gaussian continuous ranked probability score: accuracy and calibration in one number | lower |
| **Speed** | wall-clock fit and predict seconds, and predict seconds per 1,000 samples; final run on the same laptop CPU | lower |
| **Lithology** | accuracy and FORCE 2020 penalty score (matrix from the data folder) | higher |

Resistivity curves are modelled in log10 (`wmb.curves.LOG_SCALE`).
Calibration and sharpness follow Gneiting & Raftery (JASA 2007).
