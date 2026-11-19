# Models (short reference; details in the pitch deck reference slides)

| Model | Team | Licence | Notes |
|---|---|---|---|
| Gardner (1974) | black | — | ρ ≈ 0.31·Vp^0.25 (Vp in m/s); RHOB from DTC |
| Castagna mudrock / Greenberg-Castagna | black | — | DTS from DTC (Vs from Vp) |
| XGBoost / LightGBM | blue | Apache-2.0 / MIT | one depth sample = one row |
| Chronos-2 (Amazon) | red | Apache-2.0 | 120M; covariates; quantile output |
| MOMENT (CMU) | red | MIT | 40M–385M; masked-patch reconstruction; point output |
| TTM r2 (IBM Granite) | red | Apache-2.0 | from ~1M; exogenous mixer when fine-tuned |

Weights are downloaded to the local Hugging Face cache (`tools/laptop_setup/`), never committed.
Not allowed: models whose weights are non-commercial (e.g. Moirai) or incompatible with MIT.
