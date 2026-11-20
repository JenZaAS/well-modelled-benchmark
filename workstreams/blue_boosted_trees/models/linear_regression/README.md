# Multilinear regression (`blue_ml_linear`, `blue_ml_linear_trend`)

Ordinary least squares on the other logs and depth. Inputs are clipped to their 0.5 to 99.5 percentile range, missing
values are filled with the training median and flagged, and columns are standardised. Resistivity is trained in log10.
Training rows are thinned evenly to 250,000 per fit. Point predictions only (no uncertainty).

| Registered name | Features |
|---|---|
| `blue_ml_linear` | the other logs + depth |
| `blue_ml_linear_trend` | + 10 m mean above and below each depth, their difference and the local spread |

Not registered here: the variants with stratigraphy (`GROUP` / `FORMATION`) and the context features. Benchmark wells
carry no formation labels. In the study, stratigraphy helped this model most (for example RDEP, R² 0.65 to 0.90), so it is
worth using through the app's "Predict a new well" tab when formation tops are available. See [`ml_study/`](../../ml_study/)
and the [blue README](../../README.md).

The model code is shared with the other ML-study models and lives in `ml_study/` (`models.py`, `features.py`); this folder
only holds the registration wrapper, so saved models (pickles that record that module path) keep loading.

```
python -m benchmark.check_model blue_ml_linear
python -m benchmark.run --model blue_ml_linear --test rhob_missing
```
