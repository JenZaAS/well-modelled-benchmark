# XGBoost gradient boosting (`blue_ml_gbt`, `blue_ml_gbt_trend`)

Gradient-boosted trees (XGBoost, 200 trees, depth 6) that predict a log from the other logs and depth. One depth sample
is one row; missing values are handled by the trees. Resistivity is trained in log10. Training rows are thinned evenly to
250,000 per fit. Point predictions only (no uncertainty).

| Registered name | Features |
|---|---|
| `blue_ml_gbt` | the other logs + depth |
| `blue_ml_gbt_trend` | + 10 m mean above and below each depth, their difference and the local spread |

Not registered here: the variants with stratigraphy (`GROUP` / `FORMATION`) and the context features. Benchmark wells
carry no formation labels, and context did not help. They are available in the study and the app, see
[`ml_study/`](../../ml_study/) and the [blue README](../../README.md) for the results.

The model code is shared with the other two ML-study models and lives in `ml_study/` (`models.py`, `features.py`); this
folder only holds the registration wrapper, so saved models (pickles that record that module path) keep loading.

The earlier starter `blue_gbt` (with a quantile-based uncertainty) stays in `../../contender.py`.

```
python -m benchmark.check_model blue_ml_gbt
python -m benchmark.run --model blue_ml_gbt --test rhob_missing
```
