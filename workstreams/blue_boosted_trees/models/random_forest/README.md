# Random forest (`blue_ml_rf`)

Random forest (60 trees, at least 5 samples per leaf, half of the features tried per split) on the other logs and depth.
Missing values are handled natively. Resistivity is trained in log10. Training rows are thinned evenly to 250,000 per fit.
Point predictions only (no uncertainty).

| Registered name | Features |
|---|---|
| `blue_ml_rf` | the other logs + depth |

Only the base feature set is registered: a random forest with trend or context features takes about 7 minutes per fit
(about 2 hours for all logs) and has not been run in the study. The variant with stratigraphy is in the study and app,
not here, because benchmark wells carry no formation labels. See [`ml_study/`](../../ml_study/) and the
[blue README](../../README.md).

The model code is shared with the other ML-study models and lives in `ml_study/` (`models.py`, `features.py`); this folder
only holds the registration wrapper.

```
python -m benchmark.check_model blue_ml_rf
python -m benchmark.run --model blue_ml_rf --test rhob_missing
```
