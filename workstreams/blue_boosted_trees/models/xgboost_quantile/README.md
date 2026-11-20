# XGBoost with uncertainty (`blue_gbt`)

The blue team's starter model: XGBoost (200 trees, depth 6) on the other logs and depth, one depth sample per row, with
two extra quantile models (16% and 84%) that give an uncertainty `sigma = (q84 - q16) / 2`. This is the only blue model
that returns an uncertainty, so it is the one to look at for coverage and sharpness.

Unchanged by the move into this folder: the code and the registered name `blue_gbt` (so existing results and viewer
entries keep working). The newer point-prediction models of the ML study (`blue_ml_gbt`, `blue_ml_linear`, `blue_ml_rf`
and their trend variants) are in the sibling folders.

```
python -m benchmark.check_model blue_gbt
python -m benchmark.run --model blue_gbt --test rhob_missing
```
