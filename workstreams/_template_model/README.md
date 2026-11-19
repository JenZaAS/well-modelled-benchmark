# Template model

A working example of a model for the benchmark (see [Build your model](../../docs/build-your-model.md) and [`docs/benchmark.md`](../../docs/benchmark.md)).
Copy this folder, rename the class and `name`, and change the model.

`template_linear` (numpy only): for RHOB, DTC and DTS it fits `ln(target)` on transformed inputs with
`numpy.linalg.lstsq`: the Gardner relation (`ln Vp` from DTC for RHOB, `ln RHOB` for DTC, `ln DTC` and
`ln RHOB` for DTS) plus GR, NPHI and depth. Rows with missing inputs use a fit on the inputs that are present,
and the training mean when none are. Uncertainty is the residual standard deviation returned as `std`
(Gaussian 90% and 50% intervals). It works for both scenarios (`missing_curve`, `gap`) and ignores the observed
target in gap tasks.

```
python -m benchmark.check_model template_linear
python -m benchmark.run --model template_linear --test all
```

The older flags `--dataset`, `--scenario` and `--target` still work. To add a test: `python -m benchmark.new_test`.

## How to add your model

1. Put your model in `workstreams/<team>/contender.py` (copy `contender.py` from here). Do not edit `src/wmb/` or `benchmark/`.
2. Subclass `wmb.contender.BaseContender`. Set `name` (unique), `workstream`, `targets` (any of `RHOB`, `DTC`, `DTS`),
   `required_inputs` and `optional_inputs` (canonical curve names, plus `DEPTH`).
3. Implement `fit(train_wells, target)`; train wells still contain the target. Implement `predict_task(task)`
   (or the older `predict(well, target)`) returning `Prediction` with the same length as `task.depth`:
   `p50` (or `mean`), and optionally `std` or `p05`/`p95`. Use NaN where you cannot predict; never raise on missing inputs.
4. For `gap` tasks the observed target is in `task.target_observed` (NaN on `task.predict_mask`). Predict where
   `predict_mask` is True (other samples are not scored). Never read the hidden target.
5. Use canonical units (RHOB g/cm3, DTC and DTS us/ft). Do not convert and do not load data yourself in `predict`.
6. List the class in `CONTENDERS`.
7. Run `python -m benchmark.check_model <name>`, then `python -m benchmark.run --model <name> --test all` (use `--test synthetic_rhob_missing` if you do not have FORCE data yet).
8. When those pass, run a FORCE test (`--test rhob_missing`) and `python -m benchmark.scoreboard`.
9. Note decisions that matter for reproducibility (data, preprocessing, model choice) briefly in your team README or `docs/`.
10. Add a `LICENSE` or `LICENSE.md` in your model folder with the licence name, URL, and a one-line summary of restrictions. Set `licence_name`, `licence_url` and `licence_restrictions` on the class. Restricted licences may be used if declared; users must verify the licence fits their use. `python -m benchmark.check_model <name>` warns if the file is missing. Update [THIRD_PARTY.md](../../THIRD_PARTY.md).
