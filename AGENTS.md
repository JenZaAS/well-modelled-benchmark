# Agent notes

To add a model, follow **[docs/build-your-model.md](docs/build-your-model.md)**.
The usual run is `python -m benchmark.run --model <name> --test all` (or
`--test rhob_missing`). Create a new test with `python -m benchmark.new_test`.
The older flags `--dataset`, `--scenario` and `--target` still work.

Before running anything that needs well data, check that `data_paths.local.yaml`
exists at the repository root. If it is missing, follow
[docs/SETUP_DATA_PATHS.md](docs/SETUP_DATA_PATHS.md): ask the user for their
local data folder, write the file, and **never commit it**. Then
`python -m wmb.onboarding` (or `python -m wmb.paths --check`). Use
`wmb.paths.get_dataset_path(...)` rather than hard-coded folders.

## Benchmark tests

To score models on a new (dataset, target, scenario, wells) combination, add a YAML test definition instead of a
one-off script: `python -m benchmark.new_test my_test`, edit `benchmark/test_defs/my_test.yaml`, check it with
`python -m benchmark.check_test my_test`, run it with `python -m benchmark.run --model X --test my_test`
(or `--test all`). Guide: [docs/tests.md](docs/tests.md).

To align a new public log dataset to the FORCE names, units and metres-MD
standard, follow [docs/data-alignment.md](docs/data-alignment.md) and run
`python -m wmb.align`. Never commit the data.
