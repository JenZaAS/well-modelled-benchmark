# Agent notes

Before running anything that needs well data, check that `data_paths.local.yaml`
exists at the repository root. If it is missing, follow
[docs/SETUP_DATA_PATHS.md](docs/SETUP_DATA_PATHS.md): ask the user for their
local data folder, write the file, and **never commit it**.

Then run `python -m wmb.paths --check` and use `wmb.paths.get_dataset_path(...)`
rather than hard-coded folders.
