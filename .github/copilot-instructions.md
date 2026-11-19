
## PROJECT OVERVIEW

This is an experimental, hackathon-style project to build a benchmark framework for using AI models to predict missing well log data. The goal is to deliver a working framework in a short amount of time with an AI agent workforce. Agents may work in parallel on separate workstreams, so preserve clear ownership boundaries while keeping all approaches comparable.

## REPOSITORY STRUCTURE AND WORKSTREAM BOUNDARIES

- `workstreams/` contains four independently owned areas: `data_prep/`, `red_foundation_models/`, `blue_boosted_trees/`, and `black_physics/`. An agent assigned to one workstream may edit only that workstream's folder. Do not edit another workstream's files or make changes there on its behalf.
- Workstreams may read and reuse functionality elsewhere in the repository, but must not depend on, import, or reuse implementation code from another workstream. In particular, keep cross-workstream integrations behind the shared interfaces and artifacts described below.
- Shared, reusable framework code lives in `src/wmb/`: standardized curve names and the `Well` representation, data loaders, the contender interface, scoring, splits, paths, and synthetic data. `benchmark/` provides the common contender registry, runner, and scoreboard. Prefer these shared facilities over workstream-specific duplicates.
- Other reusable repository areas include `tools/` for data plumbing, `viewer/` for manifests and visualization, and `docs/` for benchmark definitions, metrics, and data licences. Document important decisions made by the agents in `docs/`. Data belongs outside the repository in the configured data/cache locations; generated results belong in `results/`.
- Shared code (`src/wmb/`, `benchmark/`, top-level files, and other shared areas) is not owned by an individual workstream. Before changing it, coordinate with the affected contributors, keep the change small and backward-compatible where practical, and update directly affected callers or documentation.

## COMMON BENCHMARK CONTRACT

All model types must use the same benchmark setup rather than introducing separate data or evaluation paths. Load data through `wmb` into the shared `Well` format and use the standard core curve names. Implement contenders using `wmb.contender.BaseContender` (or its documented protocol): `fit(train_wells, target)` and `predict(well_without_target, target) -> Prediction`. The benchmark runner supplies the same dataset and well-level split, hides the target at prediction time, and scores each contender consistently. Keep model-specific preprocessing inside its workstream while preserving this common input/output contract.

Briefly document consequential workstream decisions—such as data sources and preparation, curve mapping, model selection, preprocessing, and input representations—where they are useful to later contributors. Keep these notes concise; prioritize decisions that affect reproducibility or comparisons.

## COPILOT GENERAL CODE CREATION EDITS AND REFACTORING OPERATIONAL GUIDELINES

Use the instructions in the <prompt-instructions>[code.prompt.md](./prompts/code.prompt.md)</prompt-instructions> file as general guidance for code generation.

## COPILOT PYTHON CODE CREATION EDITS AND REFACTORING OPERATIONAL GUIDELINES

Use the instructions in the <prompt-instructions>[code-python.prompt.md](./prompts/code-python.prompt.md)</prompt-instructions> file as guidance for python code generation.

## DATA LOCATION

Well data is not in git. Before running anything that needs data, ensure
`data_paths.local.yaml` exists at the repository root. If it is missing, follow
[docs/SETUP_DATA_PATHS.md](../docs/SETUP_DATA_PATHS.md): ask the user for their
local data folder, write the file, and never commit it. Then run
`python -m wmb.paths --check` and resolve folders with `wmb.paths.get_dataset_path`.
