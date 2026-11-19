# Contributing: golden rules

We all work on `main`, with or without branches. These rules keep us out of each other's way.

1. **Only edit your own folder** (`workstreams/<your team>/`, or your team's folder in the README table).
   Read and import anything; change only what's yours.
2. **Shared code (`src/wmb/`, `benchmark/`, top-level files) changes only after a heads-up in the chat**,
   in a small commit or pull request. Keep old names working until everyone has pulled.
3. **`git pull --rebase` before every push.** If it conflicts and the file isn't yours, stop and ask the team that works on that folder.
4. **Never commit data, model weights, credentials or big outputs.** Data lives where
   `data_paths.local.yaml` points (that file is gitignored; see [docs/SETUP_DATA_PATHS.md](docs/SETUP_DATA_PATHS.md)).
   Weights in the cache, results in `results/` (gitignored). Notebooks: clear outputs before committing.
5. **Commit small and often**, with a message that says what and why (e.g. `blue: add LightGBM contender`).

## Daily loop

```bash
git pull --rebase
# ...work in your folder...
python -m benchmark.run --dataset synthetic      # still runs?
git add workstreams/<your team>/
git commit -m "<team>: <what changed>"
git pull --rebase && git push
```

Use `git add <your folder>`, not `git add .`, so you never pick up someone else's half-finished files.

## Scratch space

Each workstream folder may have a `scratch/` subfolder: it is gitignored, so put anything personal there.

## Licence

Repository code is MIT. Don't paste in code or text you can't release under MIT, and no company or client code.
Third-party models and datasets keep their own licences. If you add a model, put a `LICENSE` or
`LICENSE.md` in your workstream folder (licence name, URL, and a one-line summary of restrictions),
set `licence_name`, `licence_url` and `licence_restrictions` on the class, and add a row to
[THIRD_PARTY.md](THIRD_PARTY.md). Users and agents must verify a model's licence fits their use
before running or redistributing it.
