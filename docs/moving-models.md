# Moving a model into `models/`

Team members are already working in `workstreams/`. This page only
describes an optional folder layout. **Nobody else's files will be moved
for you.** When you are ready, you move your own code.

## What stays the same

The runner still finds models only in each workstream's
`workstreams/<team>/contender.py`, via that file's `CONTENDERS` list.
A model that is already registered there keeps working. You do not have
to restructure anything to run or score it.

## Optional layout

    workstreams/<team>/
        contender.py              # stays; this is the registration file
        models/
            <model_name>/
                ... your code and a thin wrapper that implements BaseContender ...
                LICENSE.md
                README.md         # optional

- One subfolder per model.
- `LICENSE.md` holds the licence name, URL, and a one-line summary of
  restrictions. Set `licence_name`, `licence_url` and
  `licence_restrictions` on the class as well (see
  [THIRD_PARTY.md](../THIRD_PARTY.md)).
- Weights and checkpoints go in `cache_dir`, never in git.
- Use the shared example [`workstreams/_template_model/`](../workstreams/_template_model/)
  rather than copying a second template into `models/`.

Each team already has an empty `models/` folder with a short README that
repeats this. There is no `models/` folder under `data_prep` (it is not a
model workstream) or under `_template_model` (that folder is the shared
example).

## When you are ready, do this yourself

1. Create `workstreams/<team>/models/<model_name>/`.
2. Move only your own files into it. Do not move or overwrite a teammate's
   files.
3. Add `LICENSE.md` in that subfolder.
4. Keep (or add) a thin wrapper that subclasses `BaseContender`.
5. In the workstream's existing `contender.py`, import the class and
   include it in `CONTENDERS`. Do not remove a working entry until the
   new import runs.
6. `python -m benchmark.check_model <name>`, then
   `python -m benchmark.run --model <name> --test all` (or `--test synthetic_rhob_missing`
   if you do not have the FORCE data yet). See [build-your-model.md](build-your-model.md).

## Why the registry does not scan `models/` yet

Discovery is limited to `workstreams/*/contender.py` on purpose. That is
the explicit convention: a model is in the benchmark only when the team
lists it in `CONTENDERS`. Auto-importing every nested folder would pick
up half-finished work while people are still coding. Nested discovery
can be added later if the team agrees; it is not required for this
layout, and it is not turned on now.
