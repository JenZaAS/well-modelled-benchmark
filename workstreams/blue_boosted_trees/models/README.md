# models/

Optional layout for this team's models. You do not have to move anything
now. The existing `../contender.py` and its `CONTENDERS` list still work,
and that is what the runner uses today.

When you are ready to tidy a model, give it its own subfolder (you move
the files yourself):

    models/<model_name>/
        # your wrapper and model code
        LICENSE.md          # licence name, URL, one-line restrictions
        README.md           # optional

- One subfolder per model.
- Put a thin wrapper here that implements the station contract
  (`wmb.contender.BaseContender`) and a `LICENSE.md` in that subfolder.
- Weights, checkpoints and large artefacts go in `cache_dir` (see
  `data_paths.local.yaml` and `wmb.paths.cache_dir()`), never in the repo.
- Do not copy a template into this folder. Start from the shared example
  at [`workstreams/_template_model/`](../../_template_model/).

**Registration stays explicit and backwards compatible.** Import the class
in this workstream's `contender.py` and add it to `CONTENDERS`. The registry
does not auto-discover nested `models/` folders, so work in progress here
is not picked up until you register it.

How to move code you already have, when you choose to:
[`docs/moving-models.md`](../../../docs/moving-models.md).
