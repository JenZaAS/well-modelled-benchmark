# models/

| Folder | Registered IDs |
| --- | --- |
| [`gardner/`](gardner/README.md) | `black_gardner`, `black_gardner_fixed` |
| [`castagna_mudrock/`](castagna_mudrock/README.md) | `black_castagna_dts` |
| [`greenberg_castagna/`](greenberg_castagna/README.md) | `black_gc_brine_gr` |
| [`wyllie/`](wyllie/README.md) | `black_wyllie_nphi` |
| [`volumetric_density/`](volumetric_density/README.md) | `black_density_nphi` |
| [`vpvs_ratio/`](vpvs_ratio/README.md) | `black_vpvs_ratio` |
| `_shared/` | helpers only: slowness conversion, curve access, calibration, the thin wrapper base class |

Each folder has `equation.py`, `wrapper.py`, a model card (`README.md`), `LICENSE.md` and one
`<id>.model.yaml` per registered ID.
Registration is in [`../contender.py`](../contender.py).

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
