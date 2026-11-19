# Third-party models and datasets

Repository **code** is MIT (see `LICENSE`). Models and datasets that did not
originate in this repo keep their own licences. Users and agents must verify
a model's or dataset's licence fits their use before running or redistributing
it. Restricted licences (for example a modified MIT with a company-revenue
limit) may be included if they are declared next to the model.

This is an index, not legal advice. Check the linked text before you rely on it.

## Datasets

Full credit lines and status notes: [docs/data-licences.md](docs/data-licences.md).
FORCE 2020 (the demo set) is also described in [datasets/force2020/README.md](datasets/force2020/README.md).

| Item | Licence / terms | Where recorded |
|---|---|---|
| FORCE 2020 well logs and lithofacies | Logs NLOD 2.0; lithofacies labels CC BY 4.0 (verify the Zenodo record) | `docs/data-licences.md`, `datasets/force2020/` |
| Sodir FactPages | NLOD 2.0 | `docs/data-licences.md` |
| Volve | Equinor Volve data licence (open, with attribution) | `docs/data-licences.md` |
| NLOG | NLOG terms (quote NLOG.NL as the source) | `docs/data-licences.md` |
| Kansas Geological Survey | KGS terms (use the required citation) | `docs/data-licences.md` |

Do not commit raw data or well logs to this repository.

## Models and libraries

Each workstream folder has a `LICENSE.md`. Contender classes also set
`licence_name`, `licence_url` and `licence_restrictions`.

| Item | Team / folder | Licence | Restrictions (one line) |
|---|---|---|---|
| `template_linear` (baseline) | `_template_model` | MIT | none (original repo code) |
| Gardner / Castagna implementations | `black_physics` | MIT | none for this folder's code; methods are published papers |
| `blue_gbt` folder code | `blue_boosted_trees` | MIT | none for this folder's code |
| XGBoost (optional dependency) | blue | Apache-2.0 | see [XGBoost LICENSE](https://github.com/dmlc/xgboost/blob/master/LICENSE) |
| LightGBM (optional dependency) | blue | MIT | see [LightGBM LICENSE](https://github.com/microsoft/LightGBM/blob/master/LICENSE) |
| scikit-learn (optional dependency) | blue | BSD-3-Clause | see [scikit-learn COPYING](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| `red_floor_train_mean` placeholder | `red_foundation_models` | MIT | none (placeholder); replace this entry when a real model is added |
| Chronos-2 (planned) | red | Apache-2.0 | see the upstream model card / licence before use |
| MOMENT (planned) | red | MIT | see the upstream model card / licence before use |
| TTM r2 (planned) | red | Apache-2.0 | see the upstream model card / licence before use |

When you add a model, append a row here, add or update `workstreams/<team>/LICENSE.md`,
and set the three licence fields on the class.
