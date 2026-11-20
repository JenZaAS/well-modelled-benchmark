# Models (short reference; details in the pitch deck reference slides)

To add a model, start with **[Build your model](build-your-model.md)**:
`python -m benchmark.check_model <name>`, then
`python -m benchmark.run --model <name> --test all`.

| Model | Team | Licence | Uses other logs? | Notes |
|---|---|---|---|---|
| Gardner (`black_gardner`, `black_gardner_fixed`) | black | MIT (original code) | Only DTC | ρ ≈ 0.31·Vp^0.25 (Vp in m/s); RHOB from DTC |
| Castagna mudrock (`black_castagna_dts`) | black | MIT (original code) | Only DTC | DTS from DTC (Vs from Vp) |
| Greenberg-Castagna brine mixture (`black_gc_brine_gr`) | black | MIT (original code) | Only DTC and GR | DTS from DTC and GR |
| Wyllie time average (`black_wyllie_nphi`) | black | MIT (original code) | Only NPHI | DTC from NPHI |
| Density from neutron (`black_density_nphi`) | black | MIT (original code) | Only NPHI | RHOB from NPHI (volumetric mixing) |
| Constant Vp/Vs ratio (`black_vpvs_ratio`) | black | MIT (original code) | Only DTC | DTS from DTC; control model |
| XGBoost / LightGBM (`blue_gbt`) | blue | Apache-2.0 / MIT | Yes | one depth sample = one row |
| XGBoost gradient boosting (`blue_ml_gbt`, `blue_ml_gbt_trend`) | blue | MIT (original code); XGBoost Apache-2.0 | Yes | one depth sample = one row, point predictions; `_trend` adds 10 m above/below features; stratigraphy variants only in the blue app |
| Multilinear regression (`blue_ml_linear`, `blue_ml_linear_trend`) | blue | MIT (original code); scikit-learn BSD-3-Clause | Yes | clipped, standardised inputs, point predictions |
| Random forest (`blue_ml_rf`) | blue | MIT (original code); scikit-learn BSD-3-Clause | Yes | point predictions; slow to fit (about 100 s per log) |
| Chronos-2 (Amazon, `red_chronos2`) | red | Apache-2.0 | Yes (covariates) | 120M; quantile output |
| TimesFM 2.5 + XReg (`red_timesfm25`) | red | Apache-2.0 | Yes (covariates) | quantile output |
| Timestral-5M (`red_timestral`) | red | Modified MIT weights; Apache-2.0 inference code | Yes (covariates) | licence restricts use by larger companies; see its `LICENSE.md` |
| Training-mean floor (`red_floor_train_mean`) | red | — | No | no-model floor: the training mean of the target |
| MOMENT (CMU, `red_moment`) | red | MIT | No (target-only; covariates accepted but unused) | 40M–385M; masked-patch reconstruction; p10/p90 interval from a training-set residual interval; deprecated alias `white_moment` |
| TTM r2 (IBM Granite, `red_ttm`) | red | Apache-2.0 | No (target-only; covariates accepted but unused) | from ~1M; can use other logs through its exogenous mixer after fine-tuning, which this wrapper does not do yet; deprecated alias `white_ttm` |
| Linear template (`template_linear`) | baseline | — | Yes | reference model; scores in the viewer are relative to it |

Physics and empirical models use only the curves listed in the column; each typically predicts a single property.
TTM and MOMENT, as wired today, forecast from the target alone. On whole-well tests (no visible target) this
reduces to an unconditional prior taken from a training-well demonstration of the target, so their scores there
say little about the models themselves.

Weights are downloaded to the local Hugging Face cache (`tools/laptop_setup/`), never committed.
Each model folder needs a `LICENSE.md` (or `LICENSE` / `LICENSE.txt`) and licence fields on the class (see `THIRD_PARTY.md`). `python -m benchmark.check_model` looks in that folder; a licence only in the team folder is a warning, not a failure.
Restricted licences (including a modified MIT with a company-revenue limit) may be included if
they are declared. Users and agents must verify a model's licence fits their use before running
or redistributing it.

## model.yaml: the inputs of a model

Each model declares which logs it needs, per target, in `workstreams/<workstream>/model.yaml`
(see `workstreams/_template_model/model.yaml`). The viewer's **Prediction** tab uses it to decide which
models can run on a well and which inputs each one gets; the user never picks input logs.

```yaml
name: template_linear        # the contender name
licence: MIT
targets:
  RHOB:
    required: [DTC]          # must exist in the well, else the model is greyed out
    optional: [GR, NPHI]     # used when present
```

- Curve names are canonical (`GR, CALI, RDEP, RMED, RHOB, NPHI, DTC, DTS, PEF`). `DEPTH` is always present; do not list it.
- Allowed keys: `name`, `licence`, `targets` (non-empty) and per target `required`, `optional`. A target cannot be its own
  input, and a curve cannot be both required and optional. Names outside the core set only warn.
- A workstream with several models uses `<model name>.model.yaml` per model; a plain `model.yaml` whose `name` is another model is ignored.
- **Fallback:** without a model.yaml the class metadata is the spec: `targets` (none = any), `required_inputs` / `inputs`,
  `optional_inputs` and `licence_name`, the same for every target. `python -m benchmark.check_model <name>` warns about it and fails on an invalid yaml.
- Eligibility (`wmb.model_spec.eligible`): the model supports the target and every required curve has data in the well.
- A model sees only its declared inputs (required + optional that exist); display tracks in the viewer never change them.
- `POST /api/predict` is read-only (no files under `results/`), localhost only; one failing or slow model returns an error for that model only.
