# Models (short reference; details in the pitch deck reference slides)

| Model | Team | Licence | Notes |
|---|---|---|---|
| Gardner (1974) | black | — | ρ ≈ 0.31·Vp^0.25 (Vp in m/s); RHOB from DTC |
| Castagna mudrock / Greenberg-Castagna | black | — | DTS from DTC (Vs from Vp) |
| XGBoost / LightGBM | blue | Apache-2.0 / MIT | one depth sample = one row |
| Chronos-2 (Amazon) | red | Apache-2.0 | 120M; covariates; quantile output |
| MOMENT (CMU) | red | MIT | 40M–385M; masked-patch reconstruction; point output |
| TTM r2 (IBM Granite) | red | Apache-2.0 | from ~1M; exogenous mixer when fine-tuned |

Weights are downloaded to the local Hugging Face cache (`tools/laptop_setup/`), never committed.
Each model folder needs a `LICENSE.md` and licence fields on the class (see `THIRD_PARTY.md`).
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
