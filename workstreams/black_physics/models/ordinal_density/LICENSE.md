# Licence

- Name: MIT (the wrapper and the 15-bin heads)
- URL: https://opensource.org/license/mit (and the repository root `LICENSE`)
- Restrictions: none for this folder's code. Each row also runs a base model that keeps its own licence:
  - `black_ordinal_timestral` runs Mistral Timestral-5M. Its weights use a Modified MIT licence: keep the attribution
    and permission notice, and get separate authorization if your company's or employer's global consolidated revenue
    exceeded USD 20 million in the preceding month. See
    [`../../../red_foundation_models/models/timestral/LICENSE.md`](../../../red_foundation_models/models/timestral/LICENSE.md).
  - `black_ordinal_tspulse_finetune` runs IBM Granite TSPulse R1, Apache-2.0 (weights and granite-tsfm code).

The heads are fitted locally on FORCE 2020 training wells and are not distributed with the repository. See also
the workstream notice in [`../../LICENSE.md`](../../LICENSE.md). Users and agents must check that each licence fits
their use before running or redistributing a model.
