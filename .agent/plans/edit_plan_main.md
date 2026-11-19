# Proposed edit plan: local Timestral dashboard

Working with: red foundation models workstream only; no changes to the shared viewer.
Total planned edits: four, in this order.

1. Add `dashboard_data.py` beside the red experiment. Reuse shared dataset loaders, well split, gap masking, contender and scoring. Return split counts, observed/masked/scored sample counts, paired interpolation and model scores, and aligned depth tracks. Keep scoring truth separate from model inputs. No weight updates.
2. Add `dashboard.py`, a localhost-only Streamlit app using that data layer. The benchmark tab displays split and validation statistics plus observed, deliberately hidden, predicted and uncertainty tracks. The inference tab reads a user-selected local LAS or FORCE-format CSV through shared loaders. Do not mix new wells into training or compute accuracy for unknown targets. Disable telemetry and do not persist input logs.
3. Add `test_dashboard_data.py` with deterministic synthetic wells and a stub forecast; cover target masking, differing split sizes, missing truth and aligned depth tracks. Smoke-test the app locally.
4. Document optional pinned dashboard dependencies, local launch, supported formats and interpretation in the red README. Keep Timestral opt-in and its weights outside git.

Before: the viewer plots observed data and the experiment prints aggregate scores. After: a separate local dashboard links comparable validation statistics to individual depth tracks and supports new-well inference. Implementation follows approval of this scope.