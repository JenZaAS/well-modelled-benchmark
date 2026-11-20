# Prompt: fine-tune TTM r2 for well-log gap filling (FH2026)

Use this as the brief for a coding agent (Copilot CLI, coder `claude-sonnet-5.5`,
critic `gpt-6.2-sol`) working in the well-modelled-benchmark repo on the laptop CPU.

## Goal
Fine-tune IBM TTM r2 (Tiny Time Mixer) so it fills a missing or gapped well-log
curve better than zero-shot TTM, and score it with the existing benchmark runner
so it is directly comparable with Chronos-2, TimesFM 2.5, Timestral, blue_gbt and
the physics baselines.

## Wells: never tune on the wells we score
- Train only on FORCE 2020 **training** wells.
- Build the exclusion list first: every well that appears as a test well in any
  test definition under `benchmark/test_defs/` (rhob/dts/dtc gap50 and missing).
  None of these may be used for training or early stopping.
- Hold out 2-3 further training wells as a validation set for early stopping.
- Write the final train / validation / excluded well lists to a small JSON next
  to the run outputs, so the split can be checked.

## Experiments, in order
1. **Small timing test (start here):** one target (DTS; DTC if DTS has too few
   training wells), 3 training wells.
   a. Well by well: start from the pretrained checkpoint for each well and log
      wall-clock time, rows (depth samples) used, epochs and final validation loss.
   b. Combined: one run on the same 3 wells together, timed the same way.
   Report a timing table so we can estimate the cost for more wells.
2. **One model for all curves (target design):** a single multivariate TTM with
   input channels such as GR, RHOB, NPHI, DTC, DTS, RDEP and CALI. During training,
   randomly hide one target curve (DTS, DTC or RHOB) per window, over many
   training wells, and train the model to reconstruct it from the others
   (covariates). At test time the same model fills whichever curve is missing.
3. Scale up the well count only if the timings from step 1 allow it before the
   hard stops.

## Scoring
- Wrap the fine-tuned model as a contender class (licence fields set:
  TTM r2 is Apache-2.0; the `red_ttm` wrapper in this folder) and run it with
  `python -m benchmark.run` on the
  matching gap50 and missing tests.
- Compare MAE, cover90 and runtime against zero-shot TTM and Chronos-2.

## Rules
- CPU only; one heavy process on the laptop at a time.
- Weights go in `workstreams/red_foundation_models/weights/` (git-ignored; see
  `finetune_ttm.py` in this folder) and are
  never committed. Do not change torch, transformers or scikit-learn versions.
- Seed everything and log the exact config, so a run can be repeated.
- Stop at the agreed hard stop and report partial results rather than overrun.
