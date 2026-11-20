# Granite gap rows (model card)

Opt-in (`WMB_BLACK_GRANITE=1`) station rows for reconstructing a hidden interval of bulk density (RHOB) from the
measured density above and below it. The experiment that trains, calibrates and scores them is described in
[../../granite/README.md](../../granite/README.md); results are in [../../granite/RESULTS.md](../../granite/RESULTS.md).

| ID | What it does | Inputs |
| --- | --- | --- |
| `black_ttm_r3_frozen` | IBM TTM-R3 lite (`512-336-dec-512-lite-r3`, rev `226cdc7`): forecast down from the shallow flank and up from the reversed deep flank (≤ 512 samples each), blend the two by depth position | RHOB flanks |
| `black_ttm_r3_tuned` | same, trend and residual prediction heads fine-tuned (pinball loss) on FORCE fitting wells | RHOB flanks |
| `black_tspulse_frozen` | IBM TSPulse R1 (`tspulse-hybrid-dualhead-512-p8-r1`, rev `b121645`): one 512-sample window around the gap, the gap passed as a user mask; read the reconstruction | RHOB flanks |
| `black_tspulse_tuned` | same, time-domain reconstruction head (200 parameters) fine-tuned (MAE) | RHOB flanks |
| `black_ttm_r3_ft` | TTM-R3 after the stronger fine-tune ([finetune.py](../../granite/finetune.py)): prediction heads retrained on 13,824 A gaps with the blended two-sided loss; the part trained and the learning rate were picked on B | RHOB flanks |
| `black_tspulse_ft` | TSPulse after the stronger fine-tune: the whole model (1.07 M weights) retrained on 13,824 A gaps; picked on B | RHOB flanks |
| `black_linear_gap` | straight line in depth between the nearest measured values above and below (the station's reference rule) | own flanks |
| `black_gardner_flank_resid` | fixed Gardner (0.31·Vp^0.25) from DTC, plus the RHOB − Gardner residual on the flanks interpolated linearly across the gap | DTC + RHOB flanks |

**Units.** RHOB in g/cm³; depth in metres; DTC in µs/ft (the Gardner row converts to Vp).

**What each prediction contains.** A full-depth array, NaN outside `Task.predict_mask`. Granite rows also return
P05, P25, P75 and P95. These are empirical bands, calibrated on held-out D wells: P50 ± k·s, where s is the local
TTM P90 − P10 (with a 0.005 floor), or 1 for TSPulse. The other rows return no interval.

**Status of each hidden piece.** Recorded in `predict_with_pieces`:
- `two_sided`: both flanks used.
- `one_sided_above` / `one_sided_below`: only one flank long enough.
- `fallback_interp`: the straight line is used. This happens when the model cannot cover the gap: TTM gap
  > 336 samples, or both TTM flanks < 256 contiguous samples; TSPulse gap ≥ 512 samples, or < 16 observed samples
  per side.
- `abstain`: NaN. There is no measured target on a side, or the curve is missing entirely.

**Assumptions and limits.**
- Target-only: no other log, calendar or frequency token reaches the Granite models.
- Gaps must sit inside one acquisition segment.
- The tuned heads and the D calibration are FORCE-specific. They are refused, with `StaleArtifacts`, when the
  code, configuration, split, data or gaps no longer match the frozen manifest (`manifest.json`; `manifest_ft.json`
  for the two ft rows, which also checks the checkpoint SHA-256). On non-FORCE wells, such as the
  synthetic contract check, the rows instead fit a small local calibration and train for 5 head steps.
- Whole-curve (`missing_curve`) tasks have no RHOB context, so every row here abstains on them.

**Fit and runtime.** The rows need the Granite environment (torch, granite-tsfm at commit `31fa2f0`) and the
cached weights. On a laptop Quadro T1000, fit (load) takes 2–10 s and prediction on 20 gaps under 0.1 s.

**In the viewer.** `python -m benchmark.run --model <ids> --test rhob_gap50` writes `scores.json` and `predictions.json` under `results/`; the viewer's scoreboard and prediction tracks read them from any environment. Live prediction in the viewer runs the model in the viewer's own process, so launch it from the Granite environment with the opt-in: `WMB_BLACK_GRANITE=1 <granite-python> -m wmb.viewer --launch`.

**Licences:** [LICENSE.md](LICENSE.md). Apache-2.0 covers the IBM weights and code; MIT covers these wrappers.

```
WMB_BLACK_GRANITE=1 python -m benchmark.check_model black_ttm_r3_tuned
WMB_BLACK_GRANITE=1 python -m benchmark.run --model black_ttm_r3_tuned,black_linear_gap --test rhob_gap50
```
