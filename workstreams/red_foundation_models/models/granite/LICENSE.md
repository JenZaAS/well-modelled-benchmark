# Licences: Granite gap-reconstruction rows

| Row | Code | Weights |
| --- | --- | --- |
| `black_ttm_r3_frozen`, `black_ttm_r3_tuned` | this wrapper: MIT; [granite-tsfm](https://github.com/ibm-granite/granite-tsfm/blob/31fa2f01573341f17e9997280f61481de18ce28c/LICENSE) (commit `31fa2f0`): Apache-2.0 | [ibm-granite/granite-timeseries-ttm-r3](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3/tree/226cdc78f96e1304760ca141bd82d7d91be08bd8) revision `226cdc7` (branch `512-336-dec-512-lite-r3`): Apache-2.0 |
| `black_tspulse_frozen`, `black_tspulse_tuned` | as above | [ibm-granite/granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/tree/b12164578f7b893ada0028c00d292ba10383d25a) revision `b121645` (branch `tspulse-hybrid-dualhead-512-p8-r1`): Apache-2.0 |
| `black_linear_gap`, `black_gardner_flank_resid` | original, MIT | none |

**Licence:** Apache-2.0 (<https://www.apache.org/licenses/LICENSE-2.0>) for the IBM weights and granite-tsfm; MIT for
the wrappers in this folder.

**Restrictions:** none beyond Apache-2.0 attribution and notice. The tuned heads are derivatives of the Apache-2.0
weights trained locally on FORCE 2020 training wells; they stay in the local work folder and are not redistributed.
The decision comparator Decima-small (`amyrmahdy/decima-small@480c4ea`, Apache-2.0) discloses non-commercial
training sets (ANLI, XNLI, CC BY-NC 4.0); review its card before any commercial use. Verify that each licence fits
your use before running or sharing a model ([THIRD_PARTY.md](../../../../THIRD_PARTY.md)).
