# NLOG Netherlands well logs

Public Dutch well logs used as an optional, less uniform test set for missing-log
and DTS prediction. **Data is not in git.** Align a local copy first
(see [docs/data-alignment.md](../../docs/data-alignment.md)).

- **Source:** [NLOG](https://www.nlog.nl/en)
- **Licence:** [NLOG disclaimer](https://www.nlog.nl/en/disclaimer) (not a Creative Commons licence). The Dutch text at https://www.nlog.nl/disclaimer prevails if the translation differs. Users may copy, download, disclose, distribute, process or edit the information, **provided NLOG.NL is quoted as the source**. Content is provided “as is”.
- **Required credit:** Source: NLOG.NL
- **Further citation (selection of shear-sonic wells):** Hunfeld, L.B., Foeken, J.P.T. and van Kempen, B.M.M. (20 December 2021), *Geomechanical parameters derived from compressional and shear sonic logs for main geothermal targets in The Netherlands*, https://www.nlog.nl/sites/default/files/2021-12/data_selection_and_methods.pdf.

## Place and align

1. Keep the raw public LAS files in a local folder with one subfolder per well, plus the source README and `wells.json` if you have them. Do not copy them into this repository.
2. Write an aligned copy (standard names, metres MD, one LAS per well):

```bash
python -m wmb.align --input /path/to/raw/NLOG --output /path/to/aligned/NLOG --dataset nlog_dts
```

3. In `data_paths.local.yaml` (gitignored) set `nlog_dts: "/path/to/aligned/NLOG"`, or put the aligned folder under `data_root` as `NLOG/` or `NLOG-DTS/`.
4. Run `python -m wmb.paths --check`. Load with `wmb.io.iter_manifest_wells(...)`.

There is no single official zip or checksum file for this collection (files come per borehole from the public site), so this pack has no `fetch.py` / `checksums.sha256`. Verify against the source site and keep the credit line.

Optional test: `python -m benchmark.run --model template_linear --test nlog_dts_missing` (skipped if the aligned folder is not present).

## Quirks

See `mapping.yaml`. Depth units are mixed; curve names vary a lot; many wells are deviated. After alignment, tests still use measured depth.
