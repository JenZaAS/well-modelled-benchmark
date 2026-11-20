# Kansas Geological Survey (DTS subset)

Public Kansas well logs used as an optional test set. This pack is the small
subset of wells that have a shear-sonic (DTS) curve. **Data is not in git.**
Align a local copy first (see [docs/data-alignment.md](../../docs/data-alignment.md)).

- **Source:** [Kansas Geological Survey archives of LAS files](https://www.kgs.ku.edu/PRS/Scans/Log_Summary/index.html)
- **Licence / terms:** [KGS terms of use](https://www.kgs.ku.edu/General/copyright.html). Not a Creative Commons licence. Redistribution is allowed without a fee only with the exact citation below, including the words **All Rights Reserved**. Keep the copyright notice intact. Do not use the KGS name in advertising without written permission. Data are provided as is, without warranty.
- **Required citation (verbatim):** The source of this material is the Kansas Geological Survey website at http://www.kgs.ku.edu/. All Rights Reserved.

## Place and align

1. Keep the raw public LAS files in a local folder with one subfolder per well. Do not copy them into this repository.
2. Write an aligned copy (standard names, metres MD, one LAS per well):

```bash
python -m wmb.align --input /path/to/raw/Kansas-KGS-DTS --output /path/to/aligned/Kansas-KGS --dataset kansas_kgs_dts
```

3. In `data_paths.local.yaml` (gitignored) set `kansas_kgs_dts: "/path/to/aligned/Kansas-KGS"`, or put the aligned folder under `data_root` as `Kansas-KGS/` or `Kansas-KGS-DTS/`.
4. Run `python -m wmb.paths --check`. Load with `wmb.io.iter_manifest_wells(...)`.

There is no single official zip for this DTS subset (it is screened from the public yearly archives), so this pack has no `fetch.py` / `checksums.sha256`. Keep the verbatim credit line.

Optional test: `python -m benchmark.run --model template_linear --test kansas_dts_missing` (skipped if the aligned folder is not present).

## Quirks

See `mapping.yaml`. Source depths are in feet. Resistivity and caliper aliases need the priority and value checks in the alignment guide.
