# FORCE 2020 well logs and lithofacies

Official demo dataset for this repository: 118 wells from the FORCE 2020
machine learning competition (lithology from well logs), Norwegian Continental
Shelf. Curve names and units are already uniform, so no changes to the
downloaded files are needed.

This folder holds the instructions, the column mapping, checksums, and a
fetch script. **It does not hold the data.** Data stays outside git, in a
`FORCE2020/` subfolder of your local data folder.

## Licence and credit

The [Zenodo record](https://doi.org/10.5281/zenodo.4351156) is tagged
**CC BY 4.0**. The same record says the original well-log data come from the
Norwegian government under **NLOD 2.0** (written "NOLD 2.0" on the record).
The lithofacies labels were hand-crafted for the competition (Explocrowd) and
are treated as **CC BY 4.0**. That NLOD-versus-CC-BY split is not fully
resolved on the record itself; keep both licences in mind and always give
credit.

Credit line:

> Bormann, P., Aursand, P., Dilib, F., Manral, S. and Dischington, P. (2020).
> FORCE 2020 Well well log and lithofacies dataset for machine learning
> competition. Zenodo. Also credit FORCE and the Norwegian Offshore
> Directorate (Sodir).

Please also cite the DOI and the official GitHub repository, as the Zenodo
record asks:

- Bormann, P., Aursand, P., Dilib, F., Manral, S., & Dischington, P. (2020).
  FORCE 2020 Well well log and lithofacies dataset for machine learning
  competition [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4351156
- https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition

## 1. Download

You need files from **both** official sources. They are public; you do not
need a shared team folder.

### From GitHub (CSV splits and penalty matrix)

Repository: [bolgebrygg/Force-2020-Machine-Learning-competition](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)

Folder: [`lithology_competition/data/`](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/tree/master/lithology_competition/data)

| File | What it is |
|---|---|
| `train.zip` | 98 training wells. **Unzip this** to get `train.csv` (`train.csv` is not published as a loose file on GitHub). |
| `leaderboard_test_features.csv` | 10 official leaderboard wells, logs only |
| `leaderboard_test_target.csv` | Lithology labels for those leaderboard wells |
| `hidden_test.csv` | 10 official hidden / blind-test wells, logs and labels |
| `penalty_matrix.npy` | Official lithology scoring matrix (recommended) |

Direct files (same names and sizes as above):

- https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/train.zip
- https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/leaderboard_test_features.csv
- https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/leaderboard_test_target.csv
- https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/hidden_test.csv
- https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/penalty_matrix.npy

### From Zenodo 4351156 (LAS files and NPD tables)

Record: https://zenodo.org/records/4351156  
DOI: https://doi.org/10.5281/zenodo.4351156

| File | What it is |
|---|---|
| `LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip` | 118 LAS files (optional for the CSV loader) |
| `lithology scoring matrix cost function.xlsx` | Scoring matrix as a spreadsheet |
| `NPD_Casing_depth_most_wells.xlsx` | Casing depths |
| `NPD_Lithostratigraphy_groups_all_wells.xlsx` | Lithostratigraphic groups |
| `NPD_Lithostratigraphy_member_formations_all_wells.xlsx` | Formations / members |

Direct downloads follow `https://zenodo.org/records/4351156/files/<filename>?download=1`.

The required files for the loader and for `python -m wmb.datasets --check force2020` are the four CSVs (`train.csv` after unzipping, the two leaderboard files, and `hidden_test.csv`). The rest are recommended or optional.

## 2. Place the files

Put every downloaded file in a folder named **`FORCE2020`** under your local
data folder (the `data_root` in `data_paths.local.yaml`):

```
<data_root>/FORCE2020/train.csv
<data_root>/FORCE2020/train.zip
<data_root>/FORCE2020/leaderboard_test_features.csv
<data_root>/FORCE2020/leaderboard_test_target.csv
<data_root>/FORCE2020/hidden_test.csv
<data_root>/FORCE2020/penalty_matrix.npy
...
```

Unzip `train.zip` in that folder if `train.csv` is not already there. Do not
commit these files to git.

To create `data_paths.local.yaml` and walk through this setup, run:

```bash
python -m wmb.onboarding
```

Or follow [docs/SETUP_DATA_PATHS.md](../../docs/SETUP_DATA_PATHS.md). Onboarding
is safe to run again later (see below).

You can also fetch the official files into that folder (checksums are checked
after download):

```bash
python datasets/force2020/fetch.py --dest /path/to/data_root/FORCE2020
```

If `data_paths.local.yaml` is already set, `--dest` can be omitted and the
script uses `FORCE2020` under your data root.

## 3. Verify

```bash
python -m wmb.datasets --check force2020
python -m wmb.onboarding --status
```

The check compares files to [checksums.sha256](checksums.sha256), reports any
mismatch without changing the files, and prints how many wells have each
canonical curve. A missing optional file is a warning; a missing required
file or a checksum mismatch is a failure.

## How to load it in code

```python
from wmb.datasets import load_force2020

wells = load_force2020()          # uses data_paths.local.yaml
# or: load_force2020(root="/path/to/FORCE2020")
# or: load_force2020(splits=["train"])  # also: leaderboard, hidden
```

Each well has depth in metres MD and canonical curve names (`GR`, `RDEP`,
`RMED`, `CALI`, `RHOB`, `NPHI`, `PEF`, `DTC`, `DTS`, plus `LITH` and
`LITH_CONF` when labels are present). Missing samples are NaN. The contest
CSVs are semicolon-separated; the loader handles that.

Column mapping, units, splits and the lithology code table are in
[mapping.yaml](mapping.yaml). Shared standard names and guardrails are in
`src/wmb/datasets/canonical_curves.yaml`.

## Curve availability

118 wells (98 train, 10 leaderboard, 10 hidden). A well "has" a curve if it
has at least one valid sample. These counts are from the official CSVs.

| Curve | Unit | Wells | Notes |
|---|---|---:|---|
| GR | gAPI | 118 / 118 | |
| RDEP | ohm.m | 118 / 118 | |
| RMED | ohm.m | 117 / 118 | No valid samples in train well 34/10-33 |
| CALI | in | 117 / 118 | No valid samples in train well 35/8-4 |
| RHOB | g/cm3 | 118 / 118 | |
| NPHI | v/v (m3/m3) | 118 / 118 | Fraction, not percent |
| PEF | b/e | 88 / 118 | 69 train, 10 leaderboard, 9 hidden |
| DTC | us/ft | 118 / 118 | |
| DTS | us/ft | 46 / 118 | 32 train, 6 leaderboard, 8 hidden |

## Known quirks

- Names and units are already the canonical ones. No aliasing or unit
  conversion is required.
- PEF has values well above the 20 b/e guardrail (about 23,760 samples in
  49 wells, maximum about 383). The loader should flag these; do not edit
  the files.
- A smaller number of spikes fall outside the other guardrails (DTS above
  500 us/ft in 6 wells, plus a few DTC, NPHI, GR, CALI, RHOB and RMED
  samples). Flag, do not edit.
- Soft North Sea shales can have DTS medians around 290–370 us/ft. That is
  still us/ft, not a unit error.
- The logs were slightly cleaned and partially despiked by the organisers.
- `wells.json` is not part of the official download.

## Onboarding (re-runnable)

```bash
python -m wmb.onboarding            # first run, or continue where you left off
python -m wmb.onboarding --status   # what is chosen, declined, verified
python -m wmb.onboarding --redo     # repeat steps (optionally name a step or dataset)
```

Progress is stored in gitignored `.wmb/onboarding_state.json`. If this
repository later adds a dataset or an onboarding step, the onboarding
version is bumped and a rerun will show what is new, skip work you already
finished, and let you change earlier yes/no choices. Chosen datasets are
re-verified.
