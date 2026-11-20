# Datasets

Each public dataset we use has a folder here. The folder holds instructions,
a mapping file, checksums, and any small fetch or prepare script. **It does
not hold the data.** Data lives outside git, in a subfolder of the path you
set in `data_paths.local.yaml` (see [docs/SETUP_DATA_PATHS.md](../docs/SETUP_DATA_PATHS.md)).

FORCE 2020 is the official demo dataset: a teammate can clone this repository,
download the public files, and get running without a shared data folder.

| Dataset | Folder | Data subfolder | Status |
|---|---|---|---|
| FORCE 2020 | [force2020/](force2020/) | `FORCE2020/` | Ready (official demo) |
| NLOG | [nlog_dts/](nlog_dts/) | `NLOG/` or `NLOG-DTS/` (aligned copy) | Alignment procedure ready; data stays local |
| Kansas KGS (DTS subset) | [kansas_kgs_dts/](kansas_kgs_dts/) | `Kansas-KGS/` or `Kansas-KGS-DTS/` (aligned copy) | Alignment procedure ready; data stays local |
| Volve | — | `Volve/` | Coming |
| Sodir FactPages | — | `Sodir-FactPages/` | Coming |

To add the next dataset, copy [\_template/](_template/) to a new folder and
follow the README there. To put a new public log set onto the FORCE names,
units and metres-MD standard, follow [docs/data-alignment.md](../docs/data-alignment.md)
and run `python -m wmb.align`.

Shared curve names, units and guardrails live in
`src/wmb/datasets/canonical_curves.yaml`. The loader and the check command
live in the `wmb` package:

```bash
python -m wmb.onboarding                 # set up data paths and choose datasets
python -m wmb.onboarding --status
python -m wmb.datasets --check force2020
```
