# Dataset pack template

Copy this folder to `datasets/<id>/` (for example `datasets/nlog_dts/`) and
replace every placeholder. Keep the same four files so onboarding and the
check command can find them.

## Files in a dataset pack

| File | Purpose |
|---|---|
| `README.md` | This page: what it is, licence and credit, how to download, where to place the files, how to verify, how to load, curve availability, quirks |
| `mapping.yaml` | Files expected, columns, units, splits, credit line, known quirks |
| `checksums.sha256` | SHA-256 of each official file (`hash` then two spaces then the filename) |
| `fetch.py` | Optional. Download official files into the data subfolder and verify checksums. Never write data into the git repo |

## Checklist

1. Pick a stable `id` (lowercase, no spaces, for example `nlog_dts`) and the
   exact data subfolder name under `data_root` (for example `NLOG-DTS`).
   Register the pair in `src/wmb/paths.py` if it is not already there.
2. Fill in `mapping.yaml` and `checksums.sha256`. Do not invent checksums;
   hash the official files.
3. Write the README: download URLs (verified), exact file names, licence and
   a verbatim credit line, place, verify, load, availability, quirks.
4. Point the loader and `python -m wmb.datasets --check <id>` at this folder.
5. Add a row to [../README.md](../README.md). If onboarding grows a new
   dataset or step, bump `ONBOARDING_VERSION` and add a changelog entry so a
   rerun can tell the user what is new.
6. Never commit data, and never put private folder paths, company names or
   personal names in these files.

Data is not ready until the README, mapping and checksums are filled in and
the check command passes against a clean official copy.
