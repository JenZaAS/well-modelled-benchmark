# Data licences and credit lines (summary; check each dataset README before publishing)

| Dataset | Licence / terms | Credit line | Status |
|---|---|---|---|
| FORCE 2020 (Zenodo 4351156) | Logs NLOD 2.0; lithofacies labels CC BY 4.0 (Zenodo record states CC BY 4.0) | "Lithofacies data was provided by the FORCE Machine Learning competition with well logs and seismic 2020." Bormann P., Aursand P., Dilib F., Dischington P., Manral S. (2020), doi.org/10.5281/zenodo.4351156 | **To verify** (logs NLOD 2.0 vs CC BY 4.0) |
| Sodir FactPages | NLOD 2.0 | "Contains data under the NLOD distributed by Sodir" | OK |
| Volve (Equinor) | Equinor Volve data licence (open, with attribution; not CC) | Equinor and the Volve licence partners | **To verify** exact wording |
| NLOG (Netherlands) | NLOG disclaimer, not CC: copy/distribute/edit allowed if NLOG.NL is quoted as the source | "Source: NLOG.NL" | OK, quote source |
| Kansas Geological Survey | KGS terms: free use and distribution with the exact KGS citation | "The source of this material is the Kansas Geological Survey website at http://www.kgs.ku.edu/. All Rights Reserved." | Not CC; never put raw LAS in this repo |
| Utah FORGE (GDR) | CC BY 4.0 | Utah FORGE / EGI, University of Utah, via GDR | OK |
| GLADE (GDR 1857) | CC BY 4.0 | Rhodes, R. (2026), GLADE Project well log files, GDR, OXY USA Inc. | OK |
| USGS Gulf Coast CWLDB | CC0 1.0 / US public domain | Buursink et al. 2023 (ver. 4.0, 2025), doi.org/10.5066/P95BNWT3 | OK |
| Diskos | Licensed | — | Only if access is given; raw data stays local |

The repo itself is MIT and holds **code only**. Data is shared for testing, not redistribution.
One exception: the viewer prediction files for FORCE 2020 runs in `results/` contain test-well curves and are
committed with the FORCE 2020 credit line in [results/README.md](../results/README.md). Both FORCE 2020
licences (NLOD 2.0 and CC BY 4.0) allow redistribution with credit. No other dataset's curves are committed;
the allowlist in `.gitignore` enforces this.
