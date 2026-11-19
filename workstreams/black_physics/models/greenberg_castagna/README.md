# Greenberg–Castagna brine mixture

[`equation.py`](equation.py) holds the four-lithology, brine-saturated equation. One registered adapter uses
it on ordinary benchmark inputs:

| ID | Target ← inputs | Assumption made visible in the ID |
| --- | --- | --- |
| `black_gc_brine_gr` | **DTS ← DTC, GR** | every depth is a **brine**-filled sand/shale mix, shale fraction from **GR** |

## Equation

For each active lithology i (fraction fᵢ > 0, fractions summing to 1), with Vp and sᵢ in km/s:

sᵢ = a₂Vp² + a₁Vp + a₀  ·  **Vs = [Σ fᵢsᵢ + 1 / Σ(fᵢ/sᵢ)] / 2**  ·  DTS = 304.8 / Vs (µs/ft)

| Lithology | a₂ | a₁ | a₀ |
| --- | ---: | ---: | ---: |
| sandstone | 0 | 0.80416 | −0.85588 |
| limestone | −0.05508 | 1.01677 | −1.03049 |
| dolomite | 0 | 0.58321 | −0.07775 |
| shale | 0 | 0.76969 | −0.86735 |

Example: Vp = 4 km/s → s_sand = 2.36076, s_shale = 2.21141 km/s; a 50/50 mix gives **Vs = 2.284865 km/s,
DTS = 133.3995 µs/ft**.

`gc_brine_vs(vp_km_s, {"sandstone": f1, "limestone": f2, ...})` accepts any subset of the four lithologies as
scalars or arrays (measured mineral fractions, for a future adapter). A row is NaN unless Vp is finite and
> 0, every fraction is finite and ≥ 0, the fractions sum to 1 within 1e-9, and every **active** endmember
speed is > 0. Zero-weight endmembers are skipped, so a negative sandstone speed does not mask a pure
dolomite row. Vs always lies between the smallest and largest active endmember speed.

## GR adapter (`black_gc_brine_gr`)

- **Endpoints.** Per training well, the 10th and 90th percentiles of finite GR; the median of each across
  wells gives `gr_low` and `gr_high`. GR needs no target, so every training well with GR contributes,
  including wells without DTS. No lithology labels, formation names or test-well statistics are used.
- **Fractions.** f_shale = clip((GR − gr_low)/(gr_high − gr_low), 0, 1), f_sand = 1 − f_shale.
- **New basin.** `GreenbergCastagnaBrineGR(gr_low=…, gr_high=…, endpoint_source="…")` fixes the endpoints
  and records where they came from; it is then a fixed equation calibrated on training residuals.
  Register such a configuration under its own ID.
- **Abstains** where DTC or GR is missing or invalid, where endpoints are missing or `gr_high ≤ gr_low`,
  and where an active endmember speed is ≤ 0 (DTC ≳ 270 µs/ft for shale, ≳ 286 µs/ft for sand).
  There is no extra velocity floor: very slow, unconsolidated rocks (DTC above ~180 µs/ft) are an
  extrapolation of the brine trends and can give very large DTS.
- **Known failure cases.** Carbonates (GR reads "sand"), radioactive sands (GR reads "shale"), gas- or
  oil-bearing intervals (the trends are brine-saturated), coals and evaporites.

## Uncertainty

Out-of-fold: up to five whole-well folds over training wells with measured DTS (seed 2026); endpoints for
each fold come from every other training well. σ = √(mean over wells of each well's mean squared residual).
The endpoints are then refitted on all training wells. Fewer than two contributing wells → `std=None`.

## Not implemented: full method with independent inputs

Hydrocarbon-bearing applications need porosity, saturation, mineral and fluid properties and Gassmann
fluid substitution (brine-equivalent Vp, then back to in-situ Vs at constant shear modulus). That belongs
under a separate model ID once a dataset supplies those inputs; the FORCE tasks support only this
GR/brine approximation.

## Run it

```
python -m benchmark.check_model black_gc_brine_gr
python -m benchmark.run --model black_gc_brine_gr,black_castagna_dts,black_vpvs_ratio --dataset force2020 --target DTS --scenario all
```

## References

- Greenberg, M.L. & Castagna, J.P. (1992). Shear-wave velocity estimation in porous rocks: theoretical
  formulation, preliminary verification and applications. *Geophysical Prospecting* 40(2), 195–209.
  https://doi.org/10.1111/j.1365-2478.1992.tb00371.x
- Coefficient set and mixture equation as published in a Norne field study (equation 1, table 1):
  https://www.sciencedirect.com/science/article/abs/pii/S0920410518307708
- Gamma-ray shale indicator: Kansas Geological Survey log-analysis notes,
  https://www.kgs.ku.edu/Publications/Bulletins/LA/03_gamma.html
- Gassmann fluid substitution (future work): Smith, Sondergeld & Rai (2003), https://doi.org/10.1190/1.1567211

Licence: [MIT](LICENSE.md), original code.
