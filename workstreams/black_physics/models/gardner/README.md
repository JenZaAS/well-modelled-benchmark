# Gardner density from compressional slowness

Two registered variants share [`equation.py`](equation.py). Both predict **RHOB ← DTC** for whole missing
curves and gaps, pointwise.

| ID | Coefficients | Uncertainty |
| --- | --- | --- |
| `black_gardner` | `a`, `b` refitted on the training wells, published fallback | out-of-fold residuals |
| `black_gardner_fixed` | published `a = 0.31`, `b = 0.25` in every basin | training-well residuals |

## Equation and units

ρ = a · Vp^b, with ρ in g/cm³ and **Vp = 304800 / DTC in m/s** (DTC in µs/ft). Using km/s instead of m/s
changes the result by a factor of 1000^b (≈ 5.6 for b = 0.25). This is the repository's published
coefficient convention, credited to Gardner, Gardner & Gregory (1974).

Example: DTC = 100 µs/ft → Vp = 3048 m/s → ρ = 0.31 × 3048^0.25 = **2.30338 g/cm³**.

## Fit (`black_gardner` only)

Least squares of ln ρ on ln Vp over training pairs with finite DTC strictly inside **40 < DTC < 200 µs/ft**
and finite ρ > 0. Requirements: more than 10 such rows, more than one distinct DTC value, and a finite fit
with b > 0 (density increasing with velocity). Otherwise the published coefficients are used and the
reason is recorded in `params.note`. The 40–200 µs/ft range only selects fitting rows; it does not limit
prediction.

## Domain, abstention and extrapolation

- Predicts wherever DTC is finite and > 0; NaN elsewhere (missing, zero, negative, ±∞) and where the
  result is not finite. Neighbouring rows keep their predictions.
- DTC outside 40–200 µs/ft is an extrapolation of the fitted (or published) power law, not an abstention.
- Gardner is an average trend for brine-saturated sedimentary rocks. Expect bias in gas sands, evaporites,
  coals, carbonates and unconsolidated shallow sections. FORCE plausibility flags are not used.

## Uncertainty

One Gaussian σ in g/cm³: the square root of the mean over wells of each well's mean squared residual
(equal well weight), on rows with finite, positive measured RHOB and a finite prediction. The fitted
variant uses deterministic out-of-fold predictions (up to five whole-well folds, seed 2026) and then
refits on all training wells. Fewer than two contributing wells → point predictions with `std=None`.
`std` is NaN wherever the mean is NaN.

## Run it

```python
from workstreams.black_physics.models.gardner.equation import gardner_density, fit_gardner
gardner_density(100.0)            # 2.30338
```

```
python -m benchmark.check_model black_gardner_fixed
python -m benchmark.run --model black_gardner,black_gardner_fixed --dataset force2020 --target RHOB --scenario all
```

## References

- Gardner, G.H.F., Gardner, L.W. & Gregory, A.R. (1974). Formation velocity and density—the diagnostic
  basics for stratigraphic traps. *Geophysics* 39(6), 770–780. https://doi.org/10.1190/1.1440465

Licence: [MIT](LICENSE.md), original code.
