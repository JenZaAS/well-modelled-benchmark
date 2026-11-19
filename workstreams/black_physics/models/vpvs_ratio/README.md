# Constant Vp/Vs ratio control (`black_vpvs_ratio`)

**DTS ← DTC**, pointwise, for whole missing curves and gaps. An empirical control: it measures how much the
mudrock intercept (`black_castagna_dts`) and the lithology mixture (`black_gc_brine_gr`) buy over one
fitted number.

## Equation and fit

**DTS = r × DTC**. The ratio DTS/DTC equals Vp/Vs and is dimensionless, so it is the same in µs/ft or µs/m.

r = median over training wells of each well's **median positive, finite DTS/DTC** (rows where both are
finite and > 0). Each well counts once regardless of its length. With no paired training well, r is
undefined and every prediction is NaN (unsupported).

## Domain and abstention

Predicts where DTC is finite and > 0 and r is defined; NaN elsewhere. No velocity floor is needed.

## Uncertainty

Out-of-fold: up to five whole-well folds over training wells with measured DTS (seed 2026); each fold's
ratio is fitted on the other training wells. σ = √(mean over wells of each well's mean squared residual).
r is then refitted on all training wells. Fewer than two contributing wells → `std=None`.

## Run it

```
python -m benchmark.check_model black_vpvs_ratio
python -m benchmark.run --model black_vpvs_ratio,black_castagna_dts --dataset force2020 --target DTS --scenario all
```

## References

- Pickett, G.R. (1963). Acoustic character logs and their applications in formation evaluation.
  *Journal of Petroleum Technology* 15(6), 659–667. https://doi.org/10.2118/452-PA

Licence: [MIT](LICENSE.md), original code.
