# Castagna mudrock line (`black_castagna_dts`)

**DTS ← DTC**, pointwise, for whole missing curves and gaps. Fixed published coefficients; no fitting.

## Equation and units

Vp = 304.8 / DTC (km/s, DTC in µs/ft) · **Vs = 0.8621 Vp − 1.1724** (km/s) · DTS = 304.8 / Vs (µs/ft).
The 1985 mudrock line is an empirical trend for water-saturated clastic silicate rocks.

Example: DTC = 100 µs/ft → Vp = 3.048 km/s → Vs = 1.45528 km/s → **DTS = 209.444 µs/ft**.

## Domain, abstention and extrapolation

- Requires finite DTC > 0.
- **Numerical guard (kept from the original code): abstain where Vs ≤ 0.3 km/s.** The line reaches
  Vs = 0.3 km/s at DTC ≈ 178.46 µs/ft and Vs = 0 at DTC ≈ 224.1 µs/ft. Without the guard, slower rocks
  would get DTS above 1016 µs/ft and then negative slowness. So the model abstains for DTC ≥ ~178.5 µs/ft.
- Carbonates, gas sands and anisotropic shales depart from the line; this is a trend, not a bound.

## Uncertainty

Training-well residuals on rows with finite, positive measured DTS **and** a finite mudrock prediction.
σ = √(mean over wells of each well's mean squared residual). This fixes the earlier implementation, whose
plain standard deviation over all finite pairs returned σ = NaN on the full FORCE training set because two
pairs fall beyond the guard. Fewer than two contributing wells → `std=None`.

## Run it

```python
from workstreams.black_physics.contender import MudrockDTS
MudrockDTS._dts(100.0)             # 209.444, kept for existing callers
```

```
python -m benchmark.check_model black_castagna_dts
python -m benchmark.run --model black_castagna_dts --dataset force2020 --target DTS --scenario all
```

## References

- Castagna, J.P., Batzle, M.L. & Eastwood, R.L. (1985). Relationships between compressional-wave and
  shear-wave velocities in clastic silicate rocks. *Geophysics* 50(4), 571–581. https://doi.org/10.1190/1.1441933

Licence: [MIT](LICENSE.md), original code.
