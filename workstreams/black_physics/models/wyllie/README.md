# Wyllie time average (`black_wyllie_nphi`)

**DTC ← NPHI**, pointwise, for whole missing curves and gaps. Fixed reference configuration; no fitting.

## Equation and units

**Δt = (1 − φ) Δt_matrix + φ Δt_fluid**, all slownesses in µs/ft, φ in v/v.

Reference configuration: **quartz matrix 55.5 µs/ft, water 189 µs/ft**, and the **measured neutron log NPHI
as an apparent-porosity proxy**. Example: φ = 0.20 → 0.8 × 55.5 + 0.2 × 189 = **82.2 µs/ft**.

## Assumptions you should know

- NPHI is an instrument response, not porosity. Its apparent porosity depends on the tool's reference
  lithology, shale-bound hydrogen and gas. In shale it reads high (predicts DTC too slow); in gas it reads
  low (predicts too fast).
- The time average is for consolidated, clean, water-bearing rocks. Unconsolidated sediments, gas and
  complex pore systems depart from it.
- For your own data, pass an independently established porosity and endmembers to
  `wyllie_dtc(phi, dt_matrix, dt_fluid)` and register that configuration under a **separate ID**.
  Never use sonic porosity derived from the hidden DTC: that leaks the answer.

## Domain and abstention

Predicts where NPHI is finite and within **[0, 1] v/v**; all other rows (including negative apparent
readings, which can be genuine in the source log) are counted as unsupported and left NaN. The input log
is not modified.

## Uncertainty

Training-well residuals (fixed equation) on rows with finite, positive measured DTC and a finite prediction;
σ = √(mean over wells of each well's mean squared residual). Fewer than two contributing wells →
`std=None`. A point prediction is available with zero training wells.

## Run it

```python
from workstreams.black_physics.models.wyllie.equation import wyllie_dtc
wyllie_dtc(0.20)                  # 82.2
```

```
python -m benchmark.check_model black_wyllie_nphi
python -m benchmark.run --model black_wyllie_nphi --dataset force2020 --target DTC --scenario all
```

## References

- Wyllie, M.R.J., Gregory, A.R. & Gardner, L.W. (1956). Elastic wave velocities in heterogeneous and porous
  media. *Geophysics* 21(1), 41–70. https://doi.org/10.1190/1.1438217
- Endmember values: Kansas Geological Survey Open-File Report 1979-9,
  https://www.kgs.ku.edu/Publications/OFR/1979/KGSOFR1979-9.pdf

Licence: [MIT](LICENSE.md), original code.
