# Volumetric density mixing (`black_density_nphi`)

**RHOB ← NPHI**, pointwise, for whole missing curves and gaps. Fixed reference configuration; no fitting.

## Equation and units

For solid-normalised mineral fractions fᵢ (summing to 1): **ρ = (1 − φ) Σ fᵢρᵢ + φ ρ_fluid**, densities in
g/cm³, φ in v/v.

Reference configuration: **quartz 2.65 g/cm³, water 1.00 g/cm³**, and the **measured neutron log NPHI as
porosity**. Example: φ = 0.20 → 0.8 × 2.65 + 0.2 × 1.00 = **2.32 g/cm³**.

`mixed_density(phi, {"quartz": 0.6, "calcite": 0.4}, {"quartz": 2.65, "calcite": 2.71})` gives the
multi-mineral form for user-supplied, independently measured fractions and densities; rows whose fractions
are not finite, ≥ 0 and summing to 1 are NaN.

## Assumptions you should know

- NPHI is an apparent porosity: shale-bound hydrogen makes it read high (density predicted too low), gas
  makes it read low (density too high), and its matrix reference may not be quartz.
- Pass an independent porosity and endmembers when you have them, and register that configuration under a
  **separate ID**. Never use density porosity derived from the hidden RHOB: that leaks the answer.

## Domain and abstention

Predicts where NPHI is finite and within **[0, 1] v/v**; other rows (including genuine negative apparent
readings) are unsupported and left NaN. The source log is not modified.

## Uncertainty

Training-well residuals (fixed equation), equal well weight, on rows with finite, positive measured RHOB and
a finite prediction. Fewer than two contributing wells → `std=None`.

## Run it

```python
from workstreams.black_physics.models.volumetric_density.equation import bulk_density
bulk_density(0.20)                # 2.32
```

```
python -m benchmark.check_model black_density_nphi
python -m benchmark.run --model black_density_nphi,black_gardner --dataset force2020 --target RHOB --scenario all
```

## References

- Density and neutron log examples: Kansas Geological Survey log-analysis notes,
  https://www.kgs.ku.edu/Publications/Bulletins/LA/04_nuclear.html

Licence: [MIT](LICENSE.md), original code.
