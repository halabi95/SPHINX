# Cylinder Re = 40

Steady incompressible flow around a circular cylinder of diameter D at Re = 40. The flow develops a steady symmetric recirculation zone behind the cylinder. Reference Cd ≈ 1.561.

## Files
- `sphinx_cylinder.py` — main SPHINX training script

## Reference data
- `../../data/cylinder_re40/cylinder_re40_reference.npz` — FEniCS reference solution on ~50,000 triangular elements

## Running

```bash
# Operating point: N = 200, f = 0.1
python sphinx_cylinder.py --n_sparse 200 --interface-frac 0.1 --seed 42

# XPINN baseline (no sparse data)
python sphinx_cylinder.py --n_sparse 0 --seed 42
```

## Headline result at the operating point

| Variable | Baseline (N=0) | SPHINX (N=200, f=0.1) | Reduction |
|---|---|---|---|
| u | 9.95 ± 0.30 % | 1.421 ± 0.006 % | 85.7% |
| v | 29.75 ± 0.22 % | 16.280 ± 0.045 % | 45.3% |
| p | 58.04 ± 11.20 % | 5.944 ± 0.014 % | 89.8% |

Three independent trials, seeds 42, 123, 456.
