# NACA 0012 airfoil at Re = 1,000 (α = 5°)

Steady incompressible flow over a NACA 0012 airfoil with chord length c, angle of attack α = 5°, at Re = U∞c/ν = 1,000. The flow features a thin attached boundary layer on the suction side, mild trailing-edge separation, and an asymmetric wake driven by the angle of attack.

## Files
- `sphinx_naca_re1k.py` — main SPHINX training script
- Boundary conditions: no-slip on the airfoil surface (cosine-spaced control points), freestream at far-field

## Reference data
- `../../data/naca0012_re1k/naca0012_re1k_reference.npz` — FEniCS reference on a refined mesh

## Running

```bash
# Operating point: N = 200, f = 0.1
python sphinx_naca_re1k.py --n_sparse 200 --interface-frac 0.1 --seed 42

# XPINN baseline (no sparse data)
python sphinx_naca_re1k.py --n_sparse 0 --seed 42
```

## Headline result at the operating point

| Variable | Baseline (N=0) | SPHINX (N=200, f=0.1) | Reduction |
|---|---|---|---|
| u | 24.40 ± 0.96 % | 1.132 ± 0.012 % | 95.4% |
| v | 59.66 ± 9.78 % | 8.528 ± 0.153 % | 85.7% |
| p | 97.89 ± 4.66 % | 3.998 ± 0.063 % | 95.9% |

Three independent trials, seeds 42, 123, 456. This case shows the largest u-velocity reduction across all four benchmarks.
