# NACA 0012 airfoil at Re = 10,000 (α = 5°)

Steady flow over a NACA 0012 airfoil with α = 5°, at Re = 10,000. The tenfold increase in Reynolds number relative to the Re = 1,000 case generates a more complex wake pattern, steeper pressure gradients near the airfoil surface, and noticeably thinner boundary layers (δ ~ Re⁻¹ᐟ²). The flow remains attached with structured wake and a pressure-gradient-driven suction peak.

## Files
- `sphinx_naca_re10k.py` — main SPHINX training script (identical architecture and hyperparameters to the Re = 1,000 case)

## Reference data
- `../../data/naca0012_re10k/naca0012_re10k_reference.npz` — SU2 steady RANS reference solution with Spalart-Allmaras turbulence model

## Running

```bash
# Operating point: N = 200, f = 0.1
python sphinx_naca_re10k.py --n_sparse 200 --interface-frac 0.1 --seed 42

# XPINN baseline (no sparse data)
python sphinx_naca_re10k.py --n_sparse 0 --seed 42
```

## Headline result at the operating point

| Variable | Baseline (N=0) | SPHINX (N=200, f=0.1) | Reduction |
|---|---|---|---|
| u | 24.12 ± 0.28 % | 1.908 ± 0.027 % | 92.1% |
| v | 58.36 ± 2.12 % | 14.024 ± 0.355 % | 76.0% |
| p | 196.48 ± 74.41 % | 22.246 ± 0.483 % | 88.7% |

Three independent trials, seeds 42, 123, 456. The relative reduction is comparable to the Re = 1,000 case, demonstrating Reynolds-number-independent improvement. Absolute residual errors are higher than at Re = 1,000 due to spectral-bias limits of the 8×128 architecture at higher Reynolds number.
