# Turek-Hron FSI1 benchmark (Re = 20)

Steady incompressible flow at Re = 20 around a cylinder with an attached elastic beam in a channel. Channel: 2.5 m × 0.41 m. Cylinder diameter D = 0.1 m centered at (0.2, 0.2). Elastic beam: length 0.35 m, thickness 0.02 m, extending downstream of the cylinder.

Due to the low Reynolds number, the beam undergoes negligible deformation (~0.8 μm tip displacement) and the flow reaches a steady state. SPHINX is applied in a **fluid-only formulation**: the beam deformation is prescribed from a validated turtleFSI reference solution, and the XPINN reconstructs only the fluid fields (u, v, p).

## Files
- `sphinx_fsi1.py` — main SPHINX training script

## Reference data
- `../../data/fsi1_re20/turek_hron_fsi1_reference.npz` — turtleFSI reference solution. Contains `coordinates`, `u`, `v`, `p`, `topology`, `domain_markers`, `displacement` for the deformed beam geometry.

## Running

```bash
# Operating point: N = 200, f = 0.1
python sphinx_fsi1.py --n_sparse 200 --interface-frac 0.1 --seed 42

# XPINN baseline (no sparse data)
python sphinx_fsi1.py --n_sparse 0 --seed 42
```

## Headline result at the operating point

| Variable | Baseline (N=0) | SPHINX (N=200, f=0.1) | Reduction |
|---|---|---|---|
| u | 49.86 ± 7.72 % | 5.252 ± 0.020 % | 89.5% |
| v | 44.75 ± 3.45 % | 15.904 ± 0.101 % | 64.5% |
| p | 94.22 ± 19.99 % | 2.647 ± 0.025 % | 97.2% |

Three independent trials, seeds 42, 123, 456. The pressure reduction (97.2%) is the largest of all four benchmarks, demonstrating that the channel geometry and parabolic inlet provide strong constraints once sparse data anchoring is applied.

## Boundary conditions
- Parabolic inlet profile at x = 0
- No-slip on channel walls (top and bottom)
- No-slip on cylinder surface
- No-slip on the deformed beam surface, using prescribed beam velocity v_beam = ∂d/∂t
- Zero-traction outflow at x = 2.5
