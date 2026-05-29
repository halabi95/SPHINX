# SPHINX: Sparse Physics-Hybrid Informed Neural eXtension

**Interface-aware sparse data placement for Extended Physics-Informed Neural Networks (XPINNs)**

SPHINX is a targeted data placement strategy that exploits the domain decomposition structure of XPINNs. Rather than distributing sparse measurement data uniformly, SPHINX allocates **10% of the data budget to subdomain overlap regions** and 90% randomly throughout the domain. Each interface data point simultaneously constrains both neighboring networks, providing a dual-anchoring effect that reinforces inter-subdomain consistency through the data rather than through soft-penalty compromise.

The 10% interface allocation (f = 0.1) is the consistent u-velocity optimum identified by a systematic sensitivity study sweeping f ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} across four benchmark cases. SPHINX requires no modifications to the XPINN architecture, loss function weights, or training algorithm: it is purely a data placement strategy.

## Key results

All results at the operating point N = 200 sparse data points, f = 0.1, three independent random seeds.

| Benchmark | Re | α | u-error reduction | p-error reduction |
|---|---|---|---|---|
| Cylinder | 40 | — | 85.7% | 89.8% |
| NACA 0012 | 1,000 | 5° | 95.4% | 95.9% |
| NACA 0012 | 10,000 | 5° | 92.1% | 88.7% |
| Turek-Hron FSI1 | 20 | — | 89.5% | 97.2% |

Across all four cases, SPHINX reduces u-velocity error by 86 to 95 percent using only 200 sparse points (0.4 percent of the reference mesh). Run-to-run variance collapses from baseline standard deviations of 0.22 to 74.41 percent down to below 0.04 percent at the operating point. The method adds approximately 8 percent computational overhead relative to the baseline XPINN and is architecture-agnostic.

## Repository structure

```
SPHINX/
├── README.md                          # this file
├── LICENSE                            # MIT
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── cases/                             # SPHINX training scripts (one per case)
│   ├── cylinder_re40/
│   │   ├── sphinx_cylinder.py
│   │   └── README.md
│   ├── naca0012_re1k/
│   │   ├── sphinx_naca_re1k.py
│   │   └── README.md
│   ├── naca0012_re10k/
│   │   ├── sphinx_naca_re10k.py
│   │   └── README.md
│   └── fsi1_re20/
│       ├── sphinx_fsi1.py
│       └── README.md
│
├── data/                              # reference CFD solutions (ground truth)
│   ├── cylinder_re40/
│   │   └── cylinder_re40_reference.npz
│   ├── naca0012_re1k/
│   │   └── naca0012_re1k_reference.npz
│   ├── naca0012_re10k/
│   │   └── naca0012_re10k_reference.npz
│   └── fsi1_re20/
│       └── turek_hron_fsi1_reference.npz
│
├── scripts/                           # scripts to regenerate reference data
│   ├── cylinder_re40/
│   │   └── generate_cylinder_reference.py
│   ├── naca0012_re1k/
│   │   └── generate_naca_re1k_reference.py
│   ├── naca0012_re10k/
│   │   ├── naca_re10k_su2_setup.py
│   │   └── naca_re10k_su2_postprocess.py
│   └── fsi1_re20/
│       └── generate_fsi1_reference.py
│
├── results/                           # per-trial outputs from sweep + extras
│   ├── sensitivity_sweep/             # 72 trials: 4 cases × 6 fracs × 3 seeds
│   │   ├── cylinder/
│   │   ├── naca_re1k/
│   │   ├── naca_re10k/
│   │   └── fsi1/
│   ├── n_scaling_f01/                 # N-scaling extras at f=0.1
│   │   ├── cylinder/
│   │   ├── naca_re1k/
│   │   ├── naca_re10k/
│   │   └── fsi1/
│   ├── rbf_baseline/                  # RBF interpolation comparison
│   └── ablation/                      # adaptive collocation + RAR ablations
│
└── analysis/                          # post-processing utilities
    ├── check_sweep.py                 # walks results/ and prints summary
    ├── make_figures.py                # generates manuscript figures
    └── compute_forces.py              # drag/lift coefficient computation
```

## Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 1.12 (with CUDA for GPU training)
- NumPy, SciPy, Matplotlib

```bash
pip install -r requirements.txt
```

### Clone
```bash
git clone https://github.com/halabi95/SPHINX.git
cd SPHINX
```

## Usage

Each case directory contains a self-contained training script. All scripts share a common command-line interface:

```bash
# Run SPHINX at the operating point (N=200, f=0.1)
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --interface-frac 0.1

# Run baseline XPINN (no sparse data)
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 0

# Run with pure random placement (for comparison)
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --interface-frac 0.0

# Sweep over interface fractions
for frac in 0.0 0.1 0.2 0.3 0.4 0.5; do
  python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --interface-frac $frac --seed 42
done

# Specify seed for reproducibility (manuscript uses seeds 42, 123, 456)
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --interface-frac 0.1 --seed 42
```

### Common arguments

| Argument | Default | Description |
|---|---|---|
| `--n_sparse` | 200 | Number of sparse measurement points (use 0 for baseline) |
| `--interface-frac` | 0.1 | Fraction of points placed in interface overlap regions |
| `--seed` | 42 | Random seed for reproducibility |
| `--epochs` | 100000 | Number of training epochs |
| `--output-dir` | `./output` | Where to write `metrics.npz` |

### Reproducing manuscript results

To reproduce the sensitivity sweep for one case (18 trials: 6 fractions × 3 seeds):
```bash
for frac in 0.0 0.1 0.2 0.3 0.4 0.5; do
  for seed in 42 123 456; do
    python cases/cylinder_re40/sphinx_cylinder.py \
        --n_sparse 200 --interface-frac $frac --seed $seed \
        --output-dir results/sensitivity_sweep/cylinder/frac_${frac}/seed_${seed}
  done
done
```

To reproduce the N-scaling extras at f = 0.1 (12 trials per case):
```bash
for N in 30 50 100 500; do
  for seed in 42 123 456; do
    python cases/cylinder_re40/sphinx_cylinder.py \
        --n_sparse $N --interface-frac 0.1 --seed $seed \
        --output-dir results/n_scaling_f01/cylinder/N${N}/seed_${seed}
  done
done
```

Total compute for the full sweep is approximately 72 trials × 250 minutes ≈ 300 GPU-hours per case. Results were obtained on the SQUID supercomputer at Osaka University.

After training, summarize results with:
```bash
python analysis/check_sweep.py
```

## Method

### Domain decomposition

SPHINX uses K = 4 overlapping subdomains following a Schwarz-type decomposition. For external flows around a bluff body or airfoil, the four subdomains are:

- Upstream Ω₀ (freestream region ahead of the body)
- Top-Wake Ω₁ (downstream above the centerline)
- Bottom-Wake Ω₂ (downstream below the centerline)
- Near-Body Ω₃ (boundary-layer and stagnation region; overlaps with the other three)

For the Turek-Hron channel geometry the same four-region partition is retained but adapted to the channel walls. The four-subdomain decomposition yields six interface overlap regions in principle, of which fewer are geometrically realized depending on the case.

### Interface-aware data placement

Given a total budget of N sparse measurement points:
- **10% of points (fN, default f = 0.1)** are placed in subdomain overlap regions
- **90% of points ((1-f)N)** are scattered randomly through the fluid domain, excluding solid regions

Each interface point provides a *dual-anchoring* constraint: because both neighboring networks Nᵢ and Nⱼ are evaluated at the same point in the overlap, a single measurement enters the data loss with both network outputs penalized against the shared reference value. This (i) constrains each network individually and (ii) enforces inter-network agreement through external data rather than through the soft penalty term Lᵢₙₜf.

### Network architecture

Each subdomain uses an independent fully-connected network:
- 8 hidden layers × 128 neurons
- Tanh activation
- Xavier normal initialization
- approximately 116,000 parameters per subdomain

### Training

- Adam optimizer, initial learning rate 10⁻³
- Cosine annealing to 10⁻⁵ over 100,000 epochs
- Loss weights λ_BC = λ_data = λ_intf = 10 (PDE residual at unit weight)
- 2,500 collocation points per subdomain (resampled every epoch)
- 500 interface points per subdomain pair (resampled every epoch)

## Reference data

Reference CFD solutions in `data/` were generated with the following solvers and configurations:

| Case | Solver | Configuration |
|---|---|---|
| Cylinder Re = 40 | FEniCS | Steady incompressible Navier-Stokes, ~50,000 triangular elements |
| NACA 0012 Re = 1,000 (α = 5°) | FEniCS | Steady incompressible Navier-Stokes, refined mesh |
| NACA 0012 Re = 10,000 (α = 5°) | SU2 | Steady RANS with Spalart-Allmaras turbulence model |
| Turek-Hron FSI1 (Re = 20) | turtleFSI | Steady FSI with elastic beam, structural deformation prescribed for PINN training |

Each `.npz` file contains:
- `coordinates` — (N_nodes, 2) array of (x, y) positions
- `u`, `v` — velocity components at each node
- `p` — pressure at each node

The FSI1 reference additionally contains `topology`, `domain_markers`, and `displacement` arrays for the deformed beam geometry. Scripts to regenerate the reference data from scratch are in `scripts/`.

## Sensitivity sweep results

The `results/sensitivity_sweep/` directory contains per-trial `metrics.npz` files from the 72-trial sensitivity study (4 cases × 6 fractions × 3 seeds). Each `metrics.npz` contains:
- `u_err`, `v_err`, `p_err` — relative L2 errors as percentages
- `U_pred`, `V_pred`, `P_pred` — predicted field arrays at the reference nodes (omitted for FSI1 to keep file size manageable)
- `n_points`, `interface_frac`, `seed` — run configuration

Summary statistics (mean ± standard deviation across three seeds) are reproduced in the manuscript Tables 1-6. The `analysis/check_sweep.py` script walks the results directory and prints the summary.

## Citation

If you use SPHINX in your research, please cite:

```bibtex
@article{elhalabi2026sphinx,
  title={Interface-Aware Sparse Data Placement for Extended Physics-Informed
         Neural Networks Applied to Computational Fluid Dynamics and
         Fluid-Structure Interaction},
  author={El Halabi, Hamze and Takahashi, Yusuke},
  year={2026},
  note={Submitted to Engineering with Computers}
}
```

This citation will be updated once the manuscript is accepted.

## Contact

Hamze El Halabi
Graduate School of Engineering, Hokkaido University
Email: hamzahalhalabi@gmail.com

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Computations were performed on the SQUID supercomputer at the D3 Center, Osaka University, through the HPCI System Research Project (Project ID: hp260073). This study was supported by JSPS KAKENHI grant number 24K01072.
