# SPHINX: Sparse Physics-Hybrid Informed Neural eXtension

**Interface-aware sparse data placement for Extended Physics-Informed Neural Networks (XPINNs)**

SPHINX is a targeted data placement strategy that exploits the domain decomposition structure of XPINNs. Rather than distributing sparse measurement data randomly, SPHINX allocates **30% of the data budget to subdomain overlap regions** and 70% randomly throughout the domain. Each interface data point simultaneously constrains both neighboring networks, providing a dual anchoring effect that reinforces inter-subdomain consistency.

SPHINX requires no modifications to the XPINN architecture, loss function weights, or training algorithm — it is purely a data placement strategy.

## Key Results

| Benchmark | Re | u-error reduction | Sparse points |
|-----------|----|--------------------|---------------|
| Cylinder flow | 40 | 85% | 200 |
| NACA 0012 airfoil (α=10°) | 1,000 | 92% | 200 |
| NACA 0012 airfoil (α=5°) | 10,000 | 91% | 200 |
| Turek–Hron FSI1 | 20 | 89% | 200 |

Run-to-run variance collapses by approximately two orders of magnitude compared to random placement.

## Repository Structure

```
SPHINX/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── cases/
│   ├── cylinder_re40/
│   │   └── sphinx_cylinder.py
│   ├── naca0012_re1k/
│   │   └── sphinx_naca_re1k.py
│   ├── naca0012_re10k/
│   │   └── sphinx_naca_re10k.py
│   └── fsi1_re20/
│       └── sphinx_fsi1.py
│
├── data/
│   ├── cylinder_re40/
│   │   └── cylinder_re40_reference.npz
│   ├── naca0012_re1k/
│   │   └── naca0012_re1k_reference.npz
│   ├── naca0012_re10k/
│   │   └── naca0012_re10k_reference.npz
│   └── fsi1_re20/
│       └── turek_hron_fsi1_reference.npz
│
└── scripts/
    ├── cylinder_re40/
    │   └── generate_cylinder_reference.py
    ├── naca0012_re1k/
    │   └── generate_naca_re1k_reference.py
    ├── naca0012_re10k/
    │   ├── generate_naca_re10k_setup.py
    │   └── generate_naca_re10k_postprocess.py
    └── fsi1_re20/
        └── generate_fsi1_reference.py
```

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (with CUDA for GPU training)
- NumPy, SciPy, Matplotlib

```bash
pip install torch numpy scipy matplotlib
```

### Clone

```bash
git clone https://github.com/hamze-elhalabi/SPHINX.git
cd SPHINX
```

## Usage

Each case directory contains a self-contained script. All scripts share the same interface:

```bash
# Run SPHINX with 200 sparse points (interface-aware placement)
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --mode interface

# Run baseline (no sparse data)
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 0

# Run with random placement (for comparison)
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --mode random

# Specify seed for reproducibility
python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --seed 42
```

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_sparse` | 200 | Number of sparse measurement points |
| `--mode` | `interface` | Placement mode: `interface` (30/70) or `random` |
| `--seed` | 42 | Random seed for reproducibility |
| `--epochs` | 100000 | Number of training epochs |
| `--trials` | 3 | Number of independent runs with different seeds |

### Reproducing Paper Results

To reproduce all results from the paper (baseline + SPHINX + random, 3 seeds each):

```bash
# Example for cylinder case
for mode in interface random; do
  for seed in 42 123 456; do
    python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 200 --mode $mode --seed $seed
  done
done

# Baseline (no data)
for seed in 42 123 456; do
  python cases/cylinder_re40/sphinx_cylinder.py --n_sparse 0 --seed $seed
done
```

Repeat for each case directory.

## Method Details

### Domain Decomposition

SPHINX uses 4 overlapping subdomains following a Schwarz-type decomposition:
- Upstream region
- Top-downstream region
- Bottom-downstream region
- Near-body region (overlaps with all three others)

The overlap width is 15% of the reference length, creating 6 interface overlap regions.

### Sparse Data Placement

- **30% of points** are placed in subdomain overlap regions, where each point constrains two neighboring networks simultaneously
- **70% of points** are scattered randomly through the fluid domain

### Network Architecture

Each subdomain uses an independent fully-connected network:
- 8 hidden layers × 128 neurons
- Tanh activation
- Xavier normal initialization
- ~135,000 parameters per subdomain

### Training

- Adam optimizer, learning rate 10⁻³
- Cosine annealing schedule to 10⁻⁵
- 100,000 epochs
- All loss weights fixed at λ = 10 (BC, data, interface); PDE residual at unit weight

## Reference Data

Reference solutions included in `data/` were generated with:

| Case | Solver | Notes |
|------|--------|-------|
| Cylinder Re=40 | FEniCS | Steady incompressible N-S |
| NACA 0012 Re=1,000 | FEniCS | Steady incompressible N-S |
| NACA 0012 Re=10,000 | SU2 | RANS with k-ω SST |
| Turek–Hron FSI1 Re=20 | turtleFSI | Steady FSI with elastic beam |

Scripts to regenerate reference data from scratch are in `scripts/`.

Each `.npz` file contains `coordinates`, `velocity`, and `pressure` arrays. The FSI1 reference additionally contains `topology`, `domain_markers`, and `displacement`.

## Citation

If you use SPHINX in your research, please cite:

```bibtex
@article{elhalabi2026sphinx,
  title={SPHINX: Interface-Aware Sparse Data Placement for Extended Physics-Informed Neural Networks},
  author={El Halabi, Hamze and Takahashi, Yusuke},
  journal={Physics of Fluids},
  year={2026},
  note={Under review}
}
```

## Contact

Hamze El Halabi
Space Transportation System Laboratory, Hokkaido University
hamzahalhalabi@gmail.com

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

Computations were performed on the SQUID supercomputer at the Cybermedia Center, Osaka University.
