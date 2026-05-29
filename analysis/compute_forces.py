"""
compute_forces.py — Drag and lift coefficient computation from saved SPHINX models.

This script integrates the full Cauchy stress (pressure + viscous) over the body
surface using PyTorch autograd to evaluate velocity gradients on the saved network
weights. Forces are then rotated into the wind-axis frame to recover the
conventional drag and lift coefficients.

USAGE
-----
This script requires saved network checkpoints (`.pth` files) from a SPHINX
training run, plus the corresponding reference CFD solution for surface geometry.
By default, the sweep training scripts in this repository do not save model
weights to keep storage manageable. To compute forces, re-run the training
script with the `--save-model` flag enabled (where supported), or refer to the
table of reference force values reported in the manuscript (Table 10).

    python analysis/compute_forces.py \\
        --case cylinder_re40 \\
        --models cases/cylinder_re40/sphinx_best_model_sd*.pth \\
        --reference data/cylinder_re40/cylinder_re40_reference.npz

REFERENCE FORCE VALUES (from manuscript Table 10)
-------------------------------------------------
   Case                   Cd_ref  Cd_SPHINX  Cd_diff  Cl_ref  Cl_SPHINX  Cl_diff
   Cylinder Re=40          1.561    1.590    +0.029    0.000    -0.003    -0.003
   NACA Re=1k (a=5)        0.152    0.123    -0.029    0.345     0.295    -0.050
   NACA Re=10k (a=5)       0.047    0.042    -0.005    0.404     0.267    -0.137
   FSI1 Re=20              1.195    1.090    -0.105    0.213     0.128    -0.085

Across all four cases, SPHINX recovers Cd to within |Delta Cd| <= 0.11 and
Cl to within |Delta Cl| <= 0.14, using only 200 sparse measurement points and
without any force-specific terms in the loss function.

NOTES
-----
Finite-difference gradients on grid-resolution field arrays cannot resolve the
wall shear stress accurately and will underestimate the viscous contribution to
drag. This is why the script uses autograd on the trained network weights:
gradients are evaluated symbolically at the wall offset points and capture the
boundary-layer gradient accurately. This is the same procedure used to produce
the manuscript Table 10 values.
"""

import argparse
import glob
import os
import numpy as np
import torch


# -------------------------------------------------------------------- #
# Network definition (must match the trained architecture)
# -------------------------------------------------------------------- #
class SubnetMLP(torch.nn.Module):
    """Fully-connected MLP for one subdomain: (x, y) -> (u, v, p)."""

    def __init__(self, hidden_layers=8, hidden_neurons=128):
        super().__init__()
        layers = [torch.nn.Linear(2, hidden_neurons), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [torch.nn.Linear(hidden_neurons, hidden_neurons), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden_neurons, 3)]  # u, v, p
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


# -------------------------------------------------------------------- #
# Surface sampling
# -------------------------------------------------------------------- #
def sample_cylinder_surface(D=1.0, n_points=400, offset=0.003):
    """Returns (xy, normals) for a circular cylinder of diameter D."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = (D / 2.0) + offset  # slight offset from wall into the fluid
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    nx = np.cos(theta)  # outward normal
    ny = np.sin(theta)
    return (np.stack([x, y], axis=1).astype(np.float32),
            np.stack([nx, ny], axis=1).astype(np.float32))


def sample_naca_surface(chord=1.0, alpha_deg=5.0, n_points=400, offset=0.003):
    """NACA 0012 surface points and outward normals.

    Cosine-spaced along the chord to concentrate points near LE/TE where the
    pressure gradient and curvature are largest. The airfoil is rotated by
    -alpha_deg around the leading edge so the chord line sits at -alpha
    relative to the freestream (matches the FEniCS reference convention).
    """
    n_per_side = n_points // 2
    beta = np.linspace(0, np.pi, n_per_side)
    x_c = 0.5 * (1 - np.cos(beta))  # 0 to 1, cosine-spaced
    # NACA 0012 thickness distribution
    t = 0.12
    y_t = 5 * t * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c**2
                   + 0.2843 * x_c**3 - 0.1015 * x_c**4)
    # Upper and lower surfaces
    x_upper = x_c * chord
    y_upper = y_t * chord
    x_lower = x_c[::-1] * chord
    y_lower = -y_t[::-1] * chord
    xs = np.concatenate([x_upper, x_lower])
    ys = np.concatenate([y_upper, y_lower])
    # Rotate by -alpha around LE
    a = np.deg2rad(-alpha_deg)
    xr = xs * np.cos(a) - ys * np.sin(a)
    yr = xs * np.sin(a) + ys * np.cos(a)
    # Outward normals: compute from tangent of the closed curve
    pts = np.stack([xr, yr], axis=1)
    tangents = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tangents /= (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12)
    # Outward = rotate tangent by -90 deg (counterclockwise traversal)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    # Apply offset into the fluid
    pts_offset = pts + offset * normals
    return pts_offset.astype(np.float32), normals.astype(np.float32)


# -------------------------------------------------------------------- #
# Force integration (autograd-based)
# -------------------------------------------------------------------- #
def compute_forces_autograd(model, xy_surface, normals, nu, U_inf, L_ref):
    """Integrate Cauchy stress over the surface using autograd gradients.

    sigma_ij = -p * delta_ij + nu * (du_i/dx_j + du_j/dx_i)
    F_i = integral over body surface of sigma_ij * n_j ds

    Returns (Fx, Fy, Cd_body_axis, Cl_body_axis).
    Body-axis means streamwise (x) and crossflow (y); rotation to wind axis
    happens in the caller.
    """
    device = next(model.parameters()).device
    xy = torch.tensor(xy_surface, requires_grad=True, device=device)
    n = torch.tensor(normals, device=device)

    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    # Gradients via autograd
    grad_u = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=False)[0]
    grad_v = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=False)[0]
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    # Stress tensor components
    sxx = -p + 2 * nu * du_dx
    syy = -p + 2 * nu * dv_dy
    sxy = nu * (du_dy + dv_dx)

    # Traction vector t_i = sigma_ij * n_j
    nx, ny = n[:, 0:1], n[:, 1:2]
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny

    # Surface element length: ds approximated from neighbour spacing
    pts = xy_surface
    dpts = np.roll(pts, -1, axis=0) - pts
    ds = np.linalg.norm(dpts, axis=1)
    ds_t = torch.tensor(ds.reshape(-1, 1).astype(np.float32), device=device)

    # Integrate
    Fx = (tx * ds_t).sum().item()
    Fy = (ty * ds_t).sum().item()

    # Coefficient form (per unit span for 2D)
    q = 0.5 * 1.0 * U_inf**2  # rho = 1 in nondimensional units
    Cd_body = Fx / (q * L_ref)
    Cl_body = Fy / (q * L_ref)

    return Fx, Fy, Cd_body, Cl_body


def rotate_to_wind_axis(Cd_body, Cl_body, alpha_deg):
    """Rotate body-axis forces to wind-axis (drag aligned with freestream)."""
    a = np.deg2rad(alpha_deg)
    Cd = Cd_body * np.cos(a) + Cl_body * np.sin(a)
    Cl = -Cd_body * np.sin(a) + Cl_body * np.cos(a)
    return Cd, Cl


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True,
                    choices=['cylinder_re40', 'naca0012_re1k', 'naca0012_re10k', 'fsi1_re20'])
    ap.add_argument('--models', nargs='+', required=True,
                    help='Paths to saved .pth model files (one per subdomain, '
                         'or one per seed for an aggregate)')
    ap.add_argument('--reference', required=True,
                    help='Path to reference solution .npz')
    ap.add_argument('--alpha', type=float, default=5.0, help='Angle of attack (deg)')
    args = ap.parse_args()

    # Reynolds-number dependent parameters
    if args.case == 'cylinder_re40':
        nu = 1.0 / 40.0
        U_inf = 1.0
        L_ref = 1.0
        xy_surf, normals = sample_cylinder_surface()
        alpha = 0.0
    elif args.case == 'naca0012_re1k':
        nu = 1.0 / 1000.0
        U_inf = 1.0
        L_ref = 1.0  # chord
        xy_surf, normals = sample_naca_surface(chord=1.0, alpha_deg=args.alpha)
        alpha = args.alpha
    elif args.case == 'naca0012_re10k':
        nu = 1.0 / 10000.0
        U_inf = 1.0
        L_ref = 1.0
        xy_surf, normals = sample_naca_surface(chord=1.0, alpha_deg=args.alpha)
        alpha = args.alpha
    else:  # fsi1
        nu = 0.001
        U_inf = 1.0
        L_ref = 0.1   # cylinder diameter
        # FSI1 surface is the union of cylinder + deformed beam; extracted from
        # the reference mesh topology, not constructed analytically.
        ref = np.load(args.reference, allow_pickle=True)
        if 'wet_surface' in ref:
            xy_surf = ref['wet_surface']
            normals = ref['wet_normals']
        else:
            raise RuntimeError(
                'FSI1 reference does not contain wet_surface / wet_normals. '
                'Either re-export the reference with the surface topology, '
                'or extract the surface from the mesh and pass as a separate file.'
            )
        alpha = 0.0

    # Load each model and compute forces; average across seeds
    Cds, Cls = [], []
    for path in args.models:
        if not os.path.exists(path):
            print(f'  [skip] {path} not found')
            continue
        model = SubnetMLP()
        try:
            model.load_state_dict(torch.load(path, map_location='cpu'))
        except Exception as e:
            print(f'  [skip] {path}: load failed ({e})')
            continue
        model.eval()
        Fx, Fy, Cd_body, Cl_body = compute_forces_autograd(
            model, xy_surf, normals, nu, U_inf, L_ref)
        Cd, Cl = rotate_to_wind_axis(Cd_body, Cl_body, alpha)
        print(f'  {os.path.basename(path)}: Cd = {Cd:.4f}, Cl = {Cl:.4f}')
        Cds.append(Cd); Cls.append(Cl)

    if Cds:
        print()
        print(f'  Case: {args.case}')
        print(f'  Cd  = {np.mean(Cds):.4f} ± {np.std(Cds):.4f}   (across {len(Cds)} model files)')
        print(f'  Cl  = {np.mean(Cls):.4f} ± {np.std(Cls):.4f}')
    else:
        print('\n  No models loaded; cannot compute forces.')
        print('  Reference values from manuscript Table 10 are listed in the script header.')


if __name__ == '__main__':
    main()
