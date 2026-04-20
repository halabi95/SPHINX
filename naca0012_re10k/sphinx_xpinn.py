"""
==============================================================================
RANS-Informed SPHINX for NACA 0012, Re=10,000
==============================================================================
Uses frozen eddy viscosity nu_t(x,y) from SU2 RANS solution.

PDE changes from laminar:
  u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν∇²u = 0

to RANS with frozen ν_t:
  u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ∇·[ν_eff(x,y) ∇u] = 0

where ν_eff(x,y) = ν + ν_t(x,y) is spatially varying.

The viscous term expands to:
  ∇·[ν_eff ∇u] = ν_eff·∇²u + ∂ν_eff/∂x · ∂u/∂x + ∂ν_eff/∂y · ∂u/∂y
==============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ============================================================================
# 0. SETUP
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

data = np.load('naca0012_re10k_aoa5_reference.npz')
Re = float(data['Re']); nu = float(data['nu']); U_inf = float(data['U_inf'])
chord = float(data['chord']); AoA_deg = float(data['AoA_deg'])
AoA_rad = float(data['AoA_rad'])
U_x = float(data['U_x']); U_y = float(data['U_y'])
Cd = float(data['Cd']); Cl = float(data['Cl'])
X_ref, Y_ref = data['X'], data['Y']
U_ref, V_ref, P_ref = data['U'], data['V'], data['P']
x_grid, y_grid = data['x_grid'], data['y_grid']
x_min_d = float(data['x_extract_min']); x_max_d = float(data['x_extract_max'])
y_min_d = float(data['y_extract_min']); y_max_d = float(data['y_extract_max'])
x_af = data['x_airfoil']; y_af = data['y_airfoil']

# Load eddy viscosity
if 'nu_eff' not in data.files:
    print("ERROR: nu_eff not found in reference file!")
    print("Run extract_nu_t.py first.")
    exit(1)

nu_eff_field = data['nu_eff']
nu_t_field = data['nu_t']

print(f"  NACA 0012, Re={Re}, AoA={AoA_deg}°")
print(f"  RANS-Informed: using frozen ν_eff(x,y) from SU2 k-ω SST")
print(f"  ν_eff range: [{np.nanmin(nu_eff_field):.6f}, {np.nanmax(nu_eff_field):.6f}]")
print(f"  Max ν_t/ν ratio: {np.nanmax(nu_t_field)/nu:.0f}")

# Interpolator for nu_eff
interp_nu_eff = RegularGridInterpolator((y_grid, x_grid), nu_eff_field,
                                         method='linear', bounds_error=False,
                                         fill_value=nu)

interp_u = RegularGridInterpolator((y_grid, x_grid), U_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)
interp_v = RegularGridInterpolator((y_grid, x_grid), V_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)
interp_p = RegularGridInterpolator((y_grid, x_grid), P_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)


# ============================================================================
# 1. AIRFOIL GEOMETRY
# ============================================================================
def naca0012_half_thickness(x, c=1.0):
    t = 0.12
    xc = np.clip(x/c, 0, 1)
    return 5.0*t*(0.2969*np.sqrt(xc)-0.1260*xc-0.3516*xc**2+0.2843*xc**3-0.1015*xc**4)*c

def inside_airfoil(x, y, margin=0.01):
    xc = x / chord
    in_range = (xc >= 0) & (xc <= 1)
    yt = np.zeros_like(x)
    yt[in_range] = naca0012_half_thickness(x[in_range], chord)
    return in_range & (np.abs(y) < yt + margin)


# ============================================================================
# 2. DOMAIN DECOMPOSITION (same as before)
# ============================================================================
x_split = 0.5 * chord; y_split = 0.0; overlap = 0.3
subdomains = {
    0: {'name': 'Upstream',  'x_lo': x_min_d, 'x_hi': x_split+overlap,
        'y_lo': y_min_d, 'y_hi': y_max_d},
    1: {'name': 'Top-Wake',  'x_lo': x_split-overlap, 'x_hi': x_max_d,
        'y_lo': y_split-overlap, 'y_hi': y_max_d},
    2: {'name': 'Bot-Wake',  'x_lo': x_split-overlap, 'x_hi': x_max_d,
        'y_lo': y_min_d, 'y_hi': y_split+overlap},
    3: {'name': 'Near-Body', 'x_lo': -0.3, 'x_hi': 1.8,
        'y_lo': -0.8, 'y_hi': 0.8},
}
interfaces = [
    (0, 1, 'vertical'), (0, 2, 'vertical'), (1, 2, 'horizontal'),
    (0, 3, 'overlap'), (1, 3, 'overlap'), (2, 3, 'overlap'),
]

print(f"\n  Domain decomposition: {len(subdomains)} subdomains")
for k, sd in subdomains.items():
    print(f"    Ω{k} ({sd['name']}): [{sd['x_lo']:.1f},{sd['x_hi']:.1f}] × [{sd['y_lo']:.1f},{sd['y_hi']:.1f}]")


# ============================================================================
# 3. NETWORK & SAMPLING (same as before)
# ============================================================================
class SubPINN(nn.Module):
    def __init__(self, n_layers=4, n_neurons=40):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, xy):
        return self.net(xy)

def sample_subdomain(sd, n, rng):
    pts = []
    while len(pts) < n:
        x = rng.uniform(sd['x_lo'], sd['x_hi'], n*3)
        y = rng.uniform(sd['y_lo'], sd['y_hi'], n*3)
        mask = ~inside_airfoil(x, y, margin=0.01)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n-len(pts))):
            pts.append([gx[i], gy[i]])
    return np.array(pts[:n])

def sample_interface(sd_i, sd_j, n, rng):
    x_lo = max(sd_i['x_lo'], sd_j['x_lo']); x_hi = min(sd_i['x_hi'], sd_j['x_hi'])
    y_lo = max(sd_i['y_lo'], sd_j['y_lo']); y_hi = min(sd_i['y_hi'], sd_j['y_hi'])
    if x_lo >= x_hi or y_lo >= y_hi: return np.zeros((0, 2))
    pts = []
    while len(pts) < n:
        x = rng.uniform(x_lo, x_hi, n*3); y = rng.uniform(y_lo, y_hi, n*3)
        mask = ~inside_airfoil(x, y, margin=0.02)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n-len(pts))): pts.append([gx[i], gy[i]])
    return np.array(pts[:n])

def point_in_subdomain(x, y, sd):
    return (x >= sd['x_lo']) & (x <= sd['x_hi']) & (y >= sd['y_lo']) & (y <= sd['y_hi'])

def get_bc_points_subdomain(sd):
    xy, u_vals, v_vals = [], [], []
    n_af = 400; beta = np.linspace(0, np.pi, n_af)
    xc = 0.5*(1-np.cos(beta)); yt = naca0012_half_thickness(xc*chord, chord)
    offset = 0.003
    for i in range(n_af):
        px, py = xc[i]*chord, yt[i]+offset
        if sd['x_lo']<=px<=sd['x_hi'] and sd['y_lo']<=py<=sd['y_hi']:
            xy.append([px,py]); u_vals.append(0.0); v_vals.append(0.0)
    for i in range(n_af):
        px, py = xc[i]*chord, -yt[i]-offset
        if sd['x_lo']<=px<=sd['x_hi'] and sd['y_lo']<=py<=sd['y_hi']:
            xy.append([px,py]); u_vals.append(0.0); v_vals.append(0.0)
    if sd['x_lo'] <= x_min_d+0.01:
        for yi in np.linspace(max(sd['y_lo'],y_min_d), min(sd['y_hi'],y_max_d), 150):
            xy.append([x_min_d,yi]); u_vals.append(U_x); v_vals.append(U_y)
    if sd['y_hi'] >= y_max_d-0.01:
        for xi in np.linspace(max(sd['x_lo'],x_min_d), min(sd['x_hi'],x_max_d), 100):
            xy.append([xi,y_max_d]); u_vals.append(U_x); v_vals.append(U_y)
    if sd['y_lo'] <= y_min_d+0.01:
        for xi in np.linspace(max(sd['x_lo'],x_min_d), min(sd['x_hi'],x_max_d), 100):
            xy.append([xi,y_min_d]); u_vals.append(U_x); v_vals.append(U_y)
    if len(xy) == 0: return None, None, None
    return (torch.tensor(np.array(xy), dtype=torch.float32).to(device),
            torch.tensor(np.array(u_vals), dtype=torch.float32).to(device).unsqueeze(1),
            torch.tensor(np.array(v_vals), dtype=torch.float32).to(device).unsqueeze(1))

def sample_sparse_random(n_pts, rng):
    pts = []
    while len(pts) < n_pts:
        x = rng.uniform(x_min_d+0.1, x_max_d-0.1, n_pts*3)
        y = rng.uniform(y_min_d+0.1, y_max_d-0.1, n_pts*3)
        mask = ~inside_airfoil(x, y, margin=0.02)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n_pts-len(pts))): pts.append([gx[i], gy[i]])
    return np.array(pts[:n_pts])

def sample_interface_data(n_intf, rng):
    pts = []
    for (i, j, itype) in interfaces:
        sd_i, sd_j = subdomains[i], subdomains[j]
        x_lo = max(sd_i['x_lo'], sd_j['x_lo']); x_hi = min(sd_i['x_hi'], sd_j['x_hi'])
        y_lo = max(sd_i['y_lo'], sd_j['y_lo']); y_hi = min(sd_i['y_hi'], sd_j['y_hi'])
        if x_lo >= x_hi or y_lo >= y_hi: continue
        n_per = max(2, n_intf // len(interfaces))
        x = rng.uniform(x_lo, x_hi, n_per*3); y = rng.uniform(y_lo, y_hi, n_per*3)
        mask = ~inside_airfoil(x, y, margin=0.02)
        gx, gy = x[mask], y[mask]
        for k in range(min(len(gx), n_per)): pts.append([gx[k], gy[k]])
    return np.array(pts) if pts else np.zeros((0, 2))


# ============================================================================
# 4. RANS-INFORMED PDE RESIDUAL (the key change!)
# ============================================================================
def compute_rans_pde_loss(model, xy_c):
    """
    RANS momentum + continuity with frozen eddy viscosity.
    NaN-safe: clamps nu_eff and replaces NaN gradients with 0.
    """
    xy_c.requires_grad_(True)
    out = model(xy_c)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # Get nu_eff at collocation points (from SU2 interpolator)
    xy_np = xy_c.detach().cpu().numpy()
    yx_np = xy_np[:, ::-1].copy()
    nu_eff_vals = interp_nu_eff(yx_np)
    
    # Replace NaN with laminar viscosity and clamp
    nu_eff_vals = np.where(np.isfinite(nu_eff_vals), nu_eff_vals, nu)
    nu_eff_vals = np.clip(nu_eff_vals, nu, nu * 200)  # cap at 200× laminar
    nu_eff_t = torch.tensor(nu_eff_vals, dtype=torch.float32, device=device).unsqueeze(1)

    # Compute ∂ν_eff/∂x and ∂ν_eff/∂y numerically (NaN-safe)
    eps = 1e-4
    xy_px = xy_np.copy(); xy_px[:, 0] += eps
    xy_mx = xy_np.copy(); xy_mx[:, 0] -= eps
    xy_py = xy_np.copy(); xy_py[:, 1] += eps
    xy_my = xy_np.copy(); xy_my[:, 1] -= eps

    nu_px = interp_nu_eff(xy_px[:, ::-1].copy())
    nu_mx = interp_nu_eff(xy_mx[:, ::-1].copy())
    nu_py = interp_nu_eff(xy_py[:, ::-1].copy())
    nu_my = interp_nu_eff(xy_my[:, ::-1].copy())
    
    # Replace NaN with the center value (effectively sets gradient to 0 there)
    nu_px = np.where(np.isfinite(nu_px), nu_px, nu_eff_vals)
    nu_mx = np.where(np.isfinite(nu_mx), nu_mx, nu_eff_vals)
    nu_py = np.where(np.isfinite(nu_py), nu_py, nu_eff_vals)
    nu_my = np.where(np.isfinite(nu_my), nu_my, nu_eff_vals)

    dnu_dx_np = (nu_px - nu_mx) / (2*eps)
    dnu_dy_np = (nu_py - nu_my) / (2*eps)
    
    # Final NaN safety
    dnu_dx_np = np.where(np.isfinite(dnu_dx_np), dnu_dx_np, 0.0)
    dnu_dy_np = np.where(np.isfinite(dnu_dy_np), dnu_dy_np, 0.0)
    
    dnu_dx = torch.tensor(dnu_dx_np, dtype=torch.float32, device=device).unsqueeze(1)
    dnu_dy = torch.tensor(dnu_dy_np, dtype=torch.float32, device=device).unsqueeze(1)

    # First derivatives
    gu = torch.autograd.grad(u, xy_c, torch.ones_like(u), create_graph=True)[0]
    gv = torch.autograd.grad(v, xy_c, torch.ones_like(v), create_graph=True)[0]
    gp = torch.autograd.grad(p, xy_c, torch.ones_like(p), create_graph=True)[0]
    du_dx, du_dy = gu[:, 0:1], gu[:, 1:2]
    dv_dx, dv_dy = gv[:, 0:1], gv[:, 1:2]
    dp_dx, dp_dy = gp[:, 0:1], gp[:, 1:2]

    # Second derivatives
    d2u_dx2 = torch.autograd.grad(du_dx, xy_c, torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(du_dy, xy_c, torch.ones_like(du_dy), create_graph=True)[0][:, 1:2]
    d2v_dx2 = torch.autograd.grad(dv_dx, xy_c, torch.ones_like(dv_dx), create_graph=True)[0][:, 0:1]
    d2v_dy2 = torch.autograd.grad(dv_dy, xy_c, torch.ones_like(dv_dy), create_graph=True)[0][:, 1:2]

    # RANS viscous terms: ∇·[ν_eff ∇u] = ν_eff·∇²u + ∇ν_eff · ∇u
    visc_u = nu_eff_t * (d2u_dx2 + d2u_dy2) + dnu_dx * du_dx + dnu_dy * du_dy
    visc_v = nu_eff_t * (d2v_dx2 + d2v_dy2) + dnu_dx * dv_dx + dnu_dy * dv_dy

    # Residuals
    rc = du_dx + dv_dy
    rx = u * du_dx + v * du_dy + dp_dx - visc_u
    ry = u * dv_dx + v * dv_dy + dp_dy - visc_v

    return torch.mean(rc**2) + torch.mean(rx**2) + torch.mean(ry**2)


# ============================================================================
# 5. XPINN TRAINING (uses RANS PDE)
# ============================================================================
def run_xpinn_trial(seed, sparse_xy=None, sparse_u=None, sparse_v=None, sparse_p=None,
                    lambda_data=10.0, lambda_intf=10.0, N_epochs=20000):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    rng = np.random.RandomState(seed)

    models = {k: SubPINN(n_layers=4, n_neurons=40).to(device) for k in subdomains}
    bc_data = {k: get_bc_points_subdomain(sd) for k, sd in subdomains.items()}

    sparse_per_sd = {k: {'xy': None} for k in subdomains}
    has_data = sparse_xy is not None and len(sparse_xy) > 0
    if has_data:
        for k, sd in subdomains.items():
            mask = point_in_subdomain(sparse_xy[:,0], sparse_xy[:,1], sd)
            if mask.any():
                sparse_per_sd[k] = {
                    'xy': torch.tensor(sparse_xy[mask], dtype=torch.float32).to(device),
                    'u': torch.tensor(sparse_u[mask], dtype=torch.float32).to(device).unsqueeze(1),
                    'v': torch.tensor(sparse_v[mask], dtype=torch.float32).to(device).unsqueeze(1),
                    'p': torch.tensor(sparse_p[mask], dtype=torch.float32).to(device).unsqueeze(1),
                }

    all_params = []
    for m in models.values(): all_params += list(m.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_epochs, eta_min=1e-5)

    N_colloc = 2500; N_intf = 500; lambda_bc = 10.0

    t0 = time.time()
    for ep in range(1, N_epochs+1):
        for m in models.values(): m.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for k, sd in subdomains.items():
            xy_c = torch.tensor(sample_subdomain(sd, N_colloc, rng), dtype=torch.float32).to(device)
            loss_pde = compute_rans_pde_loss(models[k], xy_c)  # RANS PDE!
            total_loss = total_loss + loss_pde

            xy_bc, u_bc, v_bc = bc_data[k]
            if xy_bc is not None and len(xy_bc) > 0:
                out_bc = models[k](xy_bc)
                loss_bc = torch.mean((out_bc[:,0:1]-u_bc)**2) + torch.mean((out_bc[:,1:2]-v_bc)**2)
                total_loss = total_loss + lambda_bc * loss_bc

            if has_data and sparse_per_sd[k]['xy'] is not None:
                sd_d = sparse_per_sd[k]
                out_d = models[k](sd_d['xy'])
                loss_data = (torch.mean((out_d[:,0:1]-sd_d['u'])**2) +
                             torch.mean((out_d[:,1:2]-sd_d['v'])**2) +
                             torch.mean((out_d[:,2:3]-sd_d['p'])**2))
                total_loss = total_loss + lambda_data * loss_data

        # Interface losses
        for (i, j, itype) in interfaces:
            sd_i, sd_j = subdomains[i], subdomains[j]
            xy_if = sample_interface(sd_i, sd_j, N_intf, rng)
            if len(xy_if) == 0: continue

            xy_if_t = torch.tensor(xy_if, dtype=torch.float32).to(device)
            xy_if_t.requires_grad_(True)
            out_i = models[i](xy_if_t); out_j = models[j](xy_if_t)

            loss_val = (torch.mean((out_i[:,0]-out_j[:,0])**2) +
                        torch.mean((out_i[:,1]-out_j[:,1])**2) +
                        torch.mean((out_i[:,2]-out_j[:,2])**2))

            x_lo_ov = max(sd_i['x_lo'], sd_j['x_lo']); x_hi_ov = min(sd_i['x_hi'], sd_j['x_hi'])
            y_lo_ov = max(sd_i['y_lo'], sd_j['y_lo']); y_hi_ov = min(sd_i['y_hi'], sd_j['y_hi'])
            normal = torch.tensor([[1.0, 0.0]], device=device) if (x_hi_ov-x_lo_ov) < (y_hi_ov-y_lo_ov) else torch.tensor([[0.0, 1.0]], device=device)

            u_i, v_i, p_i = out_i[:,0:1], out_i[:,1:2], out_i[:,2:3]
            grad_u_i = torch.autograd.grad(u_i, xy_if_t, torch.ones_like(u_i), create_graph=True)[0]
            grad_v_i = torch.autograd.grad(v_i, xy_if_t, torch.ones_like(v_i), create_graph=True)[0]
            grad_p_i = torch.autograd.grad(p_i, xy_if_t, torch.ones_like(p_i), create_graph=True)[0]
            dudn_i = torch.sum(grad_u_i*normal, dim=1, keepdim=True)
            dvdn_i = torch.sum(grad_v_i*normal, dim=1, keepdim=True)
            dpdn_i = torch.sum(grad_p_i*normal, dim=1, keepdim=True)

            xy_if_t2 = xy_if_t.detach().clone().requires_grad_(True)
            out_j2 = models[j](xy_if_t2)
            u_j, v_j, p_j = out_j2[:,0:1], out_j2[:,1:2], out_j2[:,2:3]
            grad_u_j = torch.autograd.grad(u_j, xy_if_t2, torch.ones_like(u_j), create_graph=True)[0]
            grad_v_j = torch.autograd.grad(v_j, xy_if_t2, torch.ones_like(v_j), create_graph=True)[0]
            grad_p_j = torch.autograd.grad(p_j, xy_if_t2, torch.ones_like(p_j), create_graph=True)[0]
            dudn_j = torch.sum(grad_u_j*normal, dim=1, keepdim=True)
            dvdn_j = torch.sum(grad_v_j*normal, dim=1, keepdim=True)
            dpdn_j = torch.sum(grad_p_j*normal, dim=1, keepdim=True)

            loss_grad = (torch.mean((dudn_i-dudn_j)**2) + torch.mean((dvdn_i-dvdn_j)**2) +
                         torch.mean((dpdn_i-dpdn_j)**2))
            total_loss = total_loss + lambda_intf * (loss_val + loss_grad)

        total_loss.backward()
        optimizer.step(); scheduler.step()

        if ep % 5000 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"    Ep {ep:6d}/{N_epochs} | L:{total_loss.item():.4e} | {elapsed:.0f}s")

    t_train = time.time() - t0

    # Evaluate
    for m in models.values(): m.eval()
    x_flat, y_flat = X_ref.flatten(), Y_ref.flatten()
    with torch.no_grad():
        u_accum = np.zeros_like(x_flat); v_accum = np.zeros_like(x_flat)
        p_accum = np.zeros_like(x_flat); count = np.zeros_like(x_flat)
        for k, sd in subdomains.items():
            mask = point_in_subdomain(x_flat, y_flat, sd) & ~inside_airfoil(x_flat, y_flat, margin=0.005)
            if not mask.any(): continue
            for s in range(0, mask.sum(), 10000):
                e = min(s+10000, mask.sum()); idx = np.where(mask)[0][s:e]
                xy = np.stack([x_flat[idx], y_flat[idx]], axis=1)
                o = models[k](torch.tensor(xy, dtype=torch.float32).to(device)).cpu().numpy()
                u_accum[idx] += o[:,0]; v_accum[idx] += o[:,1]; p_accum[idx] += o[:,2]
                count[idx] += 1

    vc = count > 0
    u_pred = np.full_like(x_flat, np.nan); v_pred = np.full_like(x_flat, np.nan); p_pred = np.full_like(x_flat, np.nan)
    u_pred[vc] = u_accum[vc]/count[vc]; v_pred[vc] = v_accum[vc]/count[vc]; p_pred[vc] = p_accum[vc]/count[vc]
    U_p = u_pred.reshape(X_ref.shape); V_p = v_pred.reshape(X_ref.shape); P_p = p_pred.reshape(X_ref.shape)

    valid = ~np.isnan(U_ref) & ~np.isnan(U_p)
    ue = np.sqrt(np.nanmean((U_p[valid]-U_ref[valid])**2))/np.sqrt(np.nanmean(U_ref[valid]**2))*100
    ve = np.sqrt(np.nanmean((V_p[valid]-V_ref[valid])**2))/np.sqrt(np.nanmean(V_ref[valid]**2))*100
    pe = np.sqrt(np.nanmean((P_p[valid]-P_ref[valid])**2))/np.sqrt(np.nanmean(P_ref[valid]**2))*100
    return ue, ve, pe, t_train, total_loss.item(), models, U_p, V_p, P_p


# ============================================================================
# 6. SPARSE DATA
# ============================================================================
print("\nGenerating sparse data (interface-aware)...")
sparse_data = {}
for n_sp in [30, 50, 100, 200]:
    data_rng = np.random.RandomState(999+n_sp)
    n_random = int(0.7*n_sp); n_intf = n_sp - n_random
    xy_rand = sample_sparse_random(n_random, data_rng)
    xy_intf = sample_interface_data(n_intf, data_rng)
    xy = np.vstack([xy_rand, xy_intf[:n_intf]]) if len(xy_intf) > 0 else xy_rand
    yx = xy[:,::-1]
    u_sp = interp_u(yx); v_sp = interp_v(yx); p_sp = interp_p(yx)
    valid = ~np.isnan(u_sp) & ~np.isnan(v_sp) & ~np.isnan(p_sp)
    sparse_data[n_sp] = {'xy': xy[valid], 'u': u_sp[valid], 'v': v_sp[valid], 'p': p_sp[valid]}
    print(f"  N={n_sp}: {valid.sum()} valid")


# ============================================================================
# 7. RUN EXPERIMENTS
# ============================================================================
N_trials = 3; N_epochs = 20000; seeds = [42, 123, 456]
configs = [
    ('XPINN-RANS Baseline', 0),
    ('SPHINX-XR N=30', 30),
    ('SPHINX-XR N=50', 50),
    ('SPHINX-XR N=100', 100),
    ('SPHINX-XR N=200', 200),
]

all_results = {}
total_experiments = len(configs) * N_trials
exp_count = 0; t_global = time.time()
best_u_err = 999; best_fields = None; best_config = None

for config_name, n_sparse in configs:
    print(f"\n{'='*60}")
    print(f"  {config_name} — {N_trials} trials")
    print(f"{'='*60}")
    u_trials, v_trials, p_trials, t_trials = [], [], [], []
    for trial, seed in enumerate(seeds):
        exp_count += 1
        print(f"\n  Trial {trial+1}/{N_trials} (seed={seed}) [{exp_count}/{total_experiments}]")
        if n_sparse > 0:
            sd = sparse_data[n_sparse]
            ue, ve, pe, tt, loss, models, Up, Vp, Pp = run_xpinn_trial(
                seed, sd['xy'], sd['u'], sd['v'], sd['p'], lambda_data=10.0, N_epochs=N_epochs)
        else:
            ue, ve, pe, tt, loss, models, Up, Vp, Pp = run_xpinn_trial(seed, N_epochs=N_epochs)
        u_trials.append(ue); v_trials.append(ve); p_trials.append(pe); t_trials.append(tt)
        print(f"    u={ue:.2f}%, v={ve:.2f}%, p={pe:.2f}% | {tt:.0f}s")
        if ue < best_u_err:
            best_u_err = ue; best_fields = (Up.copy(), Vp.copy(), Pp.copy()); best_config = config_name
        elapsed = time.time() - t_global
        remaining = (elapsed/exp_count) * (total_experiments - exp_count)
        print(f"    [{elapsed/60:.0f} min elapsed, ~{remaining/60:.0f} min remaining]")

    all_results[config_name] = {
        'n_sparse': n_sparse,
        'u_mean': np.mean(u_trials), 'u_std': np.std(u_trials),
        'v_mean': np.mean(v_trials), 'v_std': np.std(v_trials),
        'p_mean': np.mean(p_trials), 'p_std': np.std(p_trials),
        'u_all': u_trials, 'v_all': v_trials, 'p_all': p_trials,
        't_mean': np.mean(t_trials),
    }
    r = all_results[config_name]
    print(f"\n  {config_name} Summary:")
    print(f"    u: {r['u_mean']:.2f} ± {r['u_std']:.2f}%")
    print(f"    v: {r['v_mean']:.2f} ± {r['v_std']:.2f}%")
    print(f"    p: {r['p_mean']:.2f} ± {r['p_std']:.2f}%")

t_total = time.time() - t_global

# Save
save_dict = {}
for name, r in all_results.items():
    key = name.replace(' ', '_').replace('-', '_').replace('=', '')
    for var in ['u', 'v', 'p']:
        save_dict[f'{key}_{var}_mean'] = r[f'{var}_mean']
        save_dict[f'{key}_{var}_std'] = r[f'{var}_std']
        save_dict[f'{key}_{var}_all'] = r[f'{var}_all']
if best_fields:
    save_dict['best_U'] = best_fields[0]; save_dict['best_V'] = best_fields[1]; save_dict['best_P'] = best_fields[2]
np.savez('sphinx_xpinn_rans_naca_re10k_results.npz', **save_dict)

# Results table
print(f"\n{'='*75}")
print(f"  RANS-Informed SPHINX-X: NACA 0012, Re={int(Re)}, AoA={AoA_deg}°")
print(f"{'='*75}")
print(f"  {'Method':<24s} {'u error (%)':>15} {'v error (%)':>15} {'p error (%)':>15}")
print(f"  {'-'*69}")
for name, r in all_results.items():
    print(f"  {name:<24s} {r['u_mean']:6.2f} ± {r['u_std']:.2f}  {r['v_mean']:6.2f} ± {r['v_std']:.2f}  {r['p_mean']:6.2f} ± {r['p_std']:.2f}")

bl = all_results['XPINN-RANS Baseline']
best_key = min([k for k in all_results if 'Baseline' not in k], key=lambda k: all_results[k]['u_mean'])
best = all_results[best_key]
print(f"\n  Best: {best_key}")
print(f"  u: {bl['u_mean']:.2f}% → {best['u_mean']:.2f}% ({(1-best['u_mean']/bl['u_mean'])*100:.0f}% reduction)")
print(f"  p: {bl['p_mean']:.2f}% → {best['p_mean']:.2f}% ({(1-best['p_mean']/bl['p_mean'])*100:.0f}% reduction)")
print(f"\n  Total time: {t_total:.0f}s ({t_total/60:.1f} min, {t_total/3600:.1f} hrs)")

# Compare with laminar results
print(f"\n  --- Comparison: Laminar vs RANS-Informed SPHINX-X ---")
print(f"  Laminar N=200:      u=4.64%, v=24.51%, p=36.01%")
print(f"  RANS-Informed N=200: u={best['u_mean']:.2f}%, v={best['v_mean']:.2f}%, p={best['p_mean']:.2f}%")
print(f"{'='*75}")
