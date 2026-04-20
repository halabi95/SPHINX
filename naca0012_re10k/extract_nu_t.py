"""
==============================================================================
Extract Eddy Viscosity from SU2 for RANS-Informed SPHINX
==============================================================================
Reads the SU2 solution, extracts nu_t(x,y), and adds it to the reference .npz
==============================================================================
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata

print("="*60)
print("  Extracting Eddy Viscosity from SU2")
print("="*60)

# Read VTU
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("flow.vtu")
reader.Update()
output = reader.GetOutput()

pts = vtk_to_numpy(output.GetPoints().GetData())
points = pts[:, :2]

point_data = output.GetPointData()
n_arrays = point_data.GetNumberOfArrays()
arr_names = [point_data.GetArrayName(i) for i in range(n_arrays)]
print(f"  Available: {arr_names}")

# Get eddy viscosity (this is mu_t, not nu_t)
mu_t_data = vtk_to_numpy(point_data.GetArray('Eddy_Viscosity'))
mu_lam = vtk_to_numpy(point_data.GetArray('Laminar_Viscosity'))

rho = 1.0  # from our setup
nu_t_data = mu_t_data / rho
nu_lam_data = mu_lam / rho
nu_eff_data = nu_lam_data + nu_t_data

print(f"  nu_lam range: [{nu_lam_data.min():.6f}, {nu_lam_data.max():.6f}]")
print(f"  nu_t range:   [{nu_t_data.min():.6f}, {nu_t_data.max():.6f}]")
print(f"  nu_eff range: [{nu_eff_data.min():.6f}, {nu_eff_data.max():.6f}]")
print(f"  Max nu_t / nu_lam ratio: {(nu_t_data / nu_lam_data).max():.1f}")

# Load existing reference
ref = np.load('naca0012_re10k_aoa5_reference.npz')
X, Y = ref['X'], ref['Y']
x_grid, y_grid = ref['x_grid'], ref['y_grid']

# Interpolate nu_eff onto rectangular grid
xi = np.column_stack([X.ravel(), Y.ravel()])
nu_t_grid = griddata(points, nu_t_data, xi, method='linear').reshape(X.shape)
nu_eff_grid = griddata(points, nu_eff_data, xi, method='linear').reshape(X.shape)

# Mask inside airfoil
chord = float(ref['chord'])
def naca0012_ht(x, c=1.0):
    t = 0.12
    xc = np.clip(x/c, 0, 1)
    return 5.0*t*(0.2969*np.sqrt(xc)-0.1260*xc-0.3516*xc**2+0.2843*xc**3-0.1015*xc**4)*c

def inside_af(x, y, margin=0.005):
    xc = x / chord
    in_range = (xc >= 0) & (xc <= 1)
    yt = np.zeros_like(x)
    yt[in_range] = naca0012_ht(x[in_range], chord)
    return in_range & (np.abs(y) < yt + margin)

mask = inside_af(X.ravel(), Y.ravel(), margin=0.005).reshape(X.shape)
nu_t_grid[mask] = np.nan
nu_eff_grid[mask] = np.nan

print(f"\n  nu_t grid: valid={np.isfinite(nu_t_grid).sum()}, "
      f"range=[{np.nanmin(nu_t_grid):.6f}, {np.nanmax(nu_t_grid):.6f}]")
print(f"  nu_eff grid: valid={np.isfinite(nu_eff_grid).sum()}, "
      f"range=[{np.nanmin(nu_eff_grid):.6f}, {np.nanmax(nu_eff_grid):.6f}]")

# Save updated reference with nu_t and nu_eff
save_dict = {k: ref[k] for k in ref.files}
save_dict['nu_t'] = nu_t_grid
save_dict['nu_eff'] = nu_eff_grid

np.savez('naca0012_re10k_aoa5_reference.npz', **save_dict)
print(f"\n  Updated: naca0012_re10k_aoa5_reference.npz (added nu_t, nu_eff)")

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x_af = ref['x_airfoil']; y_af = ref['y_airfoil']
xlim, ylim = (-0.5, 3.0), (-0.5, 0.5)

ax = axes[0]
c = ax.contourf(X, Y, np.ma.masked_invalid(nu_t_grid), levels=50, cmap='hot_r')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('$\\nu_t$ (Eddy Viscosity)')

ax = axes[1]
c = ax.contourf(X, Y, np.ma.masked_invalid(nu_eff_grid), levels=50, cmap='hot_r')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('$\\nu_{eff} = \\nu + \\nu_t$')

ax = axes[2]
ratio = np.ma.masked_invalid(nu_t_grid / float(ref['nu']))
c = ax.contourf(X, Y, ratio, levels=np.linspace(0, 50, 50), cmap='hot_r', extend='max')
plt.colorbar(c, ax=ax, shrink=0.85, label='$\\nu_t / \\nu$')
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('$\\nu_t / \\nu$ ratio')

fig.suptitle('Eddy Viscosity from SU2 RANS (k-ω SST)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_eddy_viscosity.png', dpi=150)
plt.savefig('fig_eddy_viscosity.pdf', dpi=300)
plt.close()
print("  Saved: fig_eddy_viscosity.png/.pdf")
