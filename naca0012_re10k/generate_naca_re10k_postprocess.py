"""
==============================================================================
Post-process SU2 solution for SPHINX
==============================================================================
Reads the SU2 Paraview output (flow.vtu), extracts u, v, p onto a
rectangular grid, and saves in the same .npz format as the FEniCS cases.
==============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================================
# PARAMETERS
# ============================================================================
chord = 1.0
Re = 10000.0
AoA_deg = 5.0
AoA_rad = np.radians(AoA_deg)
U_inf = 1.0
nu = U_inf * chord / Re
U_x = U_inf * np.cos(AoA_rad)
U_y = U_inf * np.sin(AoA_rad)

print("="*60)
print("  SU2 Post-Processing for SPHINX")
print(f"  Re={Re}, AoA={AoA_deg}°")
print("="*60)

# ============================================================================
# READ VTU FILE
# ============================================================================
print("\nReading SU2 output...")

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName("flow.vtu")
    reader.Update()
    output = reader.GetOutput()

    pts = vtk_to_numpy(output.GetPoints().GetData())
    points = pts[:, :2]

    point_data = output.GetPointData()
    n_arrays = point_data.GetNumberOfArrays()
    arr_names = [point_data.GetArrayName(i) for i in range(n_arrays)]
    print(f"  Available arrays: {arr_names}")

    # Find pressure and velocity arrays
    p_data = None
    u_data = None
    v_data = None

    for name in arr_names:
        arr = vtk_to_numpy(point_data.GetArray(name))
        if name == 'Pressure':
            p_data = arr
            print(f"    Pressure from '{name}': shape={arr.shape}")
        if name == 'Velocity':
            if arr.ndim == 2 and arr.shape[1] >= 2:
                u_data = arr[:, 0]
                v_data = arr[:, 1]
                print(f"    Velocity from '{name}': shape={arr.shape}")

    if u_data is None or p_data is None:
        print("  ERROR: Could not find velocity/pressure fields")
        print("  Trying all arrays...")
        for i in range(n_arrays):
            name = point_data.GetArrayName(i)
            arr = vtk_to_numpy(point_data.GetArray(name))
            print(f"    {name}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")
        exit(1)

    print(f"  Read {len(points)} points via VTK")

except ImportError:
    print("  VTK not available. Install: pip install vtk --break-system-packages")
    exit(1)

print(f"  x range: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
print(f"  y range: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
print(f"  u range: [{u_data.min():.4f}, {u_data.max():.4f}]")
print(f"  v range: [{v_data.min():.4f}, {v_data.max():.4f}]")
print(f"  p range: [{p_data.min():.4f}, {p_data.max():.4f}]")

# Read Cd, Cl from SU2 output (from last iteration in history)
Cd, Cl = 0.046989, 0.403678  # from SU2 screen output
print(f"  Cd = {Cd}, Cl = {Cl}")


# ============================================================================
# AIRFOIL GEOMETRY
# ============================================================================
def naca0012_half_thickness(x):
    t = 0.12
    xc = np.clip(x / chord, 0, 1)
    return 5.0 * t * (
        0.2969 * np.sqrt(xc) - 0.1260 * xc
        - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4
    ) * chord

def inside_airfoil(x, y, margin=0.0):
    xc = x / chord
    in_range = (xc >= 0) & (xc <= 1)
    yt = np.zeros_like(x)
    yt[in_range] = naca0012_half_thickness(x[in_range])
    return in_range & (np.abs(y) < yt + margin)

# Airfoil coordinates for plotting
n_af = 200
beta = np.linspace(0, np.pi, n_af)
xc_af = 0.5 * (1 - np.cos(beta))
yt_af = naca0012_half_thickness(xc_af * chord)
x_af = np.concatenate([xc_af[::-1], xc_af[1:]]) * chord
y_af = np.concatenate([yt_af[::-1], -yt_af[1:]])


# ============================================================================
# INTERPOLATE ONTO RECTANGULAR GRID
# ============================================================================
print("\nInterpolating onto rectangular grid...")

from scipy.interpolate import griddata

# Extraction region (same scale as Re=1000 case)
x_ext_min, x_ext_max = -0.5, 3.0
y_ext_min, y_ext_max = -1.5, 1.5
nx, ny = 350, 300

x_grid = np.linspace(x_ext_min, x_ext_max, nx)
y_grid = np.linspace(y_ext_min, y_ext_max, ny)
X, Y = np.meshgrid(x_grid, y_grid)

# Interpolate using scipy griddata (linear)
xi = np.column_stack([X.ravel(), Y.ravel()])

U_flat = griddata(points, u_data, xi, method='linear')
V_flat = griddata(points, v_data, xi, method='linear')
P_flat = griddata(points, p_data, xi, method='linear')

U_field = U_flat.reshape(X.shape)
V_field = V_flat.reshape(X.shape)
P_field = P_flat.reshape(X.shape)

# Mask points inside airfoil
x_flat, y_flat = X.ravel(), Y.ravel()
mask_inside = inside_airfoil(x_flat, y_flat, margin=0.005)
U_field.ravel()[mask_inside] = np.nan
V_field.ravel()[mask_inside] = np.nan
P_field.ravel()[mask_inside] = np.nan

valid = ~np.isnan(U_field)
print(f"  Grid: {nx} x {ny} = {nx*ny}")
print(f"  Valid points: {valid.sum()}")
print(f"  Inside airfoil: {mask_inside.sum()}")


# ============================================================================
# SAVE
# ============================================================================
outfile = 'naca0012_re10k_aoa5_reference.npz'
np.savez(outfile,
         X=X, Y=Y,
         x_grid=x_grid, y_grid=y_grid,
         U=U_field, V=V_field, P=P_field,
         Re=Re, AoA_deg=AoA_deg, AoA_rad=AoA_rad,
         nu=nu, U_inf=U_inf, chord=chord,
         U_x=U_x, U_y=U_y,
         Cd=Cd, Cl=Cl,
         x_extract_min=x_ext_min, x_extract_max=x_ext_max,
         y_extract_min=y_ext_min, y_extract_max=y_ext_max,
         x_airfoil=x_af, y_airfoil=y_af,
         solver='SU2', turbulence_model='SST',
         )
print(f"\n  Saved: {outfile}")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating plots...")

U_m = np.ma.masked_invalid(U_field)
V_m = np.ma.masked_invalid(V_field)
P_m = np.ma.masked_invalid(P_field)
speed = np.ma.sqrt(U_m**2 + V_m**2)

xlim = (-0.5, 3.0)
ylim = (-1.0, 1.0)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'NACA 0012 (SU2 RANS, k-ω SST) — Re={int(Re)}, AoA={AoA_deg}°, Cd={Cd:.4f}, Cl={Cl:.4f}',
             fontsize=15, fontweight='bold')

# u-velocity
ax = axes[0, 0]
c = ax.contourf(X, Y, U_m, levels=np.linspace(-0.1, 1.3, 50), cmap='RdBu_r', extend='both')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('(a) $u$-velocity'); ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')

# v-velocity
ax = axes[0, 1]
c = ax.contourf(X, Y, V_m, levels=np.linspace(-0.3, 0.3, 50), cmap='RdBu_r', extend='both')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('(b) $v$-velocity'); ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')

# Pressure
ax = axes[1, 0]
c = ax.contourf(X, Y, P_m, levels=50, cmap='coolwarm', extend='both')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('(c) Pressure'); ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')

# Speed + streamlines
ax = axes[1, 1]
c = ax.contourf(X, Y, speed, levels=np.linspace(0, 1.4, 50), cmap='viridis')
plt.colorbar(c, ax=ax, shrink=0.85, label='$|\\mathbf{u}|/U_\\infty$')
ax.streamplot(x_grid, y_grid, U_m.filled(U_x), V_m.filled(U_y),
              color='white', linewidth=0.5, density=3, arrowsize=0.5)
ax.fill(x_af, y_af, color='0.4', zorder=5, edgecolor='white', lw=0.5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('(d) Speed + Streamlines'); ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')

plt.tight_layout()
plt.savefig('fig_naca0012_re10k_reference.png', dpi=150)
plt.savefig('fig_naca0012_re10k_reference.pdf', dpi=300)
plt.close()
print("  Saved: fig_naca0012_re10k_reference.png/.pdf")

print(f"""
{'='*60}
  POST-PROCESSING COMPLETE
{'='*60}
  Re = {Re}, AoA = {AoA_deg}°
  Cd = {Cd}, Cl = {Cl}
  Solver: SU2 RANS (k-ω SST)
  Grid: {nx} x {ny}, valid points: {valid.sum()}
  Saved: {outfile}
{'='*60}
""")
