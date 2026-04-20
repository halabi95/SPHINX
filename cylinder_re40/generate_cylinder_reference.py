"""
==============================================================================
Step 1: Reference CFD Solution — Steady 2D Flow Around a Cylinder (Re=40)
==============================================================================

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *
import mshr

# ============================================================================
# 1. PARAMETERS
# ============================================================================
Re = 40.0
U_inf = 1.0
D = 1.0
R = D / 2.0
nu = U_inf * D / Re

# Enlarged domain to minimize blockage effect
# Blockage ratio = D / (y_max - y_min) = 1/30 = 3.3%
x_min, x_max = -15.0, 25.0
y_min, y_max = -15.0, 15.0
cx, cy = 0.0, 0.0
mesh_resolution = 120

print("=" * 60)
print("  Steady Cylinder Flow CFD — FEniCS (v3)")
print("=" * 60)
print(f"  Reynolds number:  Re = {Re}")
print(f"  Free-stream vel:  U_inf = {U_inf}")
print(f"  Cylinder diameter: D = {D}")
print(f"  Kinematic visc:   nu = {nu:.6f}")
print(f"  Domain: [{x_min}, {x_max}] x [{y_min}, {y_max}]")
print(f"  Blockage ratio:   {D/(y_max-y_min)*100:.1f}%")
print(f"  Mesh resolution:  {mesh_resolution}")
print("=" * 60)

# ============================================================================
# 2. MESH GENERATION
# ============================================================================
rect = mshr.Rectangle(Point(x_min, y_min), Point(x_max, y_max))
cylinder = mshr.Circle(Point(cx, cy), R, 100)
domain = rect - cylinder

print("\nGenerating mesh...")
mesh = mshr.generate_mesh(domain, mesh_resolution)

# Refine mesh near the cylinder for better accuracy
print("Refining mesh near cylinder...")
for i in range(2):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    for cell in cells(mesh):
        mp = cell.midpoint()
        r = np.sqrt((mp[0] - cx)**2 + (mp[1] - cy)**2)
        if r < R + 1.5:
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

num_cells = mesh.num_cells()
num_vertices = mesh.num_vertices()
h_min = mesh.hmin()
h_max = mesh.hmax()

print(f"  Number of cells:    {num_cells}")
print(f"  Number of vertices: {num_vertices}")
print(f"  Min cell size:      {h_min:.6f}")
print(f"  Max cell size:      {h_max:.6f}")

# ============================================================================
# 3. FUNCTION SPACES
# ============================================================================
V_elem = VectorElement("CG", mesh.ufl_cell(), 2)
Q_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = MixedElement([V_elem, Q_elem])
W = FunctionSpace(mesh, W_elem)

print(f"\nFunction space DOFs: {W.dim()}")

# ============================================================================
# 4. BOUNDARY CONDITIONS
# ============================================================================
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_min, 1e-10)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_max, 1e-10)

class TopWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], y_max, 1e-10)

class BottomWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], y_min, 1e-10)

class CylinderSurface(SubDomain):
    def inside(self, x, on_boundary):
        r = np.sqrt((x[0] - cx)**2 + (x[1] - cy)**2)
        return on_boundary and r < R + 0.1

u_inlet = Constant((U_inf, 0.0))
u_noslip = Constant((0.0, 0.0))

bcs = [
    DirichletBC(W.sub(0), u_inlet,  Inlet()),
    DirichletBC(W.sub(0), u_inlet,  TopWall()),
    DirichletBC(W.sub(0), u_inlet,  BottomWall()),
    DirichletBC(W.sub(0), u_noslip, CylinderSurface()),
]

print("Boundary conditions defined.")

# ============================================================================
# 5. VARIATIONAL FORMULATION
# ============================================================================
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

nu_const = Constant(nu)

F = (
    inner(grad(u) * u, v) * dx
    + nu_const * inner(grad(u), grad(v)) * dx
    - p * div(v) * dx
    + q * div(u) * dx
)

J = derivative(F, w)

print("Variational formulation defined.")

# ============================================================================
# 6. SOLVE
# ============================================================================
print("\nSolving steady Navier-Stokes (Newton method)...")

problem = NonlinearVariationalProblem(F, w, bcs, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['maximum_iterations'] = 50
prm['newton_solver']['relative_tolerance'] = 1e-8
prm['newton_solver']['absolute_tolerance'] = 1e-10
prm['newton_solver']['linear_solver'] = 'mumps'

solver.solve()
print("Solve complete!")

(u_sol, p_sol) = w.split(deepcopy=True)

# ============================================================================
# 7. COMPUTE DRAG COEFFICIENT (FIXED SIGN)
# ============================================================================
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
CylinderSurface().mark(boundary_markers, 1)
ds_cyl = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=1)

n = FacetNormal(mesh)
sigma = -p_sol * Identity(2) + nu * (grad(u_sol) + grad(u_sol).T)
traction = dot(sigma, n)

# Negate: FEniCS normal points outward from fluid (into cylinder)
drag = -assemble(traction[0] * ds_cyl)
lift = -assemble(traction[1] * ds_cyl)

Cd = 2.0 * drag / (U_inf**2 * D)
Cl = 2.0 * lift / (U_inf**2 * D)

print(f"\n--- Flow Quantities ---")
print(f"  Drag coefficient:  Cd = {Cd:.4f}  (expected ~1.54 for Re=40)")
print(f"  Lift coefficient:  Cl = {Cl:.6f}  (expected ~0.0 by symmetry)")

# ============================================================================
# 8. EXTRACT DATA ON STRUCTURED GRID
# ============================================================================
# Extract on a focused region around the cylinder for PINN use
# Full domain is too large; focus on [-3, 10] x [-4, 4]
print("\nExtracting solution on structured grid...")

nx_grid, ny_grid = 300, 200
# Save both full domain bounds and extraction region
x_extract_min, x_extract_max = -3.0, 10.0
y_extract_min, y_extract_max = -4.0, 4.0
x_grid = np.linspace(x_extract_min, x_extract_max, nx_grid)
y_grid = np.linspace(y_extract_min, y_extract_max, ny_grid)
X, Y = np.meshgrid(x_grid, y_grid)

x_flat = X.flatten()
y_flat = Y.flatten()

u_data = np.full_like(x_flat, np.nan)
v_data = np.full_like(x_flat, np.nan)
p_data = np.full_like(x_flat, np.nan)

print(f"  Evaluating at {len(x_flat)} grid points...")
print(f"  Extraction region: [{x_extract_min}, {x_extract_max}] x [{y_extract_min}, {y_extract_max}]")
for i in range(len(x_flat)):
    xi, yi = x_flat[i], y_flat[i]
    r = np.sqrt((xi - cx)**2 + (yi - cy)**2)
    if r < R + 0.01:
        continue
    try:
        point = Point(xi, yi)
        u_val = u_sol(point)
        p_val = p_sol(point)
        u_data[i] = u_val[0]
        v_data[i] = u_val[1]
        p_data[i] = p_val
    except RuntimeError:
        continue

U_grid = u_data.reshape(ny_grid, nx_grid)
V_grid = v_data.reshape(ny_grid, nx_grid)
P_grid = p_data.reshape(ny_grid, nx_grid)

# ============================================================================
# 9. EXTRACT BOUNDARY DATA
# ============================================================================
print("Extracting boundary data...")

n_cyl_pts = 200
theta = np.linspace(0, 2 * np.pi, n_cyl_pts, endpoint=False)
x_cyl = cx + R * np.cos(theta)
y_cyl = cy + R * np.sin(theta)

p_cyl = np.zeros(n_cyl_pts)
for i in range(n_cyl_pts):
    try:
        p_cyl[i] = p_sol(Point(x_cyl[i], y_cyl[i]))
    except RuntimeError:
        offset = 1.01
        p_cyl[i] = p_sol(Point(cx + R * offset * np.cos(theta[i]),
                                cy + R * offset * np.sin(theta[i])))

Cp_cyl = (p_cyl - 0.0) / (0.5 * U_inf**2)

# ============================================================================
# 10. SAVE ALL DATA
# ============================================================================
output_file = "cylinder_re40_reference.npz"
np.savez(output_file,
         x_grid=x_grid, y_grid=y_grid, X=X, Y=Y,
         U=U_grid, V=V_grid, P=P_grid,
         x_cyl=x_cyl, y_cyl=y_cyl, theta_cyl=theta, p_cyl=p_cyl, Cp_cyl=Cp_cyl,
         Re=Re, U_inf=U_inf, D=D, R=R, nu=nu,
         # Full CFD domain bounds
         x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
         # Extraction region bounds (used for grid data)
         x_extract_min=x_extract_min, x_extract_max=x_extract_max,
         y_extract_min=y_extract_min, y_extract_max=y_extract_max,
         cx=cx, cy=cy, Cd=Cd, Cl=Cl)

print(f"\nData saved to '{output_file}'")

# ============================================================================
# 11. VISUALIZATION
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Steady Cylinder Flow — Re = {Re} (Cd = {Cd:.4f})", fontsize=14, fontweight='bold')

ax = axes[0, 0]
U_plot = np.ma.masked_invalid(U_grid)
c1 = ax.contourf(X, Y, U_plot, levels=50, cmap='RdBu_r', vmin=-0.3, vmax=1.3)
plt.colorbar(c1, ax=ax, label='u/U∞')
circle = plt.Circle((cx, cy), R, color='gray', fill=True, zorder=5)
ax.add_patch(circle)
ax.set_title('Streamwise Velocity (u)')
ax.set_xlabel('x/D'); ax.set_ylabel('y/D')
ax.set_aspect('equal'); ax.set_xlim(-2, 8); ax.set_ylim(-3, 3)

ax = axes[0, 1]
V_plot = np.ma.masked_invalid(V_grid)
c2 = ax.contourf(X, Y, V_plot, levels=50, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
plt.colorbar(c2, ax=ax, label='v/U∞')
circle = plt.Circle((cx, cy), R, color='gray', fill=True, zorder=5)
ax.add_patch(circle)
ax.set_title('Cross-stream Velocity (v)')
ax.set_xlabel('x/D'); ax.set_ylabel('y/D')
ax.set_aspect('equal'); ax.set_xlim(-2, 8); ax.set_ylim(-3, 3)

ax = axes[1, 0]
P_plot = np.ma.masked_invalid(P_grid)
c3 = ax.contourf(X, Y, P_plot, levels=50, cmap='coolwarm')
plt.colorbar(c3, ax=ax, label='p')
circle = plt.Circle((cx, cy), R, color='gray', fill=True, zorder=5)
ax.add_patch(circle)
ax.set_title('Pressure Field')
ax.set_xlabel('x/D'); ax.set_ylabel('y/D')
ax.set_aspect('equal'); ax.set_xlim(-2, 8); ax.set_ylim(-3, 3)

ax = axes[1, 1]
theta_deg = np.degrees(theta)
ax.plot(theta_deg, Cp_cyl, 'b-', linewidth=2)
ax.set_xlabel('θ (degrees)'); ax.set_ylabel('Cp')
ax.set_title('Surface Pressure Coefficient')
ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
ax.set_xlim(0, 360); ax.grid(True, alpha=0.3); ax.invert_yaxis()

plt.tight_layout()
plt.savefig('cylinder_re40_reference.png', dpi=150, bbox_inches='tight')
print("  Saved: cylinder_re40_reference.png")

fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
U_stream = np.ma.masked_invalid(U_grid)
V_stream = np.ma.masked_invalid(V_grid)
speed = np.sqrt(U_stream**2 + V_stream**2)
c4 = ax2.contourf(X, Y, speed, levels=50, cmap='viridis', vmin=0, vmax=1.3)
plt.colorbar(c4, ax=ax2, label='|u|/U∞')
ax2.streamplot(x_grid, y_grid, U_stream.filled(0), V_stream.filled(0),
               color='white', linewidth=0.5, density=2, arrowsize=0.8)
circle = plt.Circle((cx, cy), R, color='gray', fill=True, zorder=5)
ax2.add_patch(circle)
ax2.set_title(f'Velocity Magnitude & Streamlines — Re = {Re}', fontsize=13)
ax2.set_xlabel('x/D'); ax2.set_ylabel('y/D')
ax2.set_aspect('equal'); ax2.set_xlim(-2, 8); ax2.set_ylim(-3, 3)
plt.tight_layout()
plt.savefig('cylinder_re40_streamlines.png', dpi=150, bbox_inches='tight')
print("  Saved: cylinder_re40_streamlines.png")

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Re = {Re},  Cd = {Cd:.4f},  Cl = {Cl:.6f}")
print(f"  Grid: {nx_grid} x {ny_grid} = {nx_grid*ny_grid} points")
print(f"  Mesh: {num_cells} cells, {num_vertices} vertices")
print(f"  CFD domain: [{x_min}, {x_max}] x [{y_min}, {y_max}]")
print(f"  Extract region: [{x_extract_min}, {x_extract_max}] x [{y_extract_min}, {y_extract_max}]")
print(f"  Data saved: {output_file}")
print("=" * 60)
