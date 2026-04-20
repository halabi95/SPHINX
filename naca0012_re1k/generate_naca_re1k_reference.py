"""
==============================================================================
NACA 0012 Airfoil — CFD Reference Solution with FEniCS
==============================================================================
Flow conditions:
  - NACA 0012 symmetric airfoil, chord c = 1.0
  - Re = 1000 (based on chord and freestream velocity)
  - Angle of attack α = 10°
  - Steady, 2D, incompressible Navier-Stokes
  - Expect mild trailing-edge separation at this Re/AoA

Domain:
  - Large circular far-field (radius = 20c)
  - Refined near airfoil surface and wake region
  - Data extracted onto rectangular region for PINN training

Output:
  - naca0012_re1000_aoa10_reference.npz
==============================================================================
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ============================================================================
# 1. NACA 0012 Geometry
# ============================================================================
def naca0012_coords(n_pts=200, chord=1.0):
    """
    Generate NACA 0012 airfoil coordinates using standard formula.
    
    Thickness distribution:
      y_t/c = 5*0.12*(0.2969*sqrt(x/c) - 0.1260*(x/c) 
              - 0.3516*(x/c)^2 + 0.2843*(x/c)^3 - 0.1015*(x/c)^4)
    
    Uses cosine spacing for better leading-edge resolution.
    Returns points going clockwise: TE upper → LE → TE lower.
    """
    t = 0.12  # max thickness for NACA 0012
    
    # Cosine spacing: clusters points near LE and TE
    beta = np.linspace(0, np.pi, n_pts)
    xc = 0.5 * (1 - np.cos(beta))  # x/c from 0 to 1
    
    # Thickness distribution (closed trailing edge version)
    yt = 5.0 * t * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    ) * chord
    
    x = xc * chord
    
    # Upper surface: TE → LE (reversed)
    x_upper = x[::-1]
    y_upper = yt[::-1]
    
    # Lower surface: LE → TE (skip first point = LE, already in upper)
    x_lower = x[1:]
    y_lower = -yt[1:]
    
    # Combine clockwise
    x_all = np.concatenate([x_upper, x_lower])
    y_all = np.concatenate([y_upper, y_lower])
    
    return x_all, y_all


# ============================================================================
# 2. FEniCS Setup
# ============================================================================
try:
    from dolfin import *
    import mshr
except ImportError:
    print("ERROR: FEniCS/mshr not found.")
    print("Install: conda install -c conda-forge fenics mshr -y")
    exit(1)

# Suppress FEniCS info messages (keep warnings/errors)
set_log_level(30)

print("="*60)
print("  NACA 0012 CFD Reference Solution")
print("="*60)

# --- Flow parameters ---
chord = 1.0
Re = 1000.0
AoA_deg = 10.0
AoA_rad = np.radians(AoA_deg)
U_inf = 1.0
nu = U_inf * chord / Re

U_x = U_inf * np.cos(AoA_rad)
U_y = U_inf * np.sin(AoA_rad)

print(f"  Airfoil: NACA 0012, chord = {chord}")
print(f"  Re = {Re}, AoA = {AoA_deg}°")
print(f"  nu = {nu:.6f}")
print(f"  U_inf = ({U_x:.6f}, {U_y:.6f})")

# --- Domain geometry ---
R_far = 20.0 * chord  # far-field radius
cx_domain = 0.5 * chord  # center of the circular domain (mid-chord)
cy_domain = 0.0

print(f"  Far-field radius: {R_far}c")

# Build airfoil polygon
x_af, y_af = naca0012_coords(n_pts=150, chord=chord)
print(f"  Airfoil points: {len(x_af)}")

# mshr geometry
outer_circle = mshr.Circle(Point(cx_domain, cy_domain), R_far, 128)
airfoil_pts = [Point(x_af[i], y_af[i]) for i in range(len(x_af))]
airfoil_poly = mshr.Polygon(airfoil_pts)
domain = outer_circle - airfoil_poly

# --- Mesh generation with refinement ---
print("\nGenerating mesh...")
t0 = time.time()

mesh = mshr.generate_mesh(domain, 80)
print(f"  Base mesh: {mesh.num_cells()} cells")

# Refine near airfoil and in the wake
for pass_num in range(2):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)
    for cell in cells(mesh):
        mp = cell.midpoint()
        x, y = mp.x(), mp.y()
        
        # Distance from airfoil center
        dist = np.sqrt((x - 0.5*chord)**2 + y**2)
        
        # Refine near airfoil (within 2c)
        if dist < 2.0 * chord:
            cell_markers[cell] = True
        
        # Refine in wake (downstream strip)
        if chord < x < 6.0*chord and abs(y) < 1.0*chord:
            cell_markers[cell] = True
    
    mesh = refine(mesh, cell_markers)
    print(f"  Refine pass {pass_num+1}: {mesh.num_cells()} cells")

t_mesh = time.time() - t0
print(f"  Mesh time: {t_mesh:.1f}s")
print(f"  Final: {mesh.num_cells()} cells, h_min={mesh.hmin():.5f}")

# --- Function spaces (Taylor-Hood P2/P1) ---
P2 = VectorElement('P', mesh.ufl_cell(), 2)
P1 = FiniteElement('P', mesh.ufl_cell(), 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)

# --- Boundary conditions ---
class FarField(SubDomain):
    def inside(self, x, on_boundary):
        r = np.sqrt((x[0] - cx_domain)**2 + (x[1] - cy_domain)**2)
        return on_boundary and r > R_far * 0.9

class Airfoil(SubDomain):
    def inside(self, x, on_boundary):
        r = np.sqrt((x[0] - cx_domain)**2 + (x[1] - cy_domain)**2)
        return on_boundary and r < R_far * 0.9

farfield_bc = DirichletBC(W.sub(0),
                           Expression(('Ux', 'Uy'), Ux=U_x, Uy=U_y, degree=2),
                           FarField())
airfoil_bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), Airfoil())
bcs = [farfield_bc, airfoil_bc]

# --- Variational form: Steady Navier-Stokes ---
print("\nSolving Navier-Stokes...")
w = Function(W)
(u, p) = split(w)
(v_test, q_test) = TestFunctions(W)

nu_c = Constant(nu)

F = (nu_c * inner(grad(u), grad(v_test)) * dx
     + inner(dot(grad(u), u), v_test) * dx
     - p * div(v_test) * dx
     + q_test * div(u) * dx)

# --- Solve with Reynolds number continuation ---
# At Re=1000 with AoA=10° (separated flow), Newton may not converge
# from a cold start. We ramp up Re gradually:
#   Stokes → Re=100 → Re=300 → Re=600 → Re=1000
# Each solution is the initial guess for the next.

t0 = time.time()

# Phase 1: Stokes initial guess (infinite Re, no convection)
print("  Phase 1: Stokes initial guess...")
w_stokes = Function(W)
(u_s, p_s) = split(w_stokes)
(v_s, q_s) = TestFunctions(W)

F_stokes = (nu_c * inner(grad(u_s), grad(v_s)) * dx
            - p_s * div(v_s) * dx
            + q_s * div(u_s) * dx)

solve(F_stokes == 0, w_stokes, bcs,
      solver_parameters={"newton_solver": {
          "maximum_iterations": 20,
          "relative_tolerance": 1e-6,
      }})
print("    Stokes solved.")
w.assign(w_stokes)

# Phase 2: Reynolds number continuation
Re_steps = [100, 300, 600, 1000]
print(f"  Phase 2: Re continuation {Re_steps}")

for Re_step in Re_steps:
    nu_step = U_inf * chord / Re_step
    nu_c.assign(nu_step)
    
    print(f"    Solving Re={Re_step} (nu={nu_step:.6f})...", end=" ", flush=True)
    try:
        solve(F == 0, w, bcs,
              solver_parameters={"newton_solver": {
                  "maximum_iterations": 50,
                  "relative_tolerance": 1e-6,
                  "absolute_tolerance": 1e-8,
                  "relaxation_parameter": 1.0,
              }})
        print("converged.")
    except RuntimeError:
        # If Newton fails, try with relaxation
        print("retrying with relaxation...", end=" ", flush=True)
        try:
            solve(F == 0, w, bcs,
                  solver_parameters={"newton_solver": {
                      "maximum_iterations": 100,
                      "relative_tolerance": 1e-5,
                      "absolute_tolerance": 1e-7,
                      "relaxation_parameter": 0.7,
                  }})
            print("converged (relaxed).")
        except RuntimeError:
            print(f"FAILED at Re={Re_step}. Stopping.")
            # Use the last converged solution
            break

t_solve = time.time() - t0
print(f"  Solve time: {t_solve:.1f}s ({t_solve/60:.1f} min)")

# --- Extract solution ---
(u_sol, p_sol) = w.split(deepcopy=True)
u_func = u_sol
p_func = p_sol

print("  Solution extracted.")


# ============================================================================
# 3. Compute Forces (Cd, Cl)
# ============================================================================
print("\nComputing aerodynamic coefficients...")

# Mark airfoil boundary
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
Airfoil().mark(boundary_markers, 2)
ds_airfoil = Measure("ds", domain=mesh, subdomain_data=boundary_markers)(2)

n = FacetNormal(mesh)
I = Identity(2)

# Stress tensor: sigma = -pI + 2*nu*eps(u)
sigma = -p_sol * I + 2.0 * nu * sym(grad(u_sol))

# Force = integral of sigma . n over airfoil
# NOTE: FEniCS facet normals point outward from the fluid domain,
# which means INTO the airfoil. The force ON the airfoil from the
# fluid is the negative of this integral.
#
# Drag direction: (cos(alpha), sin(alpha))  — along freestream
# Lift direction: (-sin(alpha), cos(alpha)) — perpendicular to freestream
drag_dir = Constant((np.cos(AoA_rad), np.sin(AoA_rad)))
lift_dir = Constant((-np.sin(AoA_rad), np.cos(AoA_rad)))

# Negative sign because n points into the airfoil
F_drag = -assemble(dot(dot(sigma, n), drag_dir) * ds_airfoil)
F_lift = -assemble(dot(dot(sigma, n), lift_dir) * ds_airfoil)

# Non-dimensionalize: Cd = F_drag / (0.5 * rho * U_inf^2 * chord)
# With rho=1, U_inf=1, chord=1: Cd = 2 * F_drag
Cd = 2.0 * F_drag
Cl = 2.0 * F_lift

print(f"  Cd = {Cd:.6f}")
print(f"  Cl = {Cl:.6f}")
print(f"  L/D = {Cl/Cd:.3f}")


# ============================================================================
# 4. Extract Data on Rectangular Grid
# ============================================================================
print("\nExtracting data on rectangular grid...")

# Extraction region: covers airfoil + near wake
# x: from -0.5c upstream to 3c downstream
# y: from -1.5c to 1.5c
x_ext_min, x_ext_max = -0.5, 3.0
y_ext_min, y_ext_max = -1.5, 1.5
nx, ny = 350, 300

x_grid = np.linspace(x_ext_min, x_ext_max, nx)
y_grid = np.linspace(y_ext_min, y_ext_max, ny)
X, Y = np.meshgrid(x_grid, y_grid)

# Evaluate solution on grid
U_field = np.full(X.shape, np.nan)
V_field = np.full(X.shape, np.nan)
P_field = np.full(X.shape, np.nan)

# Check if point is inside airfoil
def inside_airfoil(x, y):
    """
    Check if point (x,y) is inside the NACA 0012 airfoil.
    Uses the thickness formula directly.
    """
    if x < 0 or x > chord:
        return False
    xc = x / chord
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(xc) - 0.1260 * xc
        - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4
    ) * chord
    return abs(y) < yt

print("  Evaluating on grid...")
t0 = time.time()
n_evaluated = 0
n_inside = 0

for j in range(ny):
    for i in range(nx):
        x, y = X[j, i], Y[j, i]
        
        # Skip points inside airfoil
        if inside_airfoil(x, y):
            n_inside += 1
            continue
        
        pt = Point(x, y)
        try:
            uv = u_func(pt)
            pp = p_func(pt)
            U_field[j, i] = uv[0]
            V_field[j, i] = uv[1]
            P_field[j, i] = pp
            n_evaluated += 1
        except RuntimeError:
            # Point outside mesh
            pass

t_extract = time.time() - t0
print(f"  Extraction time: {t_extract:.1f}s")
print(f"  Points: {n_evaluated} evaluated, {n_inside} inside airfoil, "
      f"{nx*ny - n_evaluated - n_inside} outside mesh")


# ============================================================================
# 5. Extract Airfoil Surface Data
# ============================================================================
print("\nExtracting airfoil surface data...")

# Sample points just outside the airfoil surface
n_surf = 300
beta_surf = np.linspace(0, np.pi, n_surf)
xc_surf = 0.5 * (1 - np.cos(beta_surf))

# Thickness at each point
yt_surf = 5.0 * 0.12 * (
    0.2969 * np.sqrt(xc_surf) - 0.1260 * xc_surf
    - 0.3516 * xc_surf**2 + 0.2843 * xc_surf**3 - 0.1015 * xc_surf**4
) * chord

# Surface pressure on upper and lower
x_surf_upper = xc_surf * chord
y_surf_upper = yt_surf
x_surf_lower = xc_surf * chord
y_surf_lower = -yt_surf

Cp_upper = np.full(n_surf, np.nan)
Cp_lower = np.full(n_surf, np.nan)

offset = 0.005  # small offset from surface (into fluid)

for k in range(n_surf):
    # Upper surface
    try:
        # Normal points outward (approximately +y for upper surface)
        pt = Point(x_surf_upper[k], y_surf_upper[k] + offset)
        pp = p_func(pt)
        Cp_upper[k] = (pp - 0.0) / (0.5 * U_inf**2)  # Cp = (p - p_inf) / q_inf
    except:
        pass
    
    # Lower surface
    try:
        pt = Point(x_surf_lower[k], y_surf_lower[k] - offset)
        pp = p_func(pt)
        Cp_lower[k] = (pp - 0.0) / (0.5 * U_inf**2)
    except:
        pass


# ============================================================================
# 6. Save All Data
# ============================================================================
print("\nSaving data...")

outfile = 'naca0012_re1000_aoa10_reference.npz'
np.savez(outfile,
         # Grid
         X=X, Y=Y,
         x_grid=x_grid, y_grid=y_grid,
         U=U_field, V=V_field, P=P_field,
         # Parameters
         Re=Re, AoA_deg=AoA_deg, AoA_rad=AoA_rad,
         nu=nu, U_inf=U_inf, chord=chord,
         U_x=U_x, U_y=U_y,
         Cd=Cd, Cl=Cl,
         # Extraction region
         x_extract_min=x_ext_min, x_extract_max=x_ext_max,
         y_extract_min=y_ext_min, y_extract_max=y_ext_max,
         # Airfoil geometry
         x_airfoil=x_af, y_airfoil=y_af,
         # Surface data
         x_surf=xc_surf * chord,
         Cp_upper=Cp_upper, Cp_lower=Cp_lower,
         )

print(f"  Saved: {outfile}")
print(f"  Grid shape: {X.shape}")


# ============================================================================
# 7. Visualization
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'NACA 0012 — Re={int(Re)}, AoA={AoA_deg}°, Cd={Cd:.4f}, Cl={Cl:.4f}',
             fontsize=15, fontweight='bold')

U_m = np.ma.masked_invalid(U_field)
V_m = np.ma.masked_invalid(V_field)
P_m = np.ma.masked_invalid(P_field)
speed = np.ma.sqrt(U_m**2 + V_m**2)

# Airfoil patch for plotting
af_patch_xy = np.column_stack([x_af, y_af])

xlim = (-0.5, 3.0)
ylim = (-1.0, 1.0)

# (a) u-velocity
ax = axes[0, 0]
c = ax.contourf(X, Y, U_m, levels=np.linspace(-0.3, 1.3, 50), cmap='RdBu_r', extend='both')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('(a) $u$-velocity'); ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')

# (b) v-velocity
ax = axes[0, 1]
c = ax.contourf(X, Y, V_m, levels=np.linspace(-0.5, 0.5, 50), cmap='RdBu_r', extend='both')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('(b) $v$-velocity'); ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')

# (c) Pressure
ax = axes[1, 0]
c = ax.contourf(X, Y, P_m, levels=50, cmap='coolwarm', extend='both')
plt.colorbar(c, ax=ax, shrink=0.85)
ax.fill(x_af, y_af, color='0.4', zorder=5)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_title('(c) Pressure'); ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')

# (d) Surface Cp
ax = axes[1, 1]
valid_u = ~np.isnan(Cp_upper)
valid_l = ~np.isnan(Cp_lower)
ax.plot(xc_surf[valid_u], Cp_upper[valid_u], 'b-', lw=2, label='Upper surface')
ax.plot(xc_surf[valid_l], Cp_lower[valid_l], 'r--', lw=2, label='Lower surface')
ax.set_xlabel('$x/c$'); ax.set_ylabel('$C_p$')
ax.set_title('(d) Surface pressure coefficient')
ax.invert_yaxis()
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('fig_naca0012_reference.png', dpi=150)
plt.savefig('fig_naca0012_reference.pdf', dpi=300)
print("  Saved: fig_naca0012_reference.png/.pdf")

# Streamlines
fig, ax = plt.subplots(figsize=(14, 6))
c = ax.contourf(X, Y, speed, levels=np.linspace(0, 1.4, 50), cmap='viridis')
plt.colorbar(c, ax=ax, shrink=0.85, label='$|\\mathbf{u}|/U_\\infty$')
ax.streamplot(x_grid, y_grid, U_m.filled(0), V_m.filled(0),
              color='white', linewidth=0.6, density=3, arrowsize=0.7)
ax.fill(x_af, y_af, color='0.4', zorder=5, edgecolor='white', lw=1)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
ax.set_xlabel('$x/c$'); ax.set_ylabel('$y/c$')
ax.set_title(f'NACA 0012 — Re={int(Re)}, AoA={AoA_deg}° — Streamlines',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_naca0012_streamlines.png', dpi=150)
plt.savefig('fig_naca0012_streamlines.pdf', dpi=300)
print("  Saved: fig_naca0012_streamlines.png/.pdf")


# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print(f"  NACA 0012 REFERENCE SOLUTION SUMMARY")
print(f"{'='*60}")
print(f"  Re = {Re}, AoA = {AoA_deg}°")
print(f"  Cd = {Cd:.6f}, Cl = {Cl:.6f}, L/D = {Cl/Cd:.3f}")
print(f"  Mesh: {mesh.num_cells()} cells")
print(f"  Grid: {nx} x {ny} = {nx*ny} points")
print(f"  Extraction: [{x_ext_min}, {x_ext_max}] x [{y_ext_min}, {y_ext_max}]")
print(f"  Valid flow points: {n_evaluated}")
print(f"  Saved: {outfile}")
print(f"{'='*60}")
