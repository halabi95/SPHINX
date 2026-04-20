"""
==============================================================================
NACA 0012 — Re=10,000, AoA=5° — Proper Mesh with Gmsh + SU2 Config
==============================================================================
Uses gmsh for high-quality unstructured mesh with boundary layer refinement.
Fixes from previous attempt:
  1. Proper mesh quality (gmsh handles LE/TE clustering properly)
  2. AoA applied via velocity components (INC_VELOCITY_INIT)
  3. Lower CFL with ramping for stability
  4. Proper boundary layer mesh (structured quads near wall)
==============================================================================
"""

import numpy as np
import gmsh
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
rho = 1.0

# Velocity components for AoA
U_x = U_inf * np.cos(AoA_rad)
U_y = U_inf * np.sin(AoA_rad)

print("="*60)
print("  NACA 0012 Mesh Generation (Gmsh) + SU2 Config")
print(f"  Re={Re}, AoA={AoA_deg}°")
print(f"  U_x={U_x:.6f}, U_y={U_y:.6f}")
print(f"  nu={nu:.6f}, mu={rho*nu:.6f}")
print("="*60)

# ============================================================================
# STEP 1: GENERATE MESH WITH GMSH
# ============================================================================
print("\nGenerating mesh with Gmsh...")

# NACA 0012 coordinates
def naca0012(x):
    t = 0.12
    return 5.0 * t * (
        0.2969 * np.sqrt(x) - 0.1260 * x
        - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )

# Cosine spacing for airfoil points (more points near LE and TE)
n_af = 150
beta = np.linspace(0, np.pi, n_af)
xc = 0.5 * (1 - np.cos(beta))  # 0 to 1
yt = naca0012(xc)

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)
gmsh.model.add("naca0012")

# --- Add airfoil points ---
# Upper surface: TE → LE (x: 1 → 0)
upper_pts = []
for i in range(n_af):
    idx = n_af - 1 - i  # reverse: TE to LE
    p = gmsh.model.geo.addPoint(xc[idx] * chord, yt[idx] * chord, 0)
    upper_pts.append(p)

# Lower surface: LE → TE (x: 0 → 1), skip LE (already added)
lower_pts = []
for i in range(1, n_af):
    p = gmsh.model.geo.addPoint(xc[i] * chord, -yt[i] * chord, 0)
    lower_pts.append(p)

# Create splines for upper and lower surfaces
upper_spline = gmsh.model.geo.addSpline(upper_pts)
lower_spline = gmsh.model.geo.addSpline([upper_pts[-1]] + lower_pts)
te_line = gmsh.model.geo.addLine(lower_pts[-1], upper_pts[0])  # close at TE

# Airfoil loop
airfoil_loop = gmsh.model.geo.addCurveLoop([upper_spline, lower_spline, te_line])

# --- Far-field circle ---
R_far = 30.0 * chord
center = gmsh.model.geo.addPoint(0.5 * chord, 0, 0)
p_right = gmsh.model.geo.addPoint(0.5 * chord + R_far, 0, 0)
p_top = gmsh.model.geo.addPoint(0.5 * chord, R_far, 0)
p_left = gmsh.model.geo.addPoint(0.5 * chord - R_far, 0, 0)
p_bottom = gmsh.model.geo.addPoint(0.5 * chord, -R_far, 0)

arc1 = gmsh.model.geo.addCircleArc(p_right, center, p_top)
arc2 = gmsh.model.geo.addCircleArc(p_top, center, p_left)
arc3 = gmsh.model.geo.addCircleArc(p_left, center, p_bottom)
arc4 = gmsh.model.geo.addCircleArc(p_bottom, center, p_right)

farfield_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])

# --- Domain = farfield minus airfoil ---
surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])

gmsh.model.geo.synchronize()

# --- Mesh size control ---
# Fine mesh near airfoil
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [upper_spline, lower_spline, te_line])
gmsh.model.mesh.field.setNumber(1, "Sampling", 300)

# Threshold field: fine near airfoil, coarse far away
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.005)   # near airfoil
gmsh.model.mesh.field.setNumber(2, "SizeMax", 2.0)     # far-field
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.01)    # start of transition
gmsh.model.mesh.field.setNumber(2, "DistMax", 5.0)     # end of transition

# Wake refinement
gmsh.model.mesh.field.add("Box", 3)
gmsh.model.mesh.field.setNumber(3, "VIn", 0.02)
gmsh.model.mesh.field.setNumber(3, "VOut", 2.0)
gmsh.model.mesh.field.setNumber(3, "XMin", 0.9)
gmsh.model.mesh.field.setNumber(3, "XMax", 5.0)
gmsh.model.mesh.field.setNumber(3, "YMin", -0.5)
gmsh.model.mesh.field.setNumber(3, "YMax", 0.5)

# Take minimum of all fields
gmsh.model.mesh.field.add("Min", 4)
gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
gmsh.model.mesh.field.setAsBackgroundMesh(4)

# Boundary layer mesh (structured quads near airfoil)
gmsh.model.mesh.field.add("BoundaryLayer", 5)
gmsh.model.mesh.field.setNumbers(5, "CurvesList", [upper_spline, lower_spline])
gmsh.model.mesh.field.setNumber(5, "Size", 0.002)       # first layer height
gmsh.model.mesh.field.setNumber(5, "Ratio", 1.2)        # growth ratio
gmsh.model.mesh.field.setNumber(5, "NbLayers", 15)      # number of layers
gmsh.model.mesh.field.setNumber(5, "Quads", 1)          # quad elements
gmsh.model.mesh.field.setAsBoundaryLayer(5)

# --- Physical groups (for SU2 boundary markers) ---
gmsh.model.addPhysicalGroup(1, [upper_spline, lower_spline, te_line], tag=1)
gmsh.model.setPhysicalName(1, 1, "airfoil")

gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], tag=2)
gmsh.model.setPhysicalName(1, 2, "farfield")

gmsh.model.addPhysicalGroup(2, [surface], tag=3)
gmsh.model.setPhysicalName(2, 3, "fluid")

# --- Generate mesh ---
gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
gmsh.option.setNumber("Mesh.RecombineAll", 0)  # triangles (more robust)
gmsh.model.mesh.generate(2)

# Get mesh stats
node_tags, _, _ = gmsh.model.mesh.getNodes()
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
n_nodes = len(node_tags)
n_elems = sum(len(t) for t in elem_tags)
print(f"  Nodes: {n_nodes}, Elements: {n_elems}")

# --- Export as SU2 ---
mesh_file = "naca0012_re10k.su2"
gmsh.write(mesh_file)
gmsh.finalize()

print(f"  Saved: {mesh_file} ({os.path.getsize(mesh_file)/1e6:.1f} MB)")


# ============================================================================
# STEP 2: WRITE SU2 CONFIG
# ============================================================================
print("\nWriting SU2 config...")

config_file = "naca0012_re10k.cfg"
config_text = f"""%
% NACA 0012 — Re=10,000, AoA=5°, Incompressible RANS, k-ω SST
%

% ------------- PROBLEM DEFINITION ------------------------------------------
SOLVER= INC_RANS
KIND_TURB_MODEL= SST
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% ------------- INCOMPRESSIBLE FLOW -----------------------------------------
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {rho}
INC_VELOCITY_INIT= ( {U_x:.8f}, {U_y:.8f}, 0.0 )
INC_TEMPERATURE_INIT= 300.0
INC_NONDIM= DIMENSIONAL

% ------------- VISCOSITY ---------------------------------------------------
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {rho * nu:.8f}

% ------------- FREE-STREAM TURBULENCE --------------------------------------
FREESTREAM_TURBULENCEINTENSITY= 0.01
FREESTREAM_TURB2LAMVISCRATIO= 1.0

% ------------- REFERENCE VALUES --------------------------------------------
REYNOLDS_NUMBER= {Re}
REYNOLDS_LENGTH= {chord}
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= {chord}
REF_AREA= {chord}

% ------------- BOUNDARY CONDITIONS -----------------------------------------
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )

% ------------- SURFACE MONITORING ------------------------------------------
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

% ------------- NUMERICAL METHOD - GRADIENT ---------------------------------
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES

% ------------- CFL AND TIME STEPPING --------------------------------------
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 5.0, 100.0 )

% ------------- LINEAR SOLVER -----------------------------------------------
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 15

% ------------- FLOW NUMERICS -----------------------------------------------
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.1
TIME_DISCRE_FLOW= EULER_IMPLICIT

% ------------- TURBULENCE NUMERICS -----------------------------------------
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT

% ------------- CONVERGENCE -------------------------------------------------
ITER= 15000
CONV_RESIDUAL_MINVAL= -8
CONV_FIELD= RMS_PRESSURE

% ------------- INPUT/OUTPUT ------------------------------------------------
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2

SOLUTION_FILENAME= restart_flow.dat
RESTART_FILENAME= restart_flow.dat

TABULAR_FORMAT= CSV
CONV_FILENAME= history

VOLUME_FILENAME= flow
VOLUME_OUTPUT= (COORDINATES, SOLUTION, PRIMITIVE)
SURFACE_FILENAME= surface_flow

OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
OUTPUT_WRT_FREQ= 500

SCREEN_OUTPUT= (INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_VELOCITY-Y, LIFT, DRAG)
SCREEN_WRT_FREQ_INNER= 100
"""

with open(config_file, 'w') as f:
    f.write(config_text)

print(f"  Saved: {config_file}")


# ============================================================================
# DONE
# ============================================================================
print(f"""
{'='*60}
  SETUP COMPLETE
{'='*60}
  Mesh: {mesh_file} ({n_nodes} nodes, {n_elems} elements)
  Config: {config_file}

  Run SU2:
    SU2_CFD {config_file}

  Expected output:
    - history.csv
    - flow.vtu (volume solution)
    - surface_flow.vtu (surface data)

  Estimated time: 5-15 min
{'='*60}
""")
