# Turek-Hron FSI1: Steady-state benchmark (Re=20)
# Reference: Turek and Hron (2006), Table 12-13
#
# Expected results:
#   u_x(A) = 0.0227e-3,  u_y(A) = 0.8209e-3
#   Drag   = 14.295,      Lift   = 0.7638

from turtleFSI.problems.TF_fsi import (
    set_problem_parameters as _base_params,
    get_mesh_domain_and_boundaries,
    initiate,
    create_bcs,
    pre_solve,
    post_solve,
)


def set_problem_parameters(default_variables, **namespace):
    default_variables = _base_params(default_variables, **namespace)
    default_variables.update(dict(
        Um=0.2,
        rho_f=1.0e3,
        mu_f=1.0,
        rho_s=1.0e4,
        nu_s=0.4,
        mu_s=0.5e6,
        lambda_s=2.0e6,
        T=15.0,
        dt=0.01,
        folder="turek_hron_fsi1_results",
        save_step=50,
        checkpoint_step=500,
    ))
    return default_variables
