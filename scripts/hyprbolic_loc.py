import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from pd_localization.experiment_loader import (
    load_experiments,
    get_experiment,
)
from pd_localization.localizacao import (
    batch_localization,
    batch_tau_estimates,
    single_tau_estimate,
    tau_estim_sff,
    true_taus,
)
from pd_localization.preprocessing import norm_zero_mean
from pd_localization.gcc import sff_gcc
import numpy as np
import warnings

import numpy as np

v_e = 299792458.0


def tdoa_2d(vars, taus: dict) -> np.ndarray:
    x, y, t1 = vars
    new_taus = {ch: tau for ch, tau in taus.items() if ch != "CH4"}

    eqs = []
    for ch, tau in new_taus.items():
        xi, yi = ANTENNA_POSITIONS[ch][:2]
        eq = (x - xi) ** 2 + (y - yi) ** 2 - (v_e * (t1 + tau)) ** 2
        eqs.append(eq)
    return np.array(eqs)


def jacobian_2d(vars, taus: dict) -> np.ndarray:
    x, y, t1 = vars
    new_taus = {ch: tau for ch, tau in taus.items() if ch != "CH4"}

    J = []
    for ch, tau in new_taus.items():
        xi, yi = ANTENNA_POSITIONS[ch][:2]
        df_dx = 2 * (x - xi)
        df_dy = 2 * (y - yi)
        df_dt1 = -2 * (v_e**2) * (t1 + tau)
        J.append([df_dx, df_dy, df_dt1])
    return np.array(J)


def tdoa_3d(vars, taus: dict) -> np.ndarray:
    x, y, z, t1 = vars

    eqs = []
    for ch, tau in taus.items():
        xi, yi, zi = ANTENNA_POSITIONS[ch][:3]
        eq = (x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2 - (v_e * (t1 + tau)) ** 2
        eqs.append(eq)
    return np.array(eqs)


def jacobian_3d(vars, taus: dict) -> np.ndarray:
    x, y, z, t1 = vars

    J = []
    for ch, tau in taus.items():
        xi, yi, zi = ANTENNA_POSITIONS[ch][:3]
        df_dx = 2 * (x - xi)
        df_dy = 2 * (y - yi)
        df_dz = 2 * (z - zi)
        df_dt1 = -2 * (v_e**2) * (t1 + tau)
        J.append([df_dx, df_dy, df_dz, df_dt1])
    return np.array(J)


def solve_newton_raphson(
    func, jacobian_func, initial_guess, taus, max_iter=50, tol=1e-9
):
    """
    Executes the Newton-Raphson iterations to find the source location.

    ---------------------------------------------------------------------------
    ```
    # Example Usage:
    guess_2d = [0.0, 0.0, 0.0]  # [x, y, t1]
    result_2d = solve_newton_raphson(tdoa_2d, jacobian_2d, guess_2d, sff_taus)

    guess_3d = [0.0, 0.0, 0.0, 0.0]  # [x, y, z, t1]
    result_3d = solve_newton_raphson(tdoa_3d, jacobian_3d, guess_3d, sff_taus)
    ```
    """
    v = np.array(initial_guess, dtype=float)

    for i in range(max_iter):
        F = func(v, taus)
        J = jacobian_func(v, taus)

        # Check convergence based on the residual norm
        if np.linalg.norm(F) < tol:
            print(f"Converged in {i} iterations.")
            return v

        # Update step: v_{k+1} = v_k - J^{-1} * F
        # np.linalg.solve is used for square matrices (determined systems)
        try:
            delta_v = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if the Jacobian becomes singular
            delta_v = np.linalg.pinv(J) @ F

        v = v - delta_v

        # Optional: Add a step constraint to prevent divergent jumps in early iterations
        # if np.linalg.norm(delta_v) < tol:
        #     break

    warnings.warn("Warning: Maximum iterations reached without convergence.")
    return v


def mean_taus(taus_list: list[dict]) -> dict:
    return {ch: np.mean([tau[ch] for tau in taus_list]) for ch in taus_list[0].keys()}


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
experiments = load_experiments(DATA_PATH, ["antena1"])
MODE = "2d"
# ANTENNA_POSITIONS = {
#     "CH1": np.array([0.0, 0.0, 0.97]),
#     "CH2": np.array([2.0, 0.0, 0.77]),
#     "CH3": np.array([2.0, 2.0, 0.57]),
#     "CH4": np.array([0.0, 2.0, 0.47]),
# }
ANTENNA_POSITIONS = {
    "CH1": np.array([0.0, 0.0]),
    "CH2": np.array([2.0, 0.0]),
    "CH3": np.array([2.0, 2.0]),
    "CH4": np.array([0.0, 2.0]),
}
true_sff_taus = true_taus(experiments)
ic(true_sff_taus[0])


sff_estimator = tau_estim_sff(norm_zero_mean, sff_gcc)
sff_taus = batch_tau_estimates(experiments, sff_estimator)
# ic([d["CH3"] for d in sff_taus])
est_taus = mean_taus(sff_taus)
ic(est_taus)

x, y, t1 = solve_newton_raphson(tdoa_2d, jacobian_2d, np.zeros(3), est_taus)
print(f"Estimated Position Newton-Raphson: x = {x:.4f}, y = {y:.4f}")
