import time
import numpy as np
import pandas as pd
from pd_localization.experiment_loader import load_experiments, SOURCE_POSITIONS
from pd_localization.preprocessing import cum_energy_exp, curv_detection
from pd_localization.localizacao import (
    batch_tau_estimates,
    tau_estim_dtw,
    tdoa_3d,
    jacobian_3d,
    expand_taus,
    v_e,
    LocalizationResult,
)
from pd_localization.results import mean_estimates, to_dataframe
from pd_localization.dtw import dist

print("Loading experiments...")
experiments = load_experiments("/home/murilo/dev/python/tcc/dados_dp/")
# Use radius=14 as in scripts/loc.py
dtw_estimator = tau_estim_dtw(cum_energy_exp, curv_detection, dist, radius=14)
print("Estimating taus (radius=14)...")
dtw_taus = batch_tau_estimates(experiments, dtw_estimator)
t_mid = 4.71e-9


def solve_unconstrained(taus, lam=0.01, max_iter=25, tol=1e-5):
    v = np.array([0.0, 0.0, 0.5, t_mid], dtype=float)
    dim = len(v)
    scale = np.ones(dim)
    scale[-1] = 1.0 / v_e

    for i in range(max_iter):
        F = np.array(tdoa_3d(v, taus))
        J = np.array(jacobian_3d(v, taus))

        if np.linalg.norm(F) < tol:
            return v, True

        J_scaled = J * scale[None, :]
        JTJ = J_scaled.T @ J_scaled
        JTF = J_scaled.T @ F
        diag_JTJ = np.diag(np.diag(JTJ))
        reg_matrix = JTJ + lam * diag_JTJ

        try:
            delta_v_scaled = np.linalg.solve(reg_matrix, JTF)
        except np.linalg.LinAlgError:
            delta_v_scaled = np.linalg.pinv(reg_matrix) @ JTF

        delta_v = delta_v_scaled * scale
        if np.linalg.norm(delta_v_scaled[: dim - 1]) < 1e-4:
            return v, True

        v = v - delta_v
    return v, False


def run_option1(exps, taus_list):
    results = []
    for exp, tau_dict in zip(exps, taus_list):
        variants = expand_taus(tau_dict, 0.4e-9)
        sols = []
        for tau in variants:
            sol, ier = solve_unconstrained(tau)
            if ier:
                sols.append(sol)

        if len(sols) == 0:
            sol_final = np.full(4, np.nan)
            conv = False
        else:
            sols = np.array(sols)
            # Step 1: take mean of all converged values (even negative / outside)
            sol_mean = np.mean(sols, axis=0)
            in_area = (
                0.0 <= sol_mean[0] <= 2.0
                and 0.0 <= sol_mean[1] <= 2.0
                and 0.0 <= sol_mean[2] <= 1.0
            )
            if in_area:
                sol_final = sol_mean
                conv = True
            else:
                # Retry taking only converged positive values
                pos_mask = np.all(sols >= 0.0, axis=1)
                pos_sols = sols[pos_mask]
                if len(pos_sols) > 0:
                    sol_final = np.mean(pos_sols, axis=0)
                    conv = True
                else:
                    sol_final = sol_mean
                    conv = False

        results.append(
            LocalizationResult(
                exp_antenna=exp.antenna,
                exp_index=exp.index,
                taus=tau_dict,
                mode="3d",
                position_estimated=sol_final[:-1],
                t1_estimated=sol_final[-1],
                position_true=SOURCE_POSITIONS[exp.antenna],
                converged=conv,
                taus_estimated_samples={},
                taus_true_samples={},
            )
        )
    return results


print("\n=== OPTION 1: Unconstrained mean, retry with positive only if outside ===")
res1 = run_option1(experiments, dtw_taus)
df1 = mean_estimates(to_dataframe(res1))
print(df1[["exp_antena", "x_est", "y_est", "z_est", "abs_error(m)", "rel_error(%)"]])
