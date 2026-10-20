import numpy as np
from icecream import ic
from itertools import product
from pd_localization.experiment_loader import load_experiments, get_experiment
from pd_localization.preprocessing import norm_zero_mean
from pd_localization.gcc import sff_gcc
from pd_localization.localizacao import (
    tau_estim_sff,
    single_tau_estimate,
    tdoa_2d,
    jacobian_2d,
    solve_newton_raphson,
    SOURCE_POSITIONS,
)


experiments = load_experiments()
exp = get_experiment(experiments, "antena1", 4)
sff_estimator = tau_estim_sff(norm_zero_mean, sff_gcc)
sff_taus = single_tau_estimate(exp, sff_estimator)
ic(sff_taus)

error = 0.4e-9


def expand_taus(taus, error):
    keys = list(taus.keys())
    options = [[v - error, v, v + error] if v != 0 else [v] for v in taus.values()]
    return [dict(zip(keys, combo)) for combo in product(*options)]


variants = expand_taus(sff_taus, error)
solutions = np.zeros((len(variants), 3))
t_mid = 4.71e-9  # Como na tese de George
for i, taus in enumerate(variants):
    p0_2d = [1.0, 1.0, t_mid]
    sol, ier = solve_newton_raphson(exp, tdoa_2d, jacobian_2d, p0_2d, taus)
    solutions[i, :] = sol

sol = np.mean(solutions, axis=0)
true_pos = SOURCE_POSITIONS[exp.antenna][:-1]
ic(sol)

# ----------------
variants = [sff_taus]
solutions = np.zeros((len(variants), 3))
t_mid = 4.71e-9  # Como na tese de George
for i, taus in enumerate(variants):
    p0_2d = [1.0, 1.0, t_mid]
    sol, ier = solve_newton_raphson(exp, tdoa_2d, jacobian_2d, p0_2d, taus)
    solutions[i, :] = sol

sol = np.mean(solutions, axis=0)
true_pos = SOURCE_POSITIONS[exp.antenna][:-1]
ic(sol, SOURCE_POSITIONS)
