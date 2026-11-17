import numpy as np
import pandas as pd
import time

from .experiment_loader import Experiment
from .dtw import gen_cost_matrix, backtracking
from .gcc import gcc_phat, sff_gcc
from .preprocessing import (
    cum_energy_exp,
    norm_zero_mean,
    curv_detection,
    energy_criterion,
)
from .results import mean_estimates, to_dataframe
from .localizacao import (
    tau_estim_dtw,
    tau_estim_energy,
    tau_estim_gcc,
    tau_estim_sff,
    batch_localization,
    batch_tau_estimates,
    dist,
)


def mean_loc_dtw(
    experiments: list[Experiment], mode: str = "3d", radius=13
) -> pd.DataFrame:
    start = time.time()
    dtw_estimator = tau_estim_dtw(cum_energy_exp, curv_detection, dist, radius=radius)
    dtw_taus = batch_tau_estimates(experiments, dtw_estimator)
    dtw_results = batch_localization(experiments, dtw_taus, mode=mode)
    end = time.time()
    print(f"DTW concluido. Tempo: {end - start}s")
    return mean_estimates(to_dataframe(dtw_results))


def mean_loc_gcc_phat(experiments: list[Experiment], mode: str = "3d") -> pd.DataFrame:
    start = time.time()
    phat_estimator = tau_estim_gcc(norm_zero_mean, gcc_phat)
    phat_taus = batch_tau_estimates(experiments, phat_estimator)
    phat_results = batch_localization(experiments, phat_taus, mode=mode)
    end = time.time()
    print(f"GCC-PHAT concluido. Tempo: {end - start}s")
    return mean_estimates(to_dataframe(phat_results))


def mean_loc_gcc_sff(experiments: list[Experiment], mode: str = "3d") -> pd.DataFrame:
    start = time.time()
    sff_estimator = tau_estim_sff(norm_zero_mean, sff_gcc)
    sff_taus = batch_tau_estimates(experiments, sff_estimator)
    sff_results = to_dataframe(batch_localization(experiments, sff_taus, mode=mode))
    end = time.time()
    print(f"GCC-SFF concluido. Tempo: {end - start}s")
    return mean_estimates(sff_results)


def mean_loc_cum(experiments: list[Experiment], mode: str = "3d") -> pd.DataFrame:
    start = time.time()
    energy_estimator = tau_estim_energy(energy_criterion, np.min)
    energy_taus = batch_tau_estimates(experiments, energy_estimator)
    energy_results = batch_localization(experiments, energy_taus)
    end = time.time()
    print(f"ENERGIA CUMULATIVA concluido. Tempo: {end - start}s")
    return mean_estimates(to_dataframe(energy_results))
