import numpy as np
import matplotlib.pyplot as plt
from .gcc import gcc_ht, gcc_roth, gcc_phat, gcc_scot
from .experiment_loader import (
    load_experiments,
    Experiment,
)
from .preprocessing import (
    cum_energy_exp,
    deriv_estimation,
    norm_zero_mean,
    curv_detection,
    zero_virtual,
    akaike_info,
    energy_criterion,
)
from .localizacao import (
    batch_localization,
    batch_tau_estimates,
    tau_estim_dtw,
    tau_estim_gcc,
    tau_estim_energy,
)
from .results import (
    remove_outliers,
    to_dataframe,
    print_summary,
    mean_estimates,
)
from .dtw import dist


def check_ref(exp: Experiment, ref_index: int):
    voltages = cum_energy_exp(exp)
    ref = deriv_estimation(voltages[ref_index])
    plt.plot(voltages[ref_index])
    plt.axvline(ref, c="red")


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
OUTPUT_PATH = "/home/murilo/dev/python/tcc/dtw/result_sheets/"


def main():
    experiments = load_experiments(base_dir=DATA_PATH, antennas=None)

    # ---------- DTW ----------
    dtw_estimator = tau_estim_dtw(cum_energy_exp, curv_detection, dist)
    dtw_taus = batch_tau_estimates(experiments, dtw_estimator)
    results_dtw = batch_localization(experiments, dtw_taus)

    df_dtw = to_dataframe(results_dtw)
    df_clean = remove_outliers(df_dtw)
    print("-" * 100)
    print("DTW + curvature detection")
    print("-" * 100)
    print(mean_estimates(df_clean))

    # ---------- GCC ----------
    gcc_estimator = tau_estim_gcc(norm_zero_mean, gcc_roth)
    gcc_taus = batch_tau_estimates(experiments, gcc_estimator)
    results_gcc = batch_localization(experiments, gcc_taus)

    df_gcc = to_dataframe(results_gcc)
    df_clean_gcc = remove_outliers(df_gcc)
    print("-" * 100)
    print("GCC")
    print("-" * 100)
    print(mean_estimates(df_clean_gcc))

    # ---------- AKAIKE ----------
    akaike_estimator = tau_estim_energy(akaike_info, np.min)
    akaike_taus = batch_tau_estimates(experiments, akaike_estimator)
    results_akaike = batch_localization(experiments, akaike_taus)

    df_akaike = to_dataframe(results_akaike)
    df_clean_akaike = remove_outliers(df_akaike)
    print("-" * 100)
    print("AKAIKE")
    print("-" * 100)
    print(mean_estimates(df_clean_akaike))

    # ---------- ENERGY CRITERION ----------
    energy_estimator = tau_estim_energy(energy_criterion, np.min)
    energy_taus = batch_tau_estimates(experiments, energy_estimator)
    results_energy = batch_localization(experiments, energy_taus)

    df_energy = to_dataframe(results_energy)
    df_clean_energy = remove_outliers(df_energy)
    print("-" * 100)
    print("ENERGY CRITERION")
    print("-" * 100)
    print(mean_estimates(df_clean_energy))

    # ---------- CUMULATIVE ENERGY ----------
    cum_estimator = tau_estim_energy(cum_energy_exp, curv_detection)
    cum_taus = batch_tau_estimates(experiments, cum_estimator)
    results_cum = batch_localization(experiments, cum_taus)

    df_cum = to_dataframe(results_cum)
    df_clean_cum = remove_outliers(df_cum)
    print("-" * 100)
    print("CUM")
    print("-" * 100)
    print(mean_estimates(df_clean_cum))


if __name__ == "__main__":
    main()
