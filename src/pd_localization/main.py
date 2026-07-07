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
    curv_detection,
    zero_virtual,
)
from .localizacao import batch_localization
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

    results_dtw = batch_localization(
        experiments=experiments,
        pre_processing=cum_energy_exp,
        ref_estimator=deriv_estimation,
        distance=dist,
        radius=24,
        mode="2d",
    )

    results_gcc = batch_localization(
        experiments=experiments,
        pre_processing=cum_energy_exp,
        ref_estimator=curv_detection,
        distance=dist,
        radius=24,
        mode="2d",
        dtw=False,
        gcc_func=gcc_roth,
    )

    df_dtw = to_dataframe(results_dtw)
    df_clean = remove_outliers(df_dtw)
    print("-" * 100)
    print("\t metodo da derivada")
    print("-" * 100)
    print(mean_estimates(df_clean))

    df_gcc = to_dataframe(results_gcc)
    df_clean_gcc = remove_outliers(df_gcc)
    print("-" * 100)
    print("GCC")
    print("-" * 100)
    print(mean_estimates(df_clean_gcc))

    # plot_location_scatter(results)
    # plt.show()


if __name__ == "__main__":
    main()
