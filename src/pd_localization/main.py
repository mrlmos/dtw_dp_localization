import numpy as np
import matplotlib.pyplot as plt
from .experiment_loader import (
    load_experiments,
    Experiment,
)
from .preprocessing import (
    cum_energy_exp,
    deriv_estimation,
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

    results = batch_localization(
        experiments=experiments,
        pre_processing=cum_energy_exp,
        ref_estimator=deriv_estimation,
        distance=dist,
        radius=24,
        mode="2d",
    )

    df = to_dataframe(results)
    df_clean = remove_outliers(df)
    print_summary(df_clean)
    print(mean_estimates(df_clean))

    # plot_location_scatter(results)
    # plt.show()


if __name__ == "__main__":
    main()
