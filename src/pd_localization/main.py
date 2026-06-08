import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from .experiment_loader import (
    get_experiment,
    load_experiments,
    Experiment,
)
from .preprocessing import (
    cum_energy_exp,
    deriv_estimation,
    pole_finding,
    relu_energy,
    energy_criterion,
    filtro_derivada,
    akaike_info,
    pole_finding,
)
from .localizacao import batch_localization, tdoa_2d
from .results import (
    remove_outliers,
    to_dataframe,
    export_xlsx,
    print_summary,
    plot_location_scatter,
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
    # exp = get_experiment(experiments, antenna="antena1", index=2)
    # voltages = akaike_info(exp)
    # fig, axs = plt.subplots(4, 1)
    # for ax, v in zip(axs, voltages):
    #     ax.plot(v)
    #     ax.axvline(np.argmin(v), c="red", label=f"{np.argmin(v)}")
    #     ax.legend()
    # plt.show()

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
    # export_xlsx(df, path=OUTPUT_PATH + "results_diff_sem_outliers.xlsx")
    print(mean_estimates(df_clean))

    # plot_location_scatter(results)
    # plt.show()


if __name__ == "__main__":
    main()
