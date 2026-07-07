import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pd_localization.experiment_loader import (
    load_experiments,
    get_experiment,
    Experiment,
)
from pd_localization.preprocessing import (
    cum_energy_exp,
    deriv_estimation,
)
from pd_localization.localizacao import batch_localization
from pd_localization.results import (
    remove_outliers,
    to_dataframe,
    print_summary,
    mean_estimates,
)
from pd_localization.dtw import dist


def get_sample(experimentos: list[Experiment]) -> np.ndarray:
    exp = get_experiment(experimentos, "antena1", 0)
    voltages = cum_energy_exp(exp)
    return voltages[0]


def dummy_sample(x: np.ndarray, offset: int) -> int:
    return offset


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"

experimentos = load_experiments(DATA_PATH)
exp = get_experiment(experimentos, "antena1", 0)
x_size = len(exp.voltage("CH1"))
indexes = range(20, x_size - 10)
errors = np.zeros(x_size)
for k in indexes:
    print("Iter: ", k)
    results = batch_localization(
        experiments=experimentos,
        pre_processing=cum_energy_exp,
        ref_estimator=dummy_sample,
        distance=dist,
        radius=24,
        mode="2d",
        offset=k,
    )
    means = mean_estimates(remove_outliers(to_dataframe(results)))
    mean_dist = means["abs_error(m)"].mean()
    errors[k] = mean_dist

plt.plot(get_sample(experimentos), label="reference signal", c="r")
plt.plot(errors / np.max(errors), label="Normalized errors")
plt.axvline(np.argmax(errors), c="k")
plt.legend()
plt.show()
