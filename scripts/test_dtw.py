import matplotlib.pyplot as plt
import numpy as np

from pd_localization.experiment_loader import load_experiments, get_experiment
from pd_localization.dtw import dist, plot_dtw
from pd_localization.preprocessing import (
    norm_zero_mean,
    cum_energy_exp,
    deriv_estimation,
)
from pd_localization.gcc import gcc_phat, gcc_ht

DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"


def main():
    experiments = load_experiments(DATA_PATH)
    exp = get_experiment(experiments, "antena1", 2)
    voltages = norm_zero_mean(exp)
    comp = 3
    exp.plot()
    plt.show()

    v1 = voltages[0]
    v2 = voltages[comp]
    v_comp = np.array(cum_energy_exp(exp)[comp])

    cc = gcc_phat(v1, v2)
    idx = np.abs(np.argmax(cc) - len(v1) + 1)
    print("estimated delta: ", idx)
    plt.subplot(211)
    plt.plot(cc)
    plt.subplot(212)
    plt.plot(v_comp)
    plt.axvline(deriv_estimation(np.array(cum_energy_exp(exp)[0]), 0) + idx)
    plt.show()


if __name__ == "__main__":
    main()
