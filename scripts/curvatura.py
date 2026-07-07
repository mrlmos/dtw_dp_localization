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
    curv_detection,
)


def filtro_deriv(x: np.ndarray):
    return np.convolve(x, [1, -1], mode="same")


def filtro_2deriv(x: np.ndarray):
    return np.convolve(x, [1, -2, 1], mode="same")


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"


def main():
    experiments = load_experiments(DATA_PATH)
    exp = get_experiment(experiments, "antena4", 1)
    voltages = cum_energy_exp(exp)
    v = voltages[3]
    dv = filtro_deriv(v)
    ddv = filtro_2deriv(v)
    ddv[-1] = 0.0

    kf = ddv / ((1 + dv**2) ** (3 / 2))
    kf /= np.max(kf)
    idx = np.argmax(kf >= 0.1)

    plt.plot(v)
    plt.scatter(idx, v[idx], facecolors="none", edgecolors="r", s=100)
    plt.show()


if __name__ == "__main__":
    main()
