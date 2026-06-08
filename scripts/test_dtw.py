import matplotlib.pyplot as plt

from pd_localization.experiment_loader import load_experiments, get_experiment
from pd_localization.dtw import dist, plot_dtw
from pd_localization.preprocessing import norm_zero_mean

DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"


def main():
    experiments = load_experiments(DATA_PATH)
    exp = get_experiment(experiments, "antena1", 1)
    voltages = norm_zero_mean(exp)

    v1 = voltages[0]
    v2 = voltages[1]
    plot_dtw(v1, v2[::-1], dist, radius=24)
    plt.show()


if __name__ == "__main__":
    main()
