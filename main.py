from icecream import ic
from experiment_loader import Experiment, load_experiments, get_experiment
from dtw import dtw, plot_dtw
import matplotlib.pyplot as plt
import numpy as np


def dist(x, y):
    return (x - y) ** 2


def cum_energy_exp(exp: Experiment):
    chs = exp.ch_names
    energies = []
    for ch in chs:
        cum_energy = np.cumsum(exp.voltage(ch) ** 2)
        energies.append(cum_energy / cum_energy.max())
    return energies


def norm_exp_same(exp: Experiment):
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        ch_mean = np.mean(exp.voltage(ch))
        ch_std = np.std(exp.voltage(ch))
        ic(ch_mean, ch_std)
        voltages.append((exp.voltage(ch) - ch_mean) / ch_std)
    return voltages


def norm_exp(exp: Experiment):
    chs = exp.ch_names
    total_max = 0
    total_min = 0
    for ch in chs:
        channel_max = np.max(exp.voltage(ch))
        channel_min = np.min(exp.voltage(ch))
        if channel_max > total_max:
            total_max = channel_max.copy()
        if channel_min < total_min:
            total_min = channel_min.copy()
    return [(exp.voltage(ch) - total_min) / (total_max - total_min) for ch in chs]


def main():
    path = "/home/murilo/dev/python/tcc/dados_dp/"
    experimentos = load_experiments(base_dir=path, antennas=["antena1"])
    exp = get_experiment(experimentos, "antena1", 0)

    voltages = norm_exp_same(exp)
    energies = cum_energy_exp(exp)

    fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    colors = ["C0", "C1", "C2", "C3"]
    for v, ax, ch, c in zip(energies, axs, exp.ch_names, colors):
        ax.plot(exp.time(), v, label=ch, color=c)
        ax.set_ylabel("Voltage (V)", fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.suptitle(r"Energia cumulativa $\sum{V(t)^2}$")

    plt.show()

    r = None

    v1 = energies[0]
    v2 = energies[1]
    plot_dtw(v1, v2, dist, radius=r)
    plt.show()


if __name__ == "__main__":
    main()
