from icecream import ic
from pd_localization.experiment_loader import (
    Experiment,
    load_experiments,
    get_experiment,
)
import numpy as np
import matplotlib.pyplot as plt


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
        voltages.append((exp.voltage(ch) - ch_mean) / ch_std)
    return voltages


def norm_exp_uniform(exp: Experiment):
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        ch_max = np.max(exp.voltage(ch))
        ch_min = np.min(exp.voltage(ch))
        v = (exp.voltage(ch) - ch_min) / (ch_max - ch_min)
        voltages.append(v - np.mean(v))
    return voltages


def fft_exp(voltages):
    voltages_fft = []
    for v in voltages:
        w = np.hanning(len(v))
        v_fft = np.fft.fftshift(np.fft.fft(v * w))
        mag = np.abs(v_fft)
        phase = np.unwrap(np.angle(v_fft))
        voltages_fft.append({"mag": mag, "phase": phase})
    return voltages_fft


def main():
    path = "/home/murilo/dev/python/tcc/dados_dp/"
    antena = "antena2"
    experimentos = load_experiments(base_dir=path, antennas=[antena])
    exp = get_experiment(experimentos, antena, 3)
    voltages = norm_exp_uniform(exp)
    v = voltages[0]
    fs = exp.sample_rate
    N = len(v)

    # v_fft = np.fft.fftshift(np.fft.fft(v))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))
    voltages_fft = fft_exp(voltages)

    fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    colors = ["C0", "C1", "C2", "C3"]
    for v, ax, ch, c in zip(voltages_fft, axs, exp.ch_names, colors):
        ax.plot(freq / 1e9, v["mag"], label=ch, color=c)
        ax.set_ylabel("Magnitude", fontsize=12)
        ax.set_xlabel("Frequência (GHz)", fontsize=12)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        # fig.suptitle(r"Energia cumulativa $\sum{V(t)^2}$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
