from icecream import ic
from pd_localization.experiment_loader import (
    Experiment,
    load_experiments,
    get_experiment,
)
from pd_localization.preprocessing import norm_zero_mean, envelope_exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import envelope


def dist(x, y):
    return (x - y) ** 2


def hamming_window(N):
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))


def blackman_window(N):
    n = np.arange(N)
    return 0.42 - 0.5 * np.cos((2 * np.pi * n) / (N - 1)) * 0.08 * np.cos(
        (4 * np.pi * n) / (N - 1)
    )


def filter(fs, f1, filter_length, window_function=None):
    N = filter_length
    w1 = 2 * np.pi * f1 / fs

    h_d = np.zeros(N)
    for i in range(N):
        M = (N - 1) // 2
        k = i - M
        if k == 0:
            h_d[i] = w1 / np.pi
        else:
            h_d[i] = (np.sin(w1 * k)) / (np.pi * k)

    if window_function is None:
        return h_d
    else:
        return h_d * window_function(N)


def freq_response(h, fs, length=2**16):
    H = np.fft.fft(h, length)
    H = np.fft.fftshift(H)

    freq = np.linspace(-fs / 2, fs / 2, length)
    mag = 20 * np.log10(np.abs(H))

    plt.subplot(2, 1, 1)
    plt.plot(h)
    plt.subplot(2, 1, 2)
    plt.plot(freq / 1e9, mag)
    plt.xlabel("frequência (GHz)")
    plt.ylabel("magnitude (dB)")
    plt.grid(True)


def cum_energy_exp(exp: Experiment):
    chs = exp.ch_names
    energies = []
    for ch in chs:
        cum_energy = np.cumsum(exp.voltage(ch) ** 2)
        energies.append(cum_energy / cum_energy.max())
    return energies


def norm_exp_uniform(exp: Experiment):
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        ch_max = np.max(exp.voltage(ch))
        ch_min = np.min(exp.voltage(ch))
        v = (exp.voltage(ch) - ch_min) / (ch_max - ch_min)
        voltages.append(v - np.mean(v))
    return voltages


def norm_exp_same(exp: Experiment):
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        ch_mean = np.mean(exp.voltage(ch))
        ch_std = np.std(exp.voltage(ch))
        voltages.append((exp.voltage(ch) - ch_mean) / ch_std)
    return voltages


def cum_energy(voltages: list[np.ndarray]):
    def relu(x, b=0.0):
        return np.maximum(x + b, 0.0)

    energies = []
    for v in voltages:
        v = relu(v, -0.1)
        cum_energy = np.cumsum(v**2)
        energies.append(cum_energy / cum_energy.max())
    return energies


def main():
    path = "/home/murilo/dev/python/tcc/dados_dp/"
    antena = "antena1"
    experimentos = load_experiments(base_dir=path, antennas=[antena])
    exp = get_experiment(experimentos, antena, 3)
    T = 1 / exp.sample_rate
    v_env = envelope_exp(exp)
    v_cum = cum_energy_exp(exp)
    voltages = norm_zero_mean(exp)
    # n = len(voltages[0])
    signal_start = (98, 106, 113, 106)
    # pb = (None, int(100000 * n * T))
    pb = (None, 90)

    voltages_filtered = []
    for v in voltages:
        env, res = envelope(v, pb)
        voltages_filtered.append(env + res)
    print(voltages_filtered[0].shape)

    fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    colors = ["C0", "C1", "C2", "C3"]
    for v, vf, ax, c in zip(v_cum, v_env, axs, colors):
        ax.plot(v, "--", alpha=0.5, color=c, label="sinal original")
        ax.plot(vf, label="sinal filtrado", color=c)
        ax.set_ylabel("Voltage (V)", fontsize=11)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        # fig.suptitle(r"Sinais filtrados")
    plt.show()


if __name__ == "__main__":
    main()
