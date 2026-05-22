from icecream import ic
from experiment_loader import Experiment, load_experiments, get_experiment
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
    n = np.arange(N)

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


def cum_energy_exp(voltages: list[np.ndarray]):
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
    antena = "antena4"
    experimentos = load_experiments(base_dir=path, antennas=[antena])
    exp = get_experiment(experimentos, antena, 3)
    ic(exp.distances)
    voltages = norm_exp_uniform(exp)
    energies = cum_energy_exp(voltages)

    FILTER_SIZE = 101

    f1 = 600e6
    f2 = exp.sample_rate / 2
    h = filter(exp.sample_rate, f1, FILTER_SIZE, window_function=hamming_window)
    signal_start = (98, 106, 113, 106)

    voltages_filtered = []
    for v in voltages:
        vf = np.convolve(v, h, "same")
        voltages_filtered.append(vf)
    fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    colors = ["C0", "C1", "C2", "C3"]
    for v, vf, ax, c in zip(voltages, voltages_filtered, axs, colors):
        ax.plot(v, "--", alpha=0.5, color=c, label="sinal original")
        ax.plot(vf, label="sinal filtrado", color=c)
        ax.set_ylabel("Voltage (V)", fontsize=11)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        # fig.suptitle(r"Sinais filtrados")
    plt.show()

    # vref = voltages_filtered[0]
    # v_corr = np.correlate(vref, vref, mode="full")
    # plt.plot(v_corr)
    # plt.title(f"valor maximo: {np.argmax(v_corr)}")
    # plt.show()

    # energies = cum_energy_exp(voltages_filtered)
    # fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    # colors = ["C0", "C1", "C2", "C3"]
    # for v, start, ax, ch, c in zip(energies, signal_start, axs, exp.ch_names, colors):
    #     delta = np.abs(signal_start[0] - start)
    #     ax.plot(v, label=ch, color=c)
    #     ax.set_ylabel("Voltage (V)", fontsize=8)
    #     ax.grid(True, linestyle="--", alpha=0.4)
    #     ax.axvline(x=start, c="red", label=f"amostra {start}, $\Delta = ${delta}")
    #     ax.legend(loc="upper right", fontsize=12)
    #     fig.suptitle(r"Energia cumulativa $\sum{V(t)^2}$")
    # plt.show()


if __name__ == "__main__":
    main()
