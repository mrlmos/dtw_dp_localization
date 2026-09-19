from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from pd_localization.test_signals import generate_pd
from pd_localization.dtw import dist, fast_gen_cost_matrix, backtracking, plot_dtw
from pd_localization.preprocessing import curv_detection, min_max_norm
from pd_localization.gcc import gcc_ht, gcc_phat, gcc_roth, gcc_scot

# NOTAS GERAIS
# pontos positivos: é robusto em relação a reflexão dos cabos, pois apenas olha o primeiro joelho
# negativos: é invisivel a forma do pulso, diante de alto ruido e interferência, vai pro saco

ch_names = [f"CH{i + 1}" for i in range(4)]


def cum_energy(signals):
    out = []
    for s in signals:
        sqrd = np.cumsum(s**2)
        out.append(sqrd / np.max(sqrd))
    return out


def norm_zero_mean(signals: np.ndarray) -> list[np.ndarray]:
    """
    Normaliza todos os canais para [0,1] e subtrai a média.
    """
    voltages = []
    for s in signals:
        ch_max = np.max(s)
        ch_min = np.min(s)
        v = (s - ch_min) / (ch_max - ch_min)
        voltages.append(v - np.mean(v))
    return voltages


def energy_criterion(signals):
    energies = []
    for s in signals:
        v = s**2
        px = np.mean(v)
        ec = np.zeros_like(v)
        for k in range(len(v)):
            ec[k] = np.sum(v[:k]) - k * px
        energies.append(min_max_norm(ec))
    return energies


def akaike_info(signals):
    energies = []
    for s in signals:
        eic = np.zeros_like(s)
        for k in range(len(s)):
            var1 = np.std(s[: k + 1]) ** 2
            var2 = np.std(s[k:]) ** 2
            value = k * np.log(var1 + 1e-9) + (len(s) - k - 1) * np.log(var2 + 1e-9)
            eic[k] = value
        energies.append(min_max_norm(eic))
    return energies


def plot_signals(signals, func=None):
    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, s in zip(axs, signals):
        ax.plot(s)
        if func:
            ax.axvline(func(s), color="k", linestyle="--")


def estimate_taus_dtw(voltages: list[np.ndarray]):
    ref_idx = curv_detection(voltages[0])
    v_ref = voltages[0]
    taus = {}
    for i, (v, ch) in enumerate(zip(voltages, ch_names)):
        if i == 0:
            taus[ch] = 0.0
            continue
        D = fast_gen_cost_matrix(v_ref, v, dist, radius=50)
        path = backtracking(D)[::-1]
        linked = [pj for pi, pj in path if pi == ref_idx]
        linked_diffs = np.abs(v_ref[ref_idx] - v[linked])
        diffs_sorted = np.argsort(linked_diffs)
        if len(diffs_sorted) == 1:
            estimative = np.abs(diffs_sorted[0] + linked[0] - ref_idx)
        elif len(diffs_sorted) < 4:
            estimative = np.abs(np.mean(diffs_sorted) + linked[0] - ref_idx)
        else:
            estimative = np.abs(np.mean(diffs_sorted[:3]) + linked[0] - ref_idx)

        taus[ch] = estimative
    return np.array(list(taus.values())), taus


def estimate_taus_energy(signals, knee_estimator: Callable, energy_func: Callable):
    """
    Akaike: Muito sensível a inconsistências no formato de onda. Crescimentos rápidos (menor beta)
            apresenta melhor definição do joelho e melhor precisão mesmo em alto ruído.

    Energy criterion: Mesma coisa.
    """
    voltages = energy_func(signals)
    ref_index = 0

    v_ref = voltages[ref_index]
    vref_knee = knee_estimator(v_ref)  # np.min if not cum_energy
    taus = {}
    for i, (v, ch) in enumerate(zip(voltages, ch_names)):
        if i == ref_index:
            taus[ch] = 0.0
            continue
        knee = knee_estimator(v)
        estimative = np.abs(knee - vref_knee)
        taus[ch] = estimative
    return np.array(list(taus.values())), taus


def estimate_taus_gcc(
    signals: list[np.ndarray],
    normalization: Callable = norm_zero_mean,
    gcc_func: Callable = gcc_phat,
):
    voltages = normalization(signals)
    ref_index = 0

    v_ref = voltages[ref_index]
    taus = {}
    for i, (v, ch) in enumerate(zip(voltages, ch_names)):
        if i == ref_index:
            taus[ch] = 0.0
            continue
        cross_corr = gcc_func(v_ref, v)
        estimative = np.abs(np.argmax(np.abs(cross_corr)) - len(v_ref) + 1)
        taus[ch] = estimative
    return np.array(list(taus.values())), taus


def gen_plot(taus, gains, snr, betas):
    signals = []
    for ts, gain, noise, beta in zip(taus, gains, snr, betas):
        s = generate_pd(
            ts, gain, snr=100, alpha=5e-9, beta=beta
        )  # a = 5e-9, b = 0.3e-9
        signals.append(s)

    v = akaike_info(signals)
    plot_signals(v, np.argmin)


if "__main__" == __name__:
    tamos = 0.4e-9
    t_0 = 150 * tamos  # 40ns

    taus_sample = np.array([0, 10, 15, 30])
    gains = np.array([1, 0.5, 0.6, 0.1])
    snr = np.array([15, 10, 5, 2])
    betas = np.array([0.6, 1.2, 1.2, 2.7]) * 1e-9
    # betas = np.array([0.6, 0.6, 0.6, 0.6]) * 1e-9
    # betas = np.array([2.7, 2.7, 2.7, 2.7]) * 1e-9
    taus = taus_sample * tamos + t_0

    signals = generate_pd(t_0, 1, snr=None)
    s1 = generate_pd(taus[0], 1, snr=None)
    s2 = generate_pd(taus[3], 1, snr=None)
    # print(len(s1), len(s2))
    cc = np.correlate(s1, s2, mode="full")
    aic = akaike_info([generate_pd(t_0, 1, snr=None)])
    aic = aic[0]
    print(np.argmax(cc), "\tlen: ", len(s1), "\ttau: ", 30)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(generate_pd(t_0, 1, snr=None))
    axs[0].set_xlabel("(a)", fontsize=18)
    axs[1].plot(aic)
    axs[1].set_xlabel("(b)", fontsize=18)
    # fig, ax = plt.subplot_mosaic([["s1", "s2"], ["cc", "cc"]], figsize=(11, 8))
    #
    # ax["s1"].plot(s1)
    # ax["s1"].set_xlabel("(a)", fontsize=18)
    # ax["s2"].plot(s2)
    # ax["s2"].set_xlabel("(b)", fontsize=18)
    # ax["cc"].plot(cc)
    # ax["cc"].set_xlabel("(c)", fontsize=18)
    #
    for a in axs:
        # Remove top and right spines
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        # a.spines["left"].set_visible(False)

        # Remove y-axis ticks and labels
        a.set_yticks([])
        a.tick_params(axis="y", length=0)

        # Optional: remove y-axis tick marks entirely

    plt.tight_layout(w_pad=10)
    # plt.show()
    plt.savefig("./draft_tcc/medias/akaike.png")
