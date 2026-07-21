import numpy as np
import matplotlib.pyplot as plt

from pd_localization.test_signals import generate_pd
from pd_localization.dtw import dist, gen_cost_matrix, backtracking, plot_dtw
from pd_localization.preprocessing import curv_detection

# NOTAS GERAIS
# pontos positivos: é robusto em relação a reflexão dos cabos, pois apenas olha o primeiro joelho
# negativos: é invisivel a forma do pulso, diante de alto ruido e interferência, vai pro saco


def cum_energy(signals):
    out = []
    for s in signals:
        sum = np.cumsum(s**2)
        out.append(sum / np.max(sum))
    return out


def plot_signals(signals):
    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, s in zip(axs, signals):
        ax.plot(s)


if "__main__" == __name__:
    tamos = 0.4e-9
    t_0 = 50 * tamos  # 20ns
    taus_sample = np.array([0, 10, 15, 30])
    gains = np.array([1, 0.5, 0.6, 0.1])
    taus = taus_sample * tamos + t_0
    signals = []
    for ts, gain in zip(taus, gains):
        s = generate_pd(ts, gain)
        signals.append(s)

    voltages = cum_energy(signals)

    ref_idx = curv_detection(voltages[0])
    ch_names = [f"CH{i + 1}" for i in range(len(voltages))]
    v_ref = voltages[0]
    taus = {}
    for i, (v, ch) in enumerate(zip(voltages, ch_names)):
        if i == 0:
            taus[ch] = 0.0
            continue
        D = gen_cost_matrix(v_ref, v, dist, radius=None)
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

        taus[ch] = estimative * tamos

    taus_estimados = np.array(list(taus.values())) / tamos
    print("taus estimados:\t", taus_estimados)
    print("taus reais:\t", taus_sample)
    erro = np.sum(np.abs(taus_sample - taus_estimados))
    print("erro total:\t", erro)

    plot_signals(signals)
    plt.show()
    plot_signals(voltages)
    plt.show()
