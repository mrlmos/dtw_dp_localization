import os
from icecream import ic
import pandas as pd
from collections.abc import Callable
from experiment_loader import Experiment, load_experiments, get_experiment
from dtw import gen_cost_matrix, backtracking, plot_dtw
import matplotlib.pyplot as plt
import numpy as np


def dist(x, y):
    return (x - y) ** 2


def relu(x, b):
    return np.maximum(x + b, 0.0)


def cum_energy_exp(exp: Experiment):
    chs = exp.ch_names
    energies = []
    for ch in chs:
        cum_energy = np.cumsum(exp.voltage(ch) ** 2)
        energies.append(cum_energy / cum_energy.max())
    return energies


def relu_energy(exp: Experiment, bias=-0.2):
    normalized_voltages = norm_zero_mean(exp)
    energies = []
    for v in normalized_voltages:
        v = relu(np.abs(v), bias)
        cum_energy = np.cumsum(v**2)
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


def norm_zero_mean(exp: Experiment):
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        ch_max = np.max(exp.voltage(ch))
        ch_min = np.min(exp.voltage(ch))
        v = (exp.voltage(ch) - ch_min) / (ch_max - ch_min)
        voltages.append(v - np.mean(v))
    return voltages


def plot_4signals(
    signal: np.ndarray,
    exp: Experiment,
):
    # custom stuff
    signal_start = np.array([98, 106, 113, 106])
    delta = np.abs(signal_start - signal_start[0])

    colors = ["C0", "C1", "C2", "C3"]
    fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    for v, start, ax, ch, c in zip(signal, signal_start, axs, exp.ch_names, colors):
        ve = relu(np.abs(v), -0.3)
        cum_energy = np.cumsum(ve**2)
        cum_energy /= cum_energy.max()

        deriv = filtro_derivada(cum_energy)
        idx = np.argmax(deriv > 0.01)
        ic(idx)

        # cum_energy = filtro_derivada(cum_energy)
        cum_energy2 = np.cumsum(v**2)
        cum_energy2 /= cum_energy2.max()
        # custom stuff
        # delta = np.abs(signal_start[0] - start)
        ax.axvline(x=idx - 1, c="red", label=f"amostra {idx}, $\Delta = ${delta}")

        # actual plot
        ax.plot(cum_energy2, "--", alpha=0.5, label="densidade original", color=c)
        ax.plot(cum_energy, label="densidade relu", color=c)
        ax.set_ylabel("Voltage (V)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=12)
        fig.suptitle(r"Energia cumulativa $\sum{V(t)^2}$")


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


def filtro_derivada(x: np.ndarray):
    y = np.zeros_like(x)
    for n in range(1, len(x)):
        y[n] = x[n] - x[n - 1]
    return y


def deriv_estimation(x: np.ndarray):
    x_deriv = filtro_derivada(x)
    idx = np.argmax(x_deriv > 0.01)
    return idx - 1


def check_estimation(exp: Experiment, pre_processing: Callable, method: Callable):
    voltages = pre_processing(exp)

    colors = ["C0", "C1", "C2", "C3"]
    fig, axs = plt.subplots(4, 1, figsize=(14, 8))
    for v, ax, c, ch in zip(voltages, axs, colors, exp.ch_names):
        idx_estim = method(v)
        ax.plot(v, c=c, label=ch)
        ax.axvline(idx_estim, alpha=0.6, label=f"amostra: {idx_estim}")
        ax.set_xlabel("amostras", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=12)
    fig.suptitle("Estimativa de inicial")


def print_df(df: pd.DataFrame):
    df_means = df.groupby(["pd_location", "sinal"])[["erro_min", "erro_max"]].mean()
    for local, group in df.groupby("pd_location"):
        print(f"================ {local} ================")
        print(group.drop(columns="pd_location"))
    print("===================== médias =====================")
    print(df_means)


def exel_export(df: pd.DataFrame, dir: str, fname: str):
    path = os.path.join(dir, fname)
    means = df.groupby(["pd_location", "sinal"])[["erro_min", "erro_max"]].mean()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for antenna, group in df.groupby("pd_location"):
            group.drop(columns="pd_location").to_excel(
                writer, sheet_name=antenna, index=False
            )
        means.to_excel(writer, sheet_name="médias")
    print("Excel exportado !")


def batch_estimation(
    experiments=list[Experiment],
    pre_processing=cum_energy_exp,
    radius=24,
    distance=dist,
    ref_localization=deriv_estimation,
):
    refs = {"antena1": 0, "antena2": 1, "antena3": 2, "antena4": 3}
    results = []
    for exp in experiments:
        voltages = pre_processing(exp)
        wave_delay = np.array(list(exp.travel_times.values())) * exp.sample_rate

        # Target time delay estimation
        tdoa = np.abs(wave_delay - wave_delay[refs[exp.antenna]])

        ref_index = refs[exp.antenna]
        v_ref = voltages[ref_index]
        ref_localization_index = ref_localization(v_ref)

        for i in range(len(voltages)):
            if i == ref_index:
                continue
            D = gen_cost_matrix(v_ref, voltages[i], distance, radius=radius)
            path = backtracking(D)[::-1]
            linked = [pj for pi, pj in path if pi == ref_localization_index]

            tdoa_min = np.abs(np.min(linked) - ref_localization_index)
            tdoa_max = np.abs(np.max(linked) - ref_localization_index)
            tdoa_true = tdoa[i]

            results.append(
                {
                    "pd_location": exp.antenna,
                    "experimento": exp.index,
                    "sinal_ref": exp.ch_names[ref_index],
                    "sinal": exp.ch_names[i],
                    "tdoa_true": tdoa_true,
                    "tdoa_min": tdoa_min,
                    "tdoa_max": tdoa_max,
                    "links": len(linked),
                    "erro_min": np.abs(tdoa_min - tdoa_true),
                    "erro_max": np.abs(tdoa_max - tdoa_true),
                }
            )
    return pd.DataFrame(results)


path = "/home/murilo/dev/python/tcc/dados_dp/"
experimentos = load_experiments(base_dir=path, antennas=None)

exp = get_experiment(experimentos, "antena2", 0)
ts = 1 / exp.sample_rate
t1 = 98 * ts
tau = np.array([6.0, 7.0, 15.0]) * ts  # tau_12 tau_13 tau_14

check_estimation(exp, cum_energy_exp, deriv_estimation)
plt.show()
