import numpy as np
from icecream import ic
from experiment_loader import Experiment


# ------------------------------------------
# Normalization
# -----------------------------------------


def norm_zero_mean(exp: Experiment) -> list[np.ndarray]:
    """
    Normaliza todos os canais para [0,1] e subtrai a média.
    """
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        ch_max = np.max(exp.voltage(ch))
        ch_min = np.min(exp.voltage(ch))
        v = (exp.voltage(ch) - ch_min) / (ch_max - ch_min)
        voltages.append(v - np.mean(v))
    return voltages


def normz_exp(exp: Experiment) -> list[np.ndarray]:
    """
    Normalização Z de cada canal independentemente.
    """
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        ch_mean = np.mean(exp.voltage(ch))
        ch_std = np.std(exp.voltage(ch))
        voltages.append((exp.voltage(ch) - ch_mean) / ch_std)
    return voltages


def min_max_norm(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# ------------------------------------------
# Limiarização
# ------------------------------------------


def relu(x, b):
    return np.maximum(x + b, 0.0)


# ------------------------------------------
# Energias/Densidades
# ------------------------------------------


def cum_energy_exp(exp: Experiment):
    chs = exp.ch_names
    energies = []
    for ch in chs:
        cum_energy = np.cumsum(exp.voltage(ch) ** 2)
        energies.append(cum_energy / cum_energy.max())
    return energies


def relu_energy(exp: Experiment, bias=-0.1):
    normalized_voltages = norm_zero_mean(exp)
    energies = []
    for v in normalized_voltages:
        v = relu(np.abs(v), bias)
        cum_energy = np.cumsum(v**2)
        energies.append(cum_energy / cum_energy.max())
    return energies


def energy_criterion(exp: Experiment):
    chs = exp.ch_names
    energies = []
    for ch in chs:
        v = exp.voltage(ch) ** 2
        px = np.mean(v)
        ec = np.zeros_like(v)
        for k in range(len(v)):
            ec[k] = np.sum(v[:k]) - k * px
        energies.append(min_max_norm(ec))
    return energies


def akaike_info(exp: Experiment):
    chs = exp.ch_names
    energies = []
    for ch in chs:
        v = exp.voltage(ch)
        eic = np.zeros_like(v)
        for k in range(len(v)):
            var1 = np.std(v[: k + 1]) ** 2
            var2 = np.std(v[k:]) ** 2
            eval = k * np.log(var1 + 1e-9) + (len(v) - k - 1) * np.log(var2 + 1e-9)
            eic[k] = eval
        energies.append(min_max_norm(eic))
    return energies


# ------------------------------------------
# Reference sample estimation
# ------------------------------------------


def filtro_derivada(x: np.ndarray):
    y = np.zeros_like(x)
    for n in range(1, len(x)):
        y[n] = x[n] - x[n - 1]
    return y


def deriv_estimation(x: np.ndarray):
    x_deriv = filtro_derivada(x)
    idx = np.argmax(x_deriv > 0.04)  # TODO: Mudar
    return idx - 1
