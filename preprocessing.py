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


def derivative_threshold(x: np.ndarray) -> np.float64:
    x_cum = np.cumsum(x**2)
    x_cum /= np.max(x_cum)

    half_power_index = np.argmax(x_cum > 0.5)
    split_index = len(x) - half_power_index
    if split_index <= 0:
        raise ValueError("Erro no calculo do index")

    x_noise = x[:split_index]
    max_deriv = np.max(filtro_derivada(x_noise))
    return max_deriv * 50, split_index


def deriv_estimation(x: np.ndarray) -> int:
    x_deriv = filtro_derivada(x)
    threshold, idx = derivative_threshold(x)
    print("Threshold value found: ", threshold, "\t Index: ", idx)
    idx = np.argmax(x_deriv > threshold)
    # idx = np.argmax(x_deriv > 0.04)  # TODO: Mudar
    return idx - 1


def pole_finding(x: np.ndarray) -> int:
    x_cum = np.cumsum(x**2)
    x_cum /= np.max(x_cum)
    print(np.max(x_cum))
    min_index = 10
    max_index = np.argmax(x_cum > 0.8)
    x_axis = np.arange(len(x))
    x = x_cum.copy()

    mean_sqrd_error = []
    for k in range(min_index, max_index):
        # Left model
        y1 = x[:k, np.newaxis]
        x1 = x_axis[:k, np.newaxis]
        A1 = np.concatenate((x1, np.ones_like(x1)), axis=1)
        th1 = np.linalg.pinv(A1) @ y1
        y1m = A1 @ th1

        # Right model
        y2 = x[k:, np.newaxis]
        x2 = x_axis[k:, np.newaxis]
        A2 = np.concatenate((x2, np.ones_like(x2)), axis=1)
        th2 = np.linalg.pinv(A2) @ y2
        y2m = A2 @ th2

        # Full model
        y_m = np.concatenate((y1m, y2m), axis=None)
        y_true = np.concatenate((y1, y2), axis=None)

        msq = np.mean((y_true - y_m) ** 2)
        mean_sqrd_error.append(msq)

    return np.argmin(mean_sqrd_error) + min_index
