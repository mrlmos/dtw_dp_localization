import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from pd_localization.experiment_loader import load_experiments, get_experiment
from pd_localization.preprocessing import norm_zero_mean


def power_law(x, k=10):
    return ((1 + k) ** x - 1) / k


def get_sd(x: np.ndarray, fs: float):
    n = 2048
    S1_c = np.abs(fftshift(fft(np.correlate(x, x, mode="full"), n=n)))
    f_c = fftshift(np.fft.fftfreq(n, 1 / fs))
    return S1_c, f_c


def preditor_linear(x: np.ndarray, p: int):
    """
    Calcula os Coeficientes a_k do preditor linear recursivo.
    -----------
    Entrada:
        x - vetor de entrada. \n
        p - ordem do filtro. \n
    Retorna:
        a_k - vetor de coeficientes.\n
        G - Ganho do sistema.
    """
    N = len(x)
    r_full = np.correlate(x, x, mode="full")
    r = r_full[(N - 1) : (N - 1 + p + 1)]

    psi = r[1:]
    phi = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            phi[i, j] = r[np.abs(i - j)]

    a_k = np.linalg.inv(phi) @ psi
    G = np.sqrt(r[0] - np.dot(a_k, r[1:]))
    return a_k, G


def lp_predict(x: np.ndarray, a_k: np.ndarray):
    N = len(x)
    p = len(a_k)
    x_pred = np.zeros_like(x)
    x_pred[:p] = x[:p]
    for n in range(p, N):
        x_pred[n] = np.dot(a_k[::-1], x[n - p : n])
    return x_pred


def lpc_spectrum(x: np.ndarray, freqs: np.ndarray, p: int = 8):
    """
    Calcula o espectro de potência de um sinal pelo LPC
    (Linear Predictive Coding)
    ----------
    Entradas:
        x - sinal de análise. \n
        freqs - frequências onde o ganho será avaliado \n
        p - ordem do filtro preditor linear \n
    Retorna:
        H - absoluto da resposta em frequência
    """
    a_k, G = preditor_linear(x, p)
    a_k = np.r_[1, -a_k]
    H = np.zeros_like(freqs, dtype=complex)
    for i, f in enumerate(freqs):
        omega = 2 * np.pi * f / fs
        z = np.exp(1j * omega)
        den = 0
        for k in range(p + 1):
            den += a_k[k] * (z**-k)
        H[i] = G / den
    return np.abs(H)


def lpc_gcc(x, y, fs, p=8):
    N = len(x) + len(y) - 1
    freqs = np.linspace(-fs / 2, fs / 2, num=N)
    H = lpc_spectrum(x, freqs, p)
    V1 = fftshift(fft(x, n=N))
    V2 = fftshift(fft(y, n=N))
    CC = V1 * np.conjugate(V2)
    new_gcc = fftshift(ifft(CC / H).real)
    return new_gcc


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
experiments = load_experiments(DATA_PATH)
exp = get_experiment(experiments, "antena1", 2)

fs = exp.sample_rate
t = exp.time()
voltages = norm_zero_mean(exp)
p = 8

erros = []
for v in voltages:
    a_k, _ = preditor_linear(v, p)
    erro = lp_predict(v, a_k)
    erros.append(erro)

fig, axs = plt.subplots(4, 1, sharex=True)
for i in range(len(axs)):
    axs[i].plot(t, erros[i], label="preditor")
    axs[i].plot(t, voltages[i], lw=1, ls="--", label="real")
    axs[i].legend()

fig.suptitle(f"order p = {p}")
plt.tight_layout()
plt.show()
