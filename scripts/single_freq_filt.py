from scipy.signal import lfilter
from pd_localization.experiment_loader import load_experiments, get_experiment
from pd_localization.localizacao import true_taus
from pd_localization.gcc import gcc_phat
import numpy as np
import matplotlib.pyplot as plt

FS = 1 / 4e-10


def batch_shift(signal: np.ndarray, f_start: float, f_end: float, K: int = 100, fs=FS):
    freqs = np.linspace(f_start, f_end, K).reshape(-1, 1)
    n = np.arange(len(signal))
    omega = np.pi - 2 * np.pi * freqs / fs
    return signal * np.exp(1j * omega * n)


def ressonant_filter(x_k: np.ndarray, r: float = 0.99) -> np.ndarray:
    b = [1.0]
    a = [1.0, r]
    y_k = lfilter(b, a, x_k)
    return np.abs(y_k)


def TDOA_estimation(s1, s2, f_start, f_end, K, fs=FS):
    s1_k = batch_shift(s1, f_start, f_end, K)
    y1_k = ressonant_filter(s1_k)

    s2_k = batch_shift(s2, f_start, f_end, K)
    y2_k = ressonant_filter(s2_k)

    big_gcc = np.array([gcc_phat(y1, y2) for (y1, y2) in zip(y1_k, y2_k)])
    return np.argmax(big_gcc, axis=-1) - len(s1) + 1


def main():
    DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
    experiments = load_experiments(DATA_PATH)
    exp = get_experiment(experiments, "antena1", 0)

    x = exp.voltage("CH1")
    y = exp.voltage("CH3")

    f_start = 300e6
    f_end = 1.2e9
    K = 400
    freqs = np.linspace(f_start, f_end, K)
    X = ressonant_filter(batch_shift(x, f_start, f_end, K=K))
    est = TDOA_estimation(x, y, f_start, f_end, K)
    cond = np.abs(est) < 50

    plt.subplot(1, 2, 1)
    plt.imshow(X[::-1, :])
    plt.subplot(1, 2, 2)
    plt.plot(freqs[cond], est[cond])
    plt.title(f"mean absolute delay: {np.abs(np.mean(est, where=cond))}")
    plt.show()


if __name__ == "__main__":
    main()
