from scipy.signal import lfilter
from pd_localization.test_signals import generate_pd
from pd_localization.experiment_loader import load_experiments, get_experiment
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
    print(f"matrix rank: {np.linalg.matrix_rank(big_gcc)}")
    return np.argmax(big_gcc, axis=-1) - len(s1) + 1


def main():
    DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
    experiments = load_experiments(DATA_PATH)
    exp = get_experiment(experiments, "antena1", 0)

    tamos = 0.4e-9
    t_0 = 150 * tamos  # 40ns

    taus_sample = np.array([0, 10, 15, 30])
    gains = np.array([1, 0.5, 0.6, 0.1])
    snr = np.array([15, 10, 5, 2])
    # betas = np.array([0.6, 1.2, 1.2, 2.7]) * 1e-9
    betas = np.array([0.6, 0.6, 0.6, 0.6]) * 1e-9
    # betas = np.array([2.7, 2.7, 2.7, 2.7]) * 1e-9
    taus = taus_sample * tamos + t_0

    f_start = 300e6
    f_end = 1.2e9
    K = 400

    x = exp.voltage("CH1")
    y = exp.voltage("CH2")

    f_start = 300e6
    f_end = 1.2e9
    K = 400
    freqs = np.linspace(f_start, f_end, K)
    X = ressonant_filter(batch_shift(x, f_start, f_end, K=K))
    est = TDOA_estimation(x, y, f_start, f_end, K)
    cond = np.abs(est) < 50

    plt.subplot(1, 2, 1)
    plt.imshow(X[::-1, :])
    plt.xlabel("Amostras")
    plt.ylabel("Frequência")
    plt.title('"Espectograma"')
    plt.subplot(1, 2, 2)
    plt.plot(freqs[cond] * 1e-9, est[cond])
    plt.xlabel("Frequência de Ressonancia (GHz)")
    plt.ylabel("Delay estimado (amostras)")
    plt.title(
        f"mean absolute delay: {np.abs(np.mean(est, where=cond)):.4f} | ideal: 6.66"
    )
    plt.show()


if __name__ == "__main__":
    main()
