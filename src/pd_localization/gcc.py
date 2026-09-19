import numpy as np
from .experiment_loader import Experiment
from scipy.signal import lfilter


def gcc_phat(x: np.ndarray, y: np.ndarray):
    corrlen = len(x) + len(y) - 1
    fftlen = corrlen
    spec1 = np.fft.fft(x, n=fftlen)
    spec2 = np.fft.fft(y, n=fftlen)

    spec12 = spec1 * np.conj(spec2)
    phat_fft = spec12 / np.abs(spec12)

    gcc = np.fft.ifft(phat_fft, n=fftlen).real
    return np.fft.fftshift(gcc)


def gcc_ps(x: np.ndarray, y: np.ndarray):
    p = 0.5
    corrlen = len(x) + len(y) - 1
    fftlen = corrlen
    spec1 = np.fft.fft(x, n=fftlen)
    spec2 = np.fft.fft(y, n=fftlen)

    spec11 = spec1 * np.conj(spec1)
    spec22 = spec2 * np.conj(spec2)
    spec12 = spec1 * np.conj(spec2)

    psi = (np.sqrt(spec11 * spec22) - np.abs(spec12)) / (np.abs(spec12) ** p)
    phat_fft = spec12 * np.abs(psi)
    gcc = np.fft.ifft(phat_fft, n=fftlen).real
    return np.fft.fftshift(gcc)


def gcc_phat_p(x: np.ndarray, y: np.ndarray):
    p = 0.8
    corrlen = len(x) + len(y) - 1
    fftlen = corrlen
    spec1 = np.fft.fft(x, n=fftlen)
    spec2 = np.fft.fft(y, n=fftlen)

    spec11 = spec1 * np.conj(spec1)
    spec22 = spec2 * np.conj(spec2)
    spec12 = spec1 * np.conj(spec2)

    coherence = (np.abs(spec12) ** 2) / (spec11 * spec12)
    psi = 1 / (np.abs(spec12) ** p + 0.1 * np.mean(np.abs(spec12)))
    phat_fft = spec12 * psi
    gcc = np.fft.ifft(phat_fft, n=fftlen).real
    return np.fft.fftshift(gcc)


def gcc_roth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    corrlen = len(x) + len(y) - 1
    fftlen = corrlen
    spec1 = np.fft.fft(x, n=fftlen)
    spec2 = np.fft.fft(y, n=fftlen)
    spec12 = spec1 * np.conj(spec2)
    spec11 = (spec1 * np.conj(spec1)).real

    roth_fft = spec12 / spec11
    gcc = np.fft.ifft(roth_fft, n=fftlen).real
    return np.fft.fftshift(gcc)


def gcc_scot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    corrlen = len(x) + len(y) - 1
    fftlen = corrlen
    spec1 = np.fft.fft(x, n=fftlen)
    spec2 = np.fft.fft(y, n=fftlen)

    spec12 = spec1 * np.conj(spec2)
    spec11 = (spec1 * np.conj(spec1)).real
    spec22 = (spec2 * np.conj(spec2)).real

    scot_fft = spec12 / np.sqrt(spec11 * spec22)
    gcc = np.fft.ifft(scot_fft, n=fftlen).real
    return np.fft.fftshift(gcc)


def gcc_ht(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    corrlen = len(x) + len(y) - 1
    fftlen = corrlen
    spec1 = np.fft.fft(x, n=fftlen)
    spec2 = np.fft.fft(y, n=fftlen)

    spec12 = spec1 * np.conj(spec2)
    spec11 = (spec1 * np.conj(spec1)).real
    spec22 = (spec2 * np.conj(spec2)).real

    coh = 1 / np.sqrt(spec11 * spec22)
    psi = 1 / np.abs(spec12) * (np.abs(coh) ** 2) / (1 - np.abs(coh) ** 2)
    gcc = np.fft.ifft(spec12 * psi, n=fftlen).real
    return np.fft.fftshift(gcc)


def sff_gcc(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    FS = 1 / 4e-10
    f_start = 300e6
    f_end = 1.2e9
    K = 400

    def batch_shift(
        signal: np.ndarray, f_start: float, f_end: float, K: int = 100, fs=FS
    ):
        freqs = np.linspace(f_start, f_end, K).reshape(-1, 1)
        n = np.arange(len(signal))
        omega = np.pi - 2 * np.pi * freqs / fs
        return signal * np.exp(1j * omega * n)

    def ressonant_filter(x_k: np.ndarray, r: float = 0.99) -> np.ndarray:
        b = [1.0]
        a = [1.0, r]
        y_k = lfilter(b, a, x_k)
        return np.abs(y_k)

    s1_k = batch_shift(x, f_start, f_end, K)
    y1_k = ressonant_filter(s1_k)

    s2_k = batch_shift(y, f_start, f_end, K)
    y2_k = ressonant_filter(s2_k)

    big_gcc = np.array([gcc_phat(y1, y2) for (y1, y2) in zip(y1_k, y2_k)])
    return big_gcc
