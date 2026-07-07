import numpy as np
from .experiment_loader import Experiment


def gcc_phat(x: np.ndarray, y: np.ndarray):
    corrlen = len(x) + len(y) - 1
    fftlen = corrlen
    spec1 = np.fft.fft(x, n=fftlen)
    spec2 = np.fft.fft(y, n=fftlen)

    spec12 = spec1 * np.conj(spec2)
    phat_fft = spec12 / np.abs(spec12)

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
