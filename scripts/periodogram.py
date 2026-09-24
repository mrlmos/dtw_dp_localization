import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import csd, welch, spectrogram, butter, filtfilt
from scipy.fft import fft, ifft, fftshift
from pd_localization.experiment_loader import load_experiments, get_experiment


def uhf_filter(x: np.ndarray, fs, fc=300e6):
    b, a = butter(4, fc, btype="high", fs=fs)
    x_filt = filtfilt(b, a, x)
    return x_filt


def get_sd(x: np.ndarray, fs: float):
    n = 2048
    S1_c = np.abs(fftshift(fft(np.correlate(x, x, mode="full"), n=n)))
    f_c = fftshift(np.fft.fftfreq(n, 1 / fs))
    return S1_c, f_c


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
experiments = load_experiments(DATA_PATH)
exp = get_experiment(experiments, "antena1", 0)

fs = exp.sample_rate
s1 = exp.voltage("CH2")
s2 = exp.voltage("CH2")
t = np.linspace(0, len(s1) / fs, num=len(s1)) * 1e9
n = 2048

s1_f = uhf_filter(s1, fs)
S1, f_s1 = get_sd(s1_f, fs)
f_s1 /= 1e9

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t, s1_f)
ax1.set_xlabel("time (ns)")
ax1.set_title("Time signal")

ax2.plot(f_s1[n // 2 :], S1[n // 2 :], label="Spectral Density")
ax2.axvspan(0.3, f_s1[-1], color="red", alpha=0.1, label="UHF Range")
ax2.set_xlabel("GHz")
ax2.set_title(r"$\mathcal{F}\{R[\tau]\}$")

plt.legend()
plt.tight_layout()
plt.show()

wl = 32
overlap = 30
nfft = 512

freqs, time, Sxx = spectrogram(
    s1_f, fs, "hann", nperseg=wl, noverlap=overlap, nfft=nfft, scaling="spectrum"
)
Sxx_dB = 10 * np.log10(Sxx + 1e-10)

plt.figure(figsize=(8, 5))
mesh = plt.pcolormesh(
    time * 1e6, freqs / 1e6, Sxx_dB, shading="gouraud", cmap="viridis"
)

plt.colorbar(mesh, label="Power (dB)")
plt.title("Spectrogram of 250-Sample Transient")
plt.ylabel("Frequency (MHz)")
plt.xlabel("Time (µs)")
plt.tight_layout()
plt.show()
