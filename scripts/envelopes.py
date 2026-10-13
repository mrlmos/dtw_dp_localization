import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from numpy.fft import fft, ifft, fftshift

from pd_localization.experiment_loader import load_experiments, get_experiment
from pd_localization.preprocessing import norm_zero_mean


def envelope(signal, cutoff=0.15, order=4):
    b, a = ss.butter(N=order, Wn=cutoff, btype="low")
    env = ss.filtfilt(b, a, signal**2)
    return np.sqrt(np.maximum(0, env))


experiments = load_experiments(antennas=["antena1"])
exp = get_experiment(experiments, "antena1", 0)
n = 512
fs = exp.sample_rate
freqs = np.linspace(-fs / 2, fs / 2, num=n) / fs
t = exp.time()

v1 = exp.voltage("CH2")
env = envelope(v1)
# freqs = fftshift(np.fft.fftfreq(n, d=1 / fs))
v_fft = np.abs(fftshift(fft(v1**2, n=n)))

plt.subplot(2, 1, 2)
plt.plot(freqs, v_fft)

plt.subplot(2, 1, 1)
plt.plot(t, np.abs(v1), ls="--")
plt.plot(t, env)
plt.show()
