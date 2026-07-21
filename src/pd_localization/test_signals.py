import numpy as np
import matplotlib.pyplot as plt


# double exponent attenuation oscillation function
def double_exp_oscillator(t, t_0, gain, zeta, center_frequency):
    exps = np.exp(-1.3 * (t - t_0) / zeta) - np.exp(-2.2 * (t - t_0) / zeta)
    result = gain * exps * np.sin(2 * np.pi * center_frequency * t)
    result[(t - t_0) < 0] = 0
    return result


def add_noise(signal, t, snr_db, f_n):
    amp = 1e-3
    osci_noise = amp * np.sin(t * 2 * np.pi * f_n)
    signal_2 = signal + osci_noise

    signal_power = np.mean(signal_2**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(t))
    return signal_2 + noise


def generate_pd(t_0, gain, snr=10, samples=250):
    tamos = 4e-10  # 0.4ns
    t = np.arange(0, tamos * samples, tamos)
    # t_0 = tamos * 50  # 20ns
    # t_1 = tamos * 60  # 24ns
    s1 = double_exp_oscillator(t, t_0, gain, 6e-9, 0.5e9)
    s1n = add_noise(s1, t, snr, 1.8e9)
    return s1n


def main():
    tamos = 0.4e-9
    s = generate_pd(tamos * 50, 1)
    plt.plot(s)
    plt.show()


if "__main__" == __name__:
    main()
