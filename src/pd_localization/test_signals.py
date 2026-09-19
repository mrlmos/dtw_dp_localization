import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# double exponent attenuation oscillation function
def double_exp_oscillator(t, t_0, gain, zeta, center_frequency, a1=-1.3, a2=-2.2):
    exps = np.exp(a1 * (t - t_0) / zeta) - np.exp(a2 * (t - t_0) / zeta)
    result = gain * exps * np.sin(2 * np.pi * center_frequency * t)
    result[(t - t_0) < 0] = 0
    return result


# inverse double exponent oscillation function
def inverse_double_exponential(
    t, t_0, gain, center_frequency, alpha=44e-9, beta=9.9e-9
):
    alpha = 1 / alpha
    beta = 1 / beta
    k = ((beta + alpha) / beta) * ((beta / alpha) ** (alpha / (beta + alpha)))
    exps = k * 1 / (np.exp(alpha * (t - t_0)) + np.exp(-beta * (t - t_0)))
    result = gain * exps * np.sin(2 * np.pi * center_frequency * t)
    # result[(t - t_0) < 0] = 0
    return result


def add_noise(signal, t, snr_db, f_n):
    amp = 1e-3
    osci_noise = amp * np.sin(t * 2 * np.pi * f_n)
    signal_2 = signal + osci_noise * 0

    signal_power = np.mean(signal_2**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(t))
    return signal_2 + noise


def generate_pd(t_0, gain=1, snr=10, samples=250, alpha=5e-9, beta=0.3e-9):
    tamos = 4e-10  # 0.4ns
    t = np.arange(0, tamos * samples, tamos)
    # s1 = double_exp_oscillator(t, t_0, gain, 6e-9, 0.5e9)
    s1 = inverse_double_exponential(t, t_0, gain, 0.5e9, alpha=alpha, beta=beta)
    if snr is None:
        return s1
    else:
        s1n = add_noise(s1, t, snr, 1.8e9)
        return s1n


def main():
    tamos = 0.4e-9
    t = np.arange(0, tamos * 250, tamos)
    gain = 1
    zeta = 6e-9
    fc = 1e-9
    a1 = -1.3
    a2 = -2.2
    s = double_exp_oscillator(t, tamos * 50, gain, zeta, fc, a1=a1, a2=a2)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    (line,) = ax.plot(t * 1e9, s)
    ax.set_xlabel("time [ns]")

    ax_a1 = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_a2 = plt.axes([0.2, 0.08, 0.6, 0.03])
    slider_a1 = Slider(ax_a1, "Exp 1", -3.0, -0.03, valinit=a1)
    slider_a2 = Slider(ax_a2, "Exp 2", -3.0, -0.03, valinit=a2)

    def update(val):
        s = double_exp_oscillator(
            t, tamos * 50, gain, zeta, fc, a1=slider_a1.val, a2=slider_a2.val
        )
        line.set_ydata(s)
        ax.relim()
        ax.autoscale_view()
        # fig.canvas.draw_idle()

    slider_a1.on_changed(update)
    slider_a2.on_changed(update)
    plt.show()


if "__main__" == __name__:
    main()
