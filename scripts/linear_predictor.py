import numpy as np
import matplotlib.pyplot as plt
from pd_localization.experiment_loader import (
    load_experiments,
    get_experiment,
    Experiment,
)
from pd_localization.preprocessing import (
    norm_zero_mean,
)


def ln_estimate(x: np.ndarray, p) -> np.ndarray:
    N = len(x)
    Psi = np.column_stack([x[p - k - 1 : N - k - 1] for k in range(p)])
    th = np.linalg.pinv(Psi) @ x[p:]
    ym = Psi @ th
    e = (ym - x[p:]) ** 2
    L = 2
    if L > 0:
        w = np.ones(L) / L
        e = np.convolve(e, w, mode="same")
    return e, ym


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"

experiments = load_experiments(DATA_PATH)
exp = get_experiment(experiments, "antena1", 3)
voltages = norm_zero_mean(exp)
x = voltages[0]
x2 = voltages[2]
p = 5

e_x1, ym1 = ln_estimate(x, p)
e_x2, ym2 = ln_estimate(x2, p)
cc = np.correlate(e_x1, e_x2, mode="full")

plt.subplot(311)
plt.plot(x[p:], "--", label="original")
plt.plot(ym1, label="estimate")
plt.legend()

plt.subplot(312)
plt.plot(x2[p:], "--", label="original")
plt.plot(ym2, label="estimate")
plt.legend()

plt.subplot(313)
plt.plot(cc)
plt.title(np.abs(np.argmax(cc) - len(x[p:]) + 1))
plt.show()
