import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from pd_localization.experiment_loader import (
    load_experiments,
    get_experiment,
    Experiment,
)


def norm(exp: Experiment):
    chs = exp.ch_names
    voltages = []
    for ch in chs:
        v = np.abs(exp.voltage(ch))
        v /= np.max(v)
        voltages.append(v)
    return np.array(voltages)


def prim_pico(v: np.ndarray, limiar=0.1):
    cond = np.zeros_like(v)
    for i in range(len(v) - 1):
        if v[i] > limiar and v[i] > v[i + 1]:
            cond[i] = 1.0
            break
    return np.argmax(cond)


experiments = load_experiments()
exp = get_experiment(experiments, "antena1", 4)
voltages = norm(exp)
t = exp.time()

fig, axs = plt.subplots(4, 1, sharex=True)
for i in range(4):
    v = voltages[i]
    idx = prim_pico(v)
    axs[i].plot(t, v, c=f"C{i}")
    axs[i].axhline(0.1, c="k", ls="--", lw=1)
    axs[i].scatter(t[idx], v[idx], c="k", label="primeiro pico")
    axs[i].legend()
plt.show()
