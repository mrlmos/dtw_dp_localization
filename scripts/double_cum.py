import matplotlib.pyplot as plt
import numpy as np
from pd_localization.experiment_loader import (
    load_experiments,
    get_experiment,
    Experiment,
)


def sungle_cum(exp: Experiment):
    chs = ["CH1", "CH2", "CH3", "CH4"]
    voltages = np.array([exp.voltage(c) for c in chs])
    dc = np.zeros_like(voltages)
    for i, v in enumerate(voltages):
        cum_sum = np.cumsum(v**2)
        cum_sum /= cum_sum.max()
        dc[i, :] = cum_sum
    return dc


def double_cum(exp: Experiment):
    chs = ["CH1", "CH2", "CH3", "CH4"]
    voltages = np.array([exp.voltage(c) for c in chs])
    dc = np.zeros_like(voltages)
    for i, v in enumerate(voltages):
        cum_sum = np.cumsum(np.abs(v))
        cum_sum /= cum_sum.max()
        dc[i, :] = cum_sum
    return dc


PATH = "/home/murilo/dev/python/tcc/dados_dp/"
antenas = ["antena1"]
experimentos = load_experiments(PATH, antenas)
exp = get_experiment(experimentos, "antena1", 0)
t = exp.time()

sc = sungle_cum(exp)
dc = double_cum(exp)
print(dc.shape)

fig, axs = plt.subplots(1, 2)
for i, (vs, vd) in enumerate(zip(sc, dc)):
    axs[0].plot(t, vs, label=f"antena {i + 1}")
    axs[1].plot(t, vd, label=f"antena {i + 1}")
axs[0].legend()
axs[1].legend()
plt.show()
