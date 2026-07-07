import numpy as np
import matplotlib.pyplot as plt
from pd_localization.experiment_loader import (
    load_experiments,
    get_experiment,
    Experiment,
)
from pd_localization.preprocessing import (
    cum_energy_exp,
    deriv_estimation,
    curv_detection,
)
from pd_localization.localizacao import batch_localization
from pd_localization.dtw import dist


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"


# x1 = x1(t)
# x2 = x1(t - tau) + s(t)
def main():
    experiments = load_experiments(DATA_PATH)
    exp = get_experiment(experiments, "antena4", 1)
    voltages = cum_energy_exp(exp)
    v = voltages[1]

    x = np.arange(len(v))
    y1 = 0.3
    y2 = 0.6
    A = np.argmax(v >= y1)
    B = np.argmax(v >= y2)
    m = (y2 - y1) / (B - A)
    b = y1 - m * A
    line = x * m + b
    o = np.argmax(line >= 0)
    # o = curv_detection(v, 0)
    z = curv_detection(v, 0)
    # ANTES         DEPOIS
    # 0.168267    0.092392
    # 0.043849    0.110097 piorou, mas ficou mais consistente com o joelho em todas
    # 0.042416    0.041271
    # 0.206573    0.212912
    plt.plot(x, v)
    plt.plot(x, line, "--", alpha=0.7)
    plt.scatter(o, v[o], facecolors="none", edgecolors="k", s=100, label="zero virtual")
    plt.scatter(z, v[z], facecolors="none", edgecolors="r", s=80, label="curvatura")
    plt.annotate("A", (A, y1))
    plt.annotate("B", (B, y2))
    plt.ylim(-0.1, 1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
