from experiment_loader import load_experiments, get_experiment
from dtw import dtw, gen_cost_matrix
import matplotlib.pyplot as plt
import numpy as np


def dist(x, y):
    return (x - y) ** 2


def main():
    path = "/home/murilo/dev/python/tcc/dados_dp/"
    experimentos = load_experiments(base_dir=path, antennas=["antena1"])
    exp = get_experiment(experimentos, "antena1", 1)
    # t = exp.time()
    v1 = exp.voltage("CH1")
    v2 = exp.voltage("CH2")

    exp.plot()
    plt.show()

    r = 10

    D = gen_cost_matrix(v1, v2, dist, radius=r)
    path, cost = dtw(v1, v2, dist, radius=r)

    plt.subplot(2, 2, 1)
    plt.plot(v1, label="sinal 1")
    plt.plot(v2, label="sinal 2")
    plt.title("sinais originais")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(v1[path[:, 0]], label="sinal 1")
    plt.plot(v2[path[:, 1]], label="sinal 2")
    plt.title("sinais alinhados")
    plt.legend()

    plt.subplot(2, 2, (2, 4))
    plt.imshow(np.transpose(D), origin="lower")
    plt.plot(path[:, 0], path[:, 1], color="red", linewidth=2)
    plt.title(f"custo = {cost:.2f} | raio = {r}")
    plt.xlabel("sinal 1")
    plt.ylabel("sinal 2")

    plt.show()


if __name__ == "__main__":
    main()
