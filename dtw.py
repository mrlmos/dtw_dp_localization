import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from icecream import ic


def dist(x, y):
    return (x - y) ** 2


def gen_cost_matrix(x1, x2, dist, radius=None, log=False):
    """
    Builds the DTW accumulated cost matrix for two sequences of lengths N and M.

    Parameters
    ----------
    x1 : array-like of shape (N,)
        First input sequence.
    x2 : array-like of shape (M,)
        Second input sequence.
    dist : callable
        Distance function that takes two scalars and returns a float.

    Returns
    -------
    D : np.ndarray of shape (N, M)
        Accumulated cost matrix.
    """
    N = len(x1)
    M = len(x2)
    D = np.zeros((N + 1, M + 1))
    if radius is None:
        D[1:, 0] = np.inf
        D[0, 1:] = np.inf
        mask = np.ones((N + 1, M + 1), dtype=bool)
    elif radius >= 0:
        mask = my_sakoe_chiba(N + 1, M + 1, radius=radius)
    else:
        raise ValueError("radius must be positive or None")

    for i in range(N):
        for j in range(M):
            if mask[i, j]:
                D[i + 1, j + 1] = dist(x1[i], x2[j])
                aux = [D[i, j], D[i, j + 1], D[i + 1, j]]
                D[i + 1, j + 1] += np.min(aux)
            else:
                D[i + 1, j + 1] = np.inf

    if log:
        return -np.log(D[1:, 1:])
    return D[1:, 1:]


def dtw(x1, x2, dist, return_path=True, radius=None):
    """
    Computes the Dynamic Time Warping distance and optimal alignment path
    between two sequences.

    Builds the accumulated cost matrix via gen_cost_matrix, then traces back
    the optimal path from (N-1, M-1) to (0, 0) using the stored predecessor
    choices.

    Parameters
    ----------
    x1 : array-like of shape (N,)
        First input sequence.
    x2 : array-like of shape (M,)
        Second input sequence.
    dist : callable
        Distance function that takes two scalars and returns a float.
    path : bool, optional
        If True, computes and returns the optimal warping path alongside the
        DTW cost. If False, returns only the cost. Default is True.

    Returns
    -------
    path : np.ndarray of shape (K, 2)
        Sequence of (i, j) index pairs forming the optimal warping path,
        from (N-1, M-1) to (0, 0).
    cost : float
        Total accumulated DTW cost, given by D[N-1, M-1].
    """
    D = gen_cost_matrix(x1, x2, dist, radius=radius)

    path = backtracking(D)

    if return_path:
        return np.array(path), D[-1, -1]

    return D[-1, -1]


def backtracking(cost_matrix):
    N = cost_matrix.shape[0] - 1
    M = cost_matrix.shape[1] - 1
    path = [[N, M]]

    while N != 0 or M != 0:
        if N == 0:
            M -= 1
        elif M == 0:
            N -= 1
        else:
            direction = np.argmin(
                [
                    cost_matrix[N - 1, M - 1],  # dir = 0 -> diagonal
                    cost_matrix[N - 1, M],  # dir = 1 -> vertical
                    cost_matrix[N, M - 1],  # dir = 2 -> horizontal
                ]
            )
            if direction == 0:
                N -= 1
                M -= 1
            elif direction == 1:
                N -= 1
            elif direction == 2:
                M -= 1
        path.append([N, M])
    return path


def my_sakoe_chiba(N, M, radius=1):
    mask = np.zeros((N, M), dtype=bool)
    for i in range(N):
        lower = max(0, i - radius)
        upper = min(M, i + radius)
        mask[i, lower : upper + 1] = True
    return mask


def mask_print(mask):
    for row in mask:
        print(row)


def pulse(length, center, width, amplitude=1.0, phase=0):
    """
    Generates a Gaussian pulse sequence.

    Parameters
    ----------
    length : int
        Total number of samples in the sequence.
    center : float
        Center position of the pulse in samples.
    width : float
        Standard deviation of the Gaussian, controlling pulse width.
    amplitude : float, optional
        Peak amplitude of the pulse. Default is 1.0.
    phase : float, optional
        Shifts the pulse center by this many samples. Default is 0.

    Returns
    -------
    np.ndarray of shape (length,)
        Sequence starting and ending at ~0 with a Gaussian peak.
    """
    t = np.arange(length)
    return amplitude * np.exp(-0.5 * ((t - center - phase) / width) ** 2)


def plot_dtw(x1, x2, dist, radius=None):
    D = gen_cost_matrix(x1, x2, dist, radius=radius)
    path, cost = dtw(x1, x2, dist, radius=radius)
    path = path[::-1]

    plt.subplot(2, 2, 1)
    plt.plot(x1, label="sinal 1")
    plt.plot(x2, label="sinal 2", linestyle="--")
    plt.title("sinais originais")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x1[path[:, 0]], label="sinal 1")
    plt.plot(x2[path[:, 1]], label="sinal 2", linestyle="--")
    plt.title("sinais alinhados")
    plt.legend()

    plt.subplot(2, 2, (2, 4))
    plt.imshow(np.transpose(D), origin="lower")
    plt.plot(path[:, 0], path[:, 1], color="red", linewidth=2)
    plt.title(f"custo = {cost:.2f} | raio = {radius}")
    plt.xlabel("sinal 1")
    plt.ylabel("sinal 2")


def main():
    plt.rcParams.update({"font.size": 14})
    matplotlib.use("Qt5Agg")

    # t = np.linspace(0, 2 * np.pi, 50)
    # x1 = np.sin(2 * t)
    # x2 = np.sin(2 * t + np.pi / 2)
    defasagem = 15
    x1 = pulse(100, 20, 4) + 0.4 * pulse(100, 30, 2)
    # x1 = 0.1 * np.random.randn(len(x1)) + x1
    x2 = pulse(100, 20, 4, phase=defasagem) + 0.4 * pulse(100, 30, 2, phase=defasagem)
    # x2 = 0.2 * np.random.randn(len(x2)) + x2

    costs = []
    values = range(1, 20)
    values = range(1, len(x1) // 2)
    for r in values:
        path, cost = dtw(x1, x2, dist, radius=r)
        costs.append(cost)

    r = np.argmin(costs) + 1
    r = 2

    path, cost = dtw(x1, x2, dist, radius=r)
    path = path[::-1]
    D = gen_cost_matrix(x1, x2, dist, radius=r)

    # plt.subplot(2, 2, 2)
    # plt.plot(values, costs, label="custo")
    # plt.title("Custo por valor de r")
    # plt.axvline(
    #     defasagem, color="gray", linestyle="--", alpha=0.7, label="alinhamento ideal"
    # )
    # plt.xlabel(r"$r$")
    # plt.ylabel("Custo")
    # plt.legend()

    plt.subplot(2, 2, 1)
    plt.plot(x1, label="sinal 1")
    plt.plot(x2, label="sinal 2")
    plt.title("sinais originais")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x1[path[:, 0]], label="sinal 1")
    plt.plot(x2[path[:, 1]], label="sinal 2")
    plt.title("sinais alinhados")
    plt.legend()

    plt.subplot(2, 2, (2, 4))
    plt.imshow(np.transpose(D), origin="lower")
    plt.plot(path[:, 0], path[:, 1], color="red", linewidth=2)
    plt.title(f"custo = {cost:.2f} | raio = {r} | defasagem = {defasagem}")
    plt.xlabel("sinal 1")
    plt.ylabel("sinal 2")

    plt.show()


if __name__ == "__main__":
    main()
