import matplotlib.pyplot as plt
import numpy as np
from icecream import ic


def dist(x, y):
    return (x - y) ** 2


def gen_cost_matrix(N, M, dist, return_cases=False):
    """
    Builds the DTW accumulated cost matrix for two sequences of lengths N and M.

    Uses dynamic programming: each cell (i, j) holds the local distance between
    x1[i] and x2[j] plus the minimum accumulated cost from its three predecessors
    (diagonal, top, left). Border conditions are set to infinity to enforce valid
    alignment paths.

    Parameters
    ----------
    N : int
        Length of the first sequence.
    M : int
        Length of the second sequence.
    dist : callable
        Distance function that takes two scalars and returns a float.
    return_cases : bool, optional
        If True, also returns the list of predecessor choices made at each cell.
        Default is False.

    Returns
    -------
    D : np.ndarray of shape (N, M)
        Accumulated cost matrix.
    case : list of int, optional
        Flat list (row-major) of predecessor indices for each cell:
        0 = diagonal (i-1, j-1), 1 = top (i-1, j), 2 = left (i, j-1).
        Only returned when return_cases=True.
    """
    D = np.zeros((N + 1, M + 1))
    case = []
    D[1:, 0] = np.inf
    D[0, 1:] = np.inf

    for i in range(N):
        for j in range(M):
            D[i + 1, j + 1] = dist(x1[i], x2[j])
            aux = [D[i, j], D[i, j + 1], D[i + 1, j]]
            D[i + 1, j + 1] += np.min(aux)
            idx = np.argmin(aux)
            case.append(int(idx))

    if return_cases:
        return D[1:, 1:], case
    else:
        return D[1:, 1:]


def dtw(x1, x2, dist, return_path=True):
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
    N = len(x1)
    M = len(x2)

    D, case = gen_cost_matrix(N, M, dist, return_cases=True)

    path = []
    pos = [N - 1, M - 1]
    path.append(pos.copy())
    while pos[0] != 0 or pos[1] != 0:
        idx = pos[0] * M + pos[1]
        if case[idx] == 0:
            pos[0] -= 1
            pos[1] -= 1
        elif case[idx] == 1:
            pos[0] -= 1
        elif case[idx] == 2:
            pos[1] -= 1

        path.append(pos.copy())

    if return_path:
        return np.array(path), D[-1, -1]

    return D[-1, -1]


def gabarito_sakoe_chiba(sz1, sz2, radius=1):
    mask = np.zeros((sz1, sz2), dtype=bool)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in range(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = True
    else:
        width = sz2 - sz1 + radius
        for i in range(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = True
    return mask


def main():
    print("Hello from dtw!")


if __name__ == "__main__":
    main()
