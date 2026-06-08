import numpy as np


def norm1_dist(x, y):
    return np.abs(x - y)


def norm2_dist(x, y):
    return (x - y) ** 2
