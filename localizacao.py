import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from experiment_loader import load_experiments, get_experiment, Experiment


ANTENAS = {
    1: np.array([0.0, 0.0, 0.97]),
    2: np.array([2.0, 0.0, 0.77]),
    3: np.array([2.0, 2.0, 0.57]),
    4: np.array([0.0, 2.0, 0.47]),
}


OURCE_POSITIONS = {
    "antena1": np.array([0.5, 0.5, 0.62]),
    "antena2": np.array([1.5, 0.5, 0.62]),
    "antena3": np.array([1.5, 1.5, 0.62]),
    "antena4": np.array([0.5, 1.5, 0.62]),
}

v_e = 3e8

path = "/home/murilo/dev/python/tcc/dados_dp/"
experimentos = load_experiments(base_dir=path, antennas=None)

exp = get_experiment(experimentos, "antena2", 0)
ts = 1 / exp.sample_rate
erro_osci = 0.4e-9
tau = {12: 7.0 * ts + erro_osci, 13: 7.0 * ts + erro_osci, 14: 15.0 * ts + erro_osci}


def tdoa3d(vars):
    x, y, z, t1 = vars
    f1 = (
        (x - ANTENAS[1][0]) ** 2
        + (y - ANTENAS[1][1]) ** 2
        + (z - ANTENAS[1][2]) ** 2
        - (v_e * t1) ** 2
    )
    f2 = (
        (x - ANTENAS[2][0]) ** 2
        + (y - ANTENAS[2][1]) ** 2
        + (z - ANTENAS[2][2]) ** 2
        - ((v_e * t1) - tau[12]) ** 2
    )
    f3 = (
        (x - ANTENAS[3][0]) ** 2
        + (y - ANTENAS[3][1]) ** 2
        + (z - ANTENAS[3][2]) ** 2
        - ((v_e * t1) - tau[13]) ** 2
    )
    f4 = (
        (x - ANTENAS[4][0]) ** 2
        + (y - ANTENAS[4][1]) ** 2
        + (z - ANTENAS[4][2]) ** 2
        - ((v_e * t1) - tau[14]) ** 2
    )
    return [f1, f2, f3, f4]


def tdoa2d(vars):
    x, y, t1 = vars
    f1 = (x - ANTENAS[1][0]) ** 2 + (y - ANTENAS[1][1]) ** 2 - (v_e * t1) ** 2
    f2 = (
        (x - ANTENAS[2][1]) ** 2
        + (y - ANTENAS[2][2]) ** 2
        - ((v_e * t1) - tau[12]) ** 2
    )
    f3 = (
        (x - ANTENAS[3][1]) ** 2
        + (y - ANTENAS[3][2]) ** 2
        - ((v_e * t1) - tau[13]) ** 2
    )
    return [f1, f2, f3]


t_mid = np.sqrt(2) / v_e
p0_3d = [1.5, 0.5, 0.6, t_mid]
p0_2d = [1.5, 0.5, t_mid]

dim_2 = True
if dim_2:
    sol, info, ier, msg = fsolve(tdoa2d, p0_2d, full_output=True)
    if ier == 1:
        x_sol, y_sol, t1_sol = sol
        print("--- Convergence Successful ---")
        print(f"Coordinates of DP source (x, y): ({x_sol:.4f}, {y_sol:.4f}) meters")
        print(f"Time of arrival at Sensor 1 (t1): {t1_sol:.4e} seconds")
    else:
        print("Solution did not converge.")
        print("Details:", msg)
else:
    sol, info, ier, msg = fsolve(tdoa3d, p0_3d, full_output=True)
    if ier == 1:
        x_sol, y_sol, z_sol, t1_sol = sol
        print("--- Convergence Successful ---")
        print(
            f"Coordinates of DP source (x, y, z): ({x_sol:.4f}, {y_sol:.4f}, {z_sol:.4f}) meters"
        )
        print(f"Time of arrival at Sensor 1 (t1): {t1_sol:.4e} seconds")
    else:
        print("Solution did not converge.")
        print("Details:", msg)
