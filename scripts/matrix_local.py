import matplotlib.pyplot as plt
import numpy as np

from pd_localization.experiment_loader import load_experiments, get_experiment
from pd_localization.preprocessing import (
    cum_energy_exp,
    deriv_estimation,
)
from pd_localization.localizacao import (
    estimate_taus_dtw,
    ANTENNA_POSITIONS,
    SOURCE_POSITIONS,
)

v_e = 3e8


def dist(x, y):
    return (x - y) ** 2


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"


def main():
    experiments = load_experiments(DATA_PATH)
    exp = get_experiment(experiments, "antena3", 5)
    ref_index = 2
    voltages = cum_energy_exp(exp)
    taus = estimate_taus_dtw(
        voltages,
        ref_index,
        deriv_estimation(voltages[ref_index], 0),
        exp.sample_rate,
        exp.travel_times,
        distance_func=dist,
    )

    ref = "CH3"
    x_ref, y_ref = ANTENNA_POSITIONS[ref][:2]
    K_ref = x_ref**2 + y_ref**2
    xi = np.array([])
    yi = np.array([])

    G = []
    h = []

    for ch, tau in taus.items():
        if ch == ref:
            continue

        xi, yi = ANTENNA_POSITIONS[ch][:2]

        xi_ref = xi - x_ref
        yi_ref = yi - y_ref

        Ki = xi**2 + yi**2

        ri_ref = tau * v_e

        G.append(
            [
                -xi_ref,
                -yi_ref,
                -ri_ref,
            ]
        )

        h.append(0.5 * (ri_ref**2 - Ki + K_ref))

    G = np.asarray(G)
    h = np.asarray(h)

    print(G.shape)
    x, y, r1 = np.linalg.solve(G, h)
    print("x pos:", x)
    print("y pos: ", y)
    print("taus =", taus)

    for ch, pos in ANTENNA_POSITIONS.items():
        d = np.linalg.norm(SOURCE_POSITIONS["antena3"][:2] - pos[:2])
        print(ch, d)


if __name__ == "__main__":
    main()
