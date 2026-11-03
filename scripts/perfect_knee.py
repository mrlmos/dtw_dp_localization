from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from pd_localization.localizacao import (
    batch_localization,
    batch_tau_estimates,
    tau_estim_dtw,
)
from pd_localization.results import mean_estimates, to_dataframe
from pd_localization.test_signals import generate_pd
from pd_localization.dtw import gen_cost_matrix, backtracking, dist
from pd_localization.preprocessing import curv_detection, cum_energy_exp
from pd_localization.experiment_loader import (
    load_experiments,
    get_experiment,
    Experiment,
)

# ------------------------------
# FACTORIES
# ------------------------------


def estim_dtw_base(
    base_signal: np.ndarray,
    pre_processing: Callable,
    ref_estimator: Callable,
    distance_func: Callable,
    radius: int = 24,
) -> Callable:
    refs = {"antena1": 0, "antena2": 1, "antena3": 2, "antena4": 3}

    # Passos:
    # 1. Fazer o dtw entre o sinal base e cada um dos 4
    # 2. Recolher todos os índices de taus entre base-sinal
    # 3. Subtrair todo com o valor do tau do sinal de referência
    def _estimate(exp: Experiment) -> dict:
        voltages = pre_processing(exp)
        ref_index = refs[exp.antenna]
        ref_arrival = ref_estimator(base_signal)

        ch_names = [f"CH{i + 1}" for i in range(len(voltages))]

        base_taus = {}
        for i, (v, ch) in enumerate(zip(voltages, ch_names)):
            D = gen_cost_matrix(base_signal, v, distance_func, radius=radius)
            path = backtracking(D)[::-1]
            linked = [pj for pi, pj in path if pi == ref_arrival]
            linked_diffs = np.abs(base_signal[ref_arrival] - v[linked])
            estimative = np.abs(np.argmin(linked_diffs) + linked[0] - ref_arrival)
            base_taus[ch] = estimative / exp.sample_rate

        base_value = base_taus[f"CH{ref_index + 1}"]
        taus = {ch: tau - base_value for ch, tau in base_taus.items()}
        return taus

    return _estimate


def cum_sum(x: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(x**2)
    return cdf / cdf.max()


def estim_dtw(cdf, r=13):
    dtw_estim = estim_dtw_base(cdf, cum_energy_exp, curv_detection, dist, radius=r)
    dtw_taus = batch_tau_estimates(exps, dtw_estim)
    dtw_results = batch_localization(exps, dtw_taus)
    return mean_estimates(to_dataframe(dtw_results))


def estim_dtw_old(r=13):
    dtw_estim = tau_estim_dtw(cum_energy_exp, curv_detection, dist, radius=r)
    dtw_taus = batch_tau_estimates(exps, dtw_estim)
    dtw_results = batch_localization(exps, dtw_taus)
    return mean_estimates(to_dataframe(dtw_results))


if __name__ == "__main__":
    antena = "antena1"
    exps = load_experiments()
    exp = get_experiment(exps, antena, 0)
    v = cum_sum(exp.voltage("CH1"))

    tamos = 1 / exp.sample_rate
    t = exp.time()
    idx = np.isclose(t, 0.0, atol=tamos).nonzero()[0][0]
    t_0 = idx * tamos  # 40ns

    taus_sample = np.array([0, 10, 15, 30])
    taus = taus_sample * tamos + t_0

    signals = generate_pd(t_0, 1, snr=None, beta=12e-10)
    cdf = cum_sum(signals)

    pos_mean = estim_dtw(cdf)
    print(pos_mean)

    pos_mean_old = estim_dtw_old()
    print(pos_mean_old)
