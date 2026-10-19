import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pd_localization.experiment_loader import load_experiments
from pd_localization.preprocessing import curv_detection, cum_energy_exp
from pd_localization.localizacao import (
    batch_localization,
    tau_estim_dtw,
    batch_tau_estimates,
    true_taus,
)
from pd_localization.results import mean_estimates, to_dataframe


def dist(x, y):
    return (x - y) ** 2


def dtw_process(experiments, r) -> pd.DataFrame:
    dtw_estimator = tau_estim_dtw(cum_energy_exp, curv_detection, dist, radius=r)
    taus_est = batch_tau_estimates(experiments, dtw_estimator)
    dtw_results = batch_localization(experiments, taus_est)
    mean_res = mean_estimates(to_dataframe(dtw_results))
    return taus_est, mean_res


def get_all_taus(exps):
    ttaus = []
    for exp in exps:
        ttaus.append(true_taus(exp))
    return ttaus


def compare_taus(taus: list[dict], ttaus: list[dict]):
    N = len(taus)
    means = []
    for i in range(N):
        est = np.array(list(taus[i].values()))
        ref = np.array(list(ttaus[i].values()))
        means.append(np.mean(np.abs(est - ref)))
    return np.mean(means) / 4e-10


r_vals = np.arange(12, 30)
means_dist = []
means_samples = []
experiments = load_experiments()
for i, r in enumerate(r_vals):
    taus, df = dtw_process(experiments, r)
    means_dist.append(df["abs_error(m)"].mean())
    means_samples.append(compare_taus(taus, true_taus(experiments)))
    print(f"r = {r} | erro = {df['abs_error(m)'].mean()}\n")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(r_vals, means_dist, marker="o", ls="--")
ax1.set_xlabel("raio sakoe chiba")
ax1.set_ylabel("erro médio absoluto (m)")
ax1.set_title("Erro de localização")

ax2.plot(r_vals, means_samples, marker="o", ls="--", c="C1")
ax2.set_xlabel("raio sakoe chiba")
ax2.set_ylabel("erro médio absoluto (amostras)")
ax2.set_title("Erro de TDOA")
plt.show()
