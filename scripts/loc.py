import numpy as np
import matplotlib.pyplot as plt
from pd_localization.experiment_loader import (
    load_experiments,
)
from pd_localization.preprocessing import (
    cum_energy_exp,
    curv_detection,
    exp_filter,
    norm_zero_mean,
)
from pd_localization.localizacao import (
    batch_localization,
    batch_tau_estimates,
    tau_estim_dtw,
    tau_estim_energy,
    tau_estim_gcc,
    true_taus,
    tau_estim_sff,
)
from pd_localization.results import (
    plot_location_scatter,
    mean_estimates,
    to_dataframe,
    remove_outliers,
    print_summary,
)
from pd_localization.gcc import gcc_phat, gcc_phat_p, sff_gcc
from pd_localization.dtw import dist

DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
experiments = load_experiments(DATA_PATH)
MODE = "2d"

# # ---------- DTW ----------
# dtw_estimator = tau_estim_dtw(cum_energy_exp, curv_detection, dist)
# dtw_taus = batch_tau_estimates(experiments, dtw_estimator)
# dtw_results = batch_localization(experiments, dtw_taus, mode=MODE)
# print("----------------------------DTW------------------------")
# # pos_mean = mean_estimates(remove_outliers(to_dataframe(dtw_results)))
# pos_mean = mean_estimates(to_dataframe(dtw_results))
# print(pos_mean)
# plot_location_scatter(dtw_results)
# plt.show()
# plt.bar(pos_mean["exp_antena"], pos_mean["abs_error(m)"])
# plt.xlabel("posições")
# plt.ylabel("erro médio absoluto (m)")
# plt.show()

# ---------- SFF-GCC ----------
sff_estimator = tau_estim_sff(norm_zero_mean, sff_gcc)
sff_taus = batch_tau_estimates(experiments, sff_estimator)
sff_results = to_dataframe(batch_localization(experiments, sff_taus, mode=MODE))
print("-----------------------------GCC-----------------------")
pos_mean = mean_estimates(sff_results)
print(pos_mean)
# plot_location_scatter(sff_results, only_mean_position=False)
# plt.grid(True)
# plt.show()

plt.bar(pos_mean["exp_antena"], pos_mean["abs_error(m)"])
plt.xlabel("posições")
plt.ylabel("erro médio absoluto (m)")
plt.show()
