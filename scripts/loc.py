import time
import numpy as np
import matplotlib.pyplot as plt
from pd_localization.experiment_loader import (
    load_experiments,
)
from pd_localization.preprocessing import (
    cum_energy_exp,
    curv_detection,
    norm_zero_mean,
)
from pd_localization.localizacao import (
    batch_localization,
    batch_tau_estimates,
    tau_estim_dtw,
    tau_estim_sff,
)
from pd_localization.results import (
    plot_location_scatter,
    mean_estimates,
    to_dataframe,
)
from pd_localization.gcc import sff_gcc
from pd_localization.dtw import dist


DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
experiments = load_experiments(DATA_PATH)
MODE = "3d"

# ---------- DTW ----------
dtw_estimator = tau_estim_dtw(cum_energy_exp, curv_detection, dist, radius=14)
start = time.time()
dtw_taus = batch_tau_estimates(experiments, dtw_estimator)
dtw_results = batch_localization(experiments, dtw_taus, mode=MODE)
print("----------------------------DTW------------------------")
pos_mean_dtw = mean_estimates(to_dataframe(dtw_results))
end = time.time()
print(f"Excecution time: {end - start}s\n")
tempo_dtw = end - start
print(pos_mean_dtw)
# plot_location_scatter(dtw_results)
# plt.show()
# plt.bar(pos_mean["exp_antena"], pos_mean["abs_error(m)"])
# plt.xlabel("posições")
# plt.ylabel("erro médio absoluto (m)")
# plt.show()

print("-----------------------------GCC-----------------------")
sff_estimator = tau_estim_sff(norm_zero_mean, sff_gcc)
start = time.time()
sff_taus = batch_tau_estimates(experiments, sff_estimator)
sff_results = to_dataframe(batch_localization(experiments, sff_taus, mode=MODE))
pos_mean_gcc = mean_estimates(sff_results)
end = time.time()
print(f"Excecution time: {end - start}s\n")
time_gcc = end - start
print(pos_mean_gcc)
# plot_location_scatter(sff_results, only_mean_position=False)
# plt.grid(True)
# plt.show()
#
# plt.bar(pos_mean["exp_antena"], pos_mean["abs_error(m)"])
# plt.xlabel("posições")
# plt.ylabel("erro médio absoluto (m)")
# plt.show()


# ---------- STATS ----------

width = 0.4
x = np.arange(len(pos_mean_dtw))

plt.rcParams["font.size"] = 12
plt.bar(
    x - width / 2, pos_mean_dtw["abs_error(m)"], width, label=f"DTW: {tempo_dtw:.2f}sec"
)
plt.bar(
    x + width / 2, pos_mean_gcc["abs_error(m)"], width, label=f"GCC: {time_gcc:.2f}sec"
)

plt.xticks(x, pos_mean_dtw["exp_antena"])  # class labels centered between the pair
plt.ylabel("erro absoluto (m)")
plt.title(f"Localização {MODE}")
plt.legend()
plt.show()
