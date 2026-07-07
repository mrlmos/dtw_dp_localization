import numpy as np
import matplotlib.pyplot as plt
from pd_localization.experiment_loader import (
    load_experiments,
)
from pd_localization.preprocessing import (
    cum_energy_exp,
    curv_detection,
    akaike_info,
    energy_criterion,
    norm_zero_mean,
)
from pd_localization.localizacao import (
    batch_localization,
    batch_tau_estimates,
    tau_estim_dtw,
    tau_estim_energy,
    tau_estim_gcc,
    true_taus,
)
from pd_localization.gcc import gcc_roth, gcc_ht, gcc_phat, gcc_scot
from pd_localization.dtw import dist

REF_CHANNEL = {
    "antena1": "CH1",
    "antena2": "CH2",
    "antena3": "CH3",
    "antena4": "CH4",
}

DATA_PATH = "/home/murilo/dev/python/tcc/dados_dp/"
experiments = load_experiments(DATA_PATH)
target = true_taus(experiments)

# ---------- DTW ----------
dtw_estimator = tau_estim_dtw(cum_energy_exp, curv_detection, dist)
dtw_taus = batch_tau_estimates(experiments, dtw_estimator)

# ---------- GCC ----------
gcc_estimator = tau_estim_gcc(norm_zero_mean, gcc_phat)
gcc_taus = batch_tau_estimates(experiments, gcc_estimator)

# ---------- AKAIKE ----------
akaike_estimator = tau_estim_energy(akaike_info, np.min)
akaike_taus = batch_tau_estimates(experiments, akaike_estimator)

# ---------- ENERGY CRITERION ----------
energy_estimator = tau_estim_energy(energy_criterion, np.min)
energy_taus = batch_tau_estimates(experiments, energy_estimator)

# ---------- CUMULATIVE ENERGY ----------
cum_estimator = tau_estim_energy(cum_energy_exp, curv_detection)
cum_taus = batch_tau_estimates(experiments, cum_estimator)

# ------------------------------
#           PLOTS
# ------------------------------
estimators = {
    "DTW": dtw_taus,
    "GCC-PHAT": gcc_taus,
    "Akaike": akaike_taus,
    "Energia + curv": cum_taus,
}

ref_channel = {
    "antena1": "CH1",
    "antena2": "CH2",
    "antena3": "CH3",
    "antena4": "CH4",
}

channels = ["CH1", "CH2", "CH3", "CH4"]

bar_width = 0.16

for antenna in ("antena1", "antena2", "antena3", "antena4"):
    idxs = [i for i, exp in enumerate(experiments) if exp.antenna == antenna]

    x = np.arange(len(idxs))

    plot_channels = [c for c in channels if c != ref_channel[antenna]]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    fig.suptitle(f"Erro absoluto - {antenna} (1 sample = 0.4ns)", fontsize=14)

    offsets = (np.arange(len(estimators)) - (len(estimators) - 1) / 2) * bar_width

    for ax, ch in zip(axes, plot_channels):
        true = np.array([target[i][ch] for i in idxs])

        for offset, (name, estimates) in zip(offsets, estimators.items()):
            est = np.array([estimates[i][ch] for i in idxs])

            error = np.abs(est - true) * 1e9

            ax.bar(
                x + offset,
                error,
                width=bar_width,
                label=name,
            )

        ax.set_title(ch)
        ax.set_ylabel("Erro (ns)")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(np.arange(1, len(idxs) + 1))
    axes[-1].set_xlabel("Experimento")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        # bbox_to_anchor=(1.02, 0.5),
    )

plt.show()


## BOX PLOT
locations = ["antena1", "antena2", "antena3", "antena4"]

means = {name: [] for name in estimators}

for antenna in locations:
    idxs = [i for i, exp in enumerate(experiments) if exp.antenna == antenna]

    ref = REF_CHANNEL[antenna]

    for name, estimates in estimators.items():
        errors = []

        for i in idxs:
            for ch in channels:
                if ch == ref:
                    continue

                errors.append(abs(estimates[i][ch] - target[i][ch]) * 1e9)

        means[name].append(np.mean(errors))


fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(locations))
width = 0.15

offsets = (np.arange(len(estimators)) - (len(estimators) - 1) / 2) * width

for offset, (name, vals) in zip(offsets, means.items()):
    ax.bar(x + offset, vals, width, label=name)

ax.set_xticks(x)
ax.set_xticklabels(locations)

ax.set_ylabel("Erro médio (ns)")
ax.set_title("Erro médio por fonte DP")
ax.legend()

plt.tight_layout()
plt.show()
