import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pd_localization.geral import (
    mean_loc_cum,
    mean_loc_dtw,
    mean_loc_gcc_phat,
    mean_loc_gcc_sff,
)
from pd_localization.experiment_loader import load_experiments

MODE = "3d"
experiments = load_experiments()

dtw = mean_loc_dtw(experiments, mode=MODE, radius=13)
phat = mean_loc_gcc_phat(experiments, mode=MODE)
sff = mean_loc_gcc_sff(experiments, mode=MODE)
energy = mean_loc_cum(experiments, mode=MODE)

estims = {
    "DTW": dtw["abs_error(m)"],
    "GCC-PHAT": phat["abs_error(m)"],
    "GCC-SFF": sff["abs_error(m)"],
    "Energy": energy["abs_error(m)"],
}
antenas = {
    "antena1": 0,
    "antena2": 1,
    "antena3": 2,
    "antena4": 3,
}

# Setup plot parameters
antena_labels = list(antenas.keys())
estimator_names = list(estims.keys())

x = np.arange(len(antena_labels))  # Base X locations for antennas (0, 1, 2, 3)
num_estimators = len(estimator_names)
width = 0.18  # Bar width

plt.rcParams["font.size"] = 12
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each estimator's error values per antenna
for i, (est_name, error_series) in enumerate(estims.items()):
    # Extract values in antenna index order (0 to 3)
    values = [error_series[antenas[ant]] for ant in antena_labels]

    # Offset each set of bars centered around the tick label
    offset = x + (i - (num_estimators - 1) / 2) * width
    ax.bar(offset, values, width, label=est_name)

# Formatting
ax.set_ylabel("erro absoluto (m)")
ax.set_title(f"Localização {MODE}")
ax.set_xticks(x)
ax.set_xticklabels(antena_labels)
ax.legend()

plt.tight_layout()
plt.show()
