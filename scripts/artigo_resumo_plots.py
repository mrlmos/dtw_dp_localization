import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PATH = "/home/murilo/dev/python/tcc/dtw/result_sheets/benchmark_artigo.xlsx"

df = pd.read_excel(PATH, sheet_name="Master_Summary")
s_values = df["Scenario_ID"].unique()
methods = df["Method"].unique().tolist()
print(len(s_values))
scenarios = []

for sv in s_values:
    s = df[df["Scenario_ID"] == sv]
    scenarios.append(s)

fig, axs = plt.subplots(2, 3)

for ax, s, title in zip(axs.reshape(-1), scenarios, s_values):
    var = s["Overall_Std (samples)"].to_numpy()

    yerr = np.where(var > 4, np.nan, var)
    ax.bar(methods, s["Overall_MAE (samples)"], yerr=yerr, capsize=4)
    ax.set_title(title)
    ax.set_ylabel("Erro Médio (amostras)")
    ax.set_ylim(0, 7)
    ax.grid(visible=True, axis="y", alpha=0.5)
    # plt.xticklabels(methods, rotation=45)
plt.show()
