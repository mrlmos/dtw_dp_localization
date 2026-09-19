import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def confidence_interval(std, N=100):
    return 1.96 * std / np.sqrt(N)


N = 1000
filename = "/home/murilo/dev/python/tcc/dtw/result_sheets/synthetic_benchmark2.csv"

df = pd.read_csv(filename, decimal=",")
results = df[
    [
        "Scenario_ID",
        "Method",
        "Overall_MAE (samples)",
        "Overall_Std (samples)",
    ]
]


scenario_order = [
    "Test_1a",
    "Test_1b",
    "Test_2",
    "Test_3a",
    "Test_3b",
    "Test_4",
]
scenarios = {
    "Test_1a": r"$\beta = 0.6$ | SNR = 15dB",
    "Test_1b": r"$\beta = 2.7$ | SNR = 15dB",
    "Test_2": r"$\beta$ variado | SNR = 15dB",
    "Test_3a": r"$\beta = 0.6$ | SNR variado",
    "Test_3b": r"$\beta = 2.7$ | SNR variado",
    "Test_4": r"$\beta$ variado | SNR variado",
}

m_list = [
    "Akaike",
    "EC",
    "DTW",
    "EC+T",
    # "GCC-HT",
    "PHAT",
    # "GCC-ROTH",
    "SCOT",
]

results["Scenario_ID"] = pd.Categorical(
    results["Scenario_ID"],
    categories=scenario_order,
    ordered=True,
)

results = results.sort_values(["Scenario_ID", "Method"])

mae = {}
std = {}

skip_methods = {"GCC-HT", "GCC-ROTH"}
methods = []

colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
for method in results["Method"].unique():
    if method in skip_methods:
        continue
    methods.append(method)

    subset = results[results["Method"] == method]

    mae[method] = subset["Overall_MAE (samples)"].to_numpy()
    std[method] = subset["Overall_Std (samples)"].to_numpy()


plt.rcParams["font.size"] = 14
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
axs = axs.ravel()
for i in range(6):
    tests = scenarios[scenario_order[i]]
    heights = [mae[m][i] for m in methods]
    errors = [confidence_interval(std[m][i]) for m in methods]

    axs[i].bar(m_list, heights, yerr=errors, color=colors)
    axs[i].set_title(tests)
    axs[i].grid(axis="y", linestyle="--", alpha=0.5)
    axs[i].set_title(list(scenarios.values())[i])
    axs[i].set_xticklabels(m_list, rotation=45)

    axs[i].spines["top"].set_visible(False)
    axs[i].spines["right"].set_visible(False)
    # axs[i].spines["left"].set_visible(False)
    # if i == 0 or i == 3:
    #     axs[i].spines["left"].set_visible(True)


plt.xticks(rotation=45)
fig.supylabel("Erro Médio Absoluto (amostras)")
fig.tight_layout()
plt.savefig("../draft_2/medias/testes_gerais.png")
# plt.show()


# plt.errorbar(
#     tests,
#     mae["DTW"],
#     yerr=std["DTW"] / np.sqrt(N) * 1.96,
#     marker="o",
#     capsize=4,
#     label="DTW",
# )
# plt.show()
