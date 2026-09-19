import numpy as np
import matplotlib.pyplot as plt

from pd_localization.dtw import plot_dtw, dist, dtw, gen_cost_matrix, my_sakoe_chiba
from pd_localization.test_signals import generate_pd
from pd_localization.gcc import gcc_phat, gcc_scot


def akaike_info(v):
    eic = np.zeros_like(v)
    for k in range(len(v)):
        var1 = np.std(v[: k + 1]) ** 2
        var2 = np.std(v[k:]) ** 2
        eval = k * np.log(var1 + 1e-9) + (len(v) - k - 1) * np.log(var2 + 1e-9)
        eic[k] = eval
    return eic


def energy_criterion(signal):
    v = signal**2
    px = np.mean(v)
    ec = np.zeros_like(v)
    for k in range(len(v)):
        ec[k] = np.sum(v[:k]) - k * px
    return ec


def cum_energy(signal):
    sqrd = np.cumsum(signal**2)
    out = sqrd / np.max(sqrd)
    return out


betas = np.array([0.3, 1.2, 1.8, 2.7]) * 1e-9
tamos = 0.4e-9
tau_true = 40
t_0 = 100 * tamos  # 40ns
t = np.arange(0, tamos * 250, tamos) * 50e7


s1 = generate_pd(t_0, 1, snr=10)
c1 = energy_criterion(s1)
idx1 = int(np.argmin(c1))

s2 = generate_pd(t_0 + 40 * tamos, 1, snr=10, beta=betas[2])
c2 = energy_criterion(s2)
idx2 = int(np.argmin(c2))

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(t, c1, c="b", label="sensor 1")
ax.plot(t, c2, c="r", label="sensor 2")
ax.scatter(t[idx1], c1[idx1], facecolors="none", edgecolors="k", s=100)
ax.scatter(t[idx2], c2[idx2], facecolors="none", edgecolors="k", s=100)
ax.axvspan(t[idx1], t[idx2], color="gray", alpha=0.3)
ax.set_xlabel("Tempo (ns)")

# --- TDOA Annotation ---
y_arrow = float(-1.0)
x1, x2 = float(t[idx1]), float(t[idx2])
x_mid = (x1 + x2) / 2.0

# 1. Double-headed arrow across the interval
ax.annotate(
    "",
    xy=(x1, y_arrow),
    xytext=(x2, y_arrow),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1),
)

# 2. "TDOA" text with arrow pointing to the midpoint
# Using offset points (integers) avoids the float deprecation warning
ax.annotate(
    "TDOA",
    xy=(x_mid, y_arrow),
    xytext=(-50, 30),  # Offset in points: -15pt left, +30pt up (both ints)
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->",
        color="black",
        lw=0.8,
        connectionstyle="arc3,rad=-0.2",
    ),
    ha="center",
    fontsize=14,
)
# -----------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(True, alpha=0.4)
plt.legend()
plt.savefig("../draft_2/medias/exemplo_akaike.png")
