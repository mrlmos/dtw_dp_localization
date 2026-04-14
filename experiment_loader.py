import os
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    antenna: str  # e.g. "antena1"
    index: int  # 0-based index within antenna
    channels: dict = field(default_factory=dict)  # {"CH1": pd.DataFrame, ...}
    metadata: dict = field(default_factory=dict)  # header info from CSV

    @property
    def label(self):
        return f"{self.antenna} / exp {self.index + 1:02d}"

    @property
    def sample_rate(self) -> Optional[float]:
        """Sample rate in Hz, parsed from metadata if available."""
        si = self.metadata.get("Sample Interval")
        return 1.0 / float(si) if si else None

    @property
    def ch_names(self):
        return list(self.channels.keys())

    def time(self, ch="CH1") -> np.ndarray:
        """Return time array for a channel as numpy array."""
        return self.channels[ch]["TIME"].values

    def voltage(self, ch="CH1") -> np.ndarray:
        """Return voltage array for a channel as numpy array."""
        return self.channels[ch]["VOLTAGE"].values

    def plot(self, channels=None, figsize=(14, 8)):
        """Quick-look plot of all (or selected) channels."""
        chs = channels or self.ch_names
        fig, axes = plt.subplots(len(chs), 1, figsize=figsize, sharex=True)
        if len(chs) == 1:
            axes = [axes]
        colors = {
            "CH1": "#1f77b4",
            "CH2": "#ff7f0e",
            "CH3": "#2ca02c",
            "CH4": "#d62728",
        }
        fig.suptitle(self.label, fontsize=13, fontweight="bold")
        for ax, ch in zip(axes, chs):
            ax.plot(
                self.time(ch),
                self.voltage(ch),
                color=colors.get(ch, "black"),
                linewidth=1,
                label=ch,
            )
            ax.set_ylabel("Voltage (V)", fontsize=8)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
        axes[-1].set_xlabel("Time (s)", fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig


def _read_metadata(filepath) -> dict:
    """Parse the key-value metadata from the first ~20 lines."""
    meta = {}
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i >= 21:
                break
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] and parts[1]:
                meta[parts[0].strip()] = parts[1].strip()
    return meta


def _read_signal(filepath) -> pd.DataFrame:
    """Read time/voltage data (starts at line 22)."""
    df = pd.read_csv(filepath, skiprows=21, header=0, names=["TIME", "VOLTAGE"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def _group_experiments(csv_files: list) -> list:
    """
    Sort files and group into sets of 4 channels (CH1–CH4).
    Returns list of lists: [[('CH1', path), ('CH2', path), ...], ...]
    """
    pattern = re.compile(r"tek(\d+)(CH\d+)\.csv", re.IGNORECASE)
    parsed = []
    for f in csv_files:
        m = pattern.match(os.path.basename(f))
        if m:
            parsed.append((int(m.group(1)), m.group(2).upper(), f))

    parsed.sort(key=lambda x: (x[0], x[1]))

    groups = []
    for i in range(0, len(parsed) - 3, 4):
        block = parsed[i : i + 4]
        if len(block) == 4:
            groups.append([(ch, path) for _, ch, path in block])
    return groups


def load_experiments(base_dir=".", antennas=None) -> list:
    """
    Load all experiments from antenna subfolders.

    Parameters
    ----------
    base_dir : str
        Root directory containing antena1/, antena2/, etc.
    antennas : list of str, optional
        Which antenna folders to load. Defaults to all found.

    Returns
    -------
    list of Experiment
    """
    if antennas is None:
        antennas = sorted(
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("antena")
        )

    experiments = []

    for antenna in antennas:
        folder = os.path.join(base_dir, antenna)
        csv_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
        groups = _group_experiments(csv_files)

        for idx, group in enumerate(groups):
            channels = {}
            metadata = {}
            for ch, filename in group:
                filepath = os.path.join(folder, os.path.basename(filename))
                if not metadata:  # read metadata once per experiment
                    metadata = _read_metadata(filepath)
                channels[ch] = _read_signal(filepath)

            experiments.append(
                Experiment(
                    antenna=antenna,
                    index=idx,
                    channels=channels,
                    metadata=metadata,
                )
            )
        print(f"{antenna}: loaded {len(groups)} experiment(s)")

    print(f"\nTotal: {len(experiments)} experiments loaded.")
    return experiments


def filter_by_antenna(experiments: list, antenna: str) -> list:
    """Return only experiments from a specific antenna."""
    return [e for e in experiments if e.antenna == antenna]


def get_experiment(experiments: list, antenna: str, index: int) -> Experiment:
    """Fetch a single experiment by antenna name and 0-based index."""
    matches = [e for e in experiments if e.antenna == antenna and e.index == index]
    if not matches:
        raise ValueError(f"No experiment found for {antenna}, index {index}")
    return matches[0]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exps = load_experiments(base_dir=".")

    # --- Access a single experiment ---
    exp = get_experiment(exps, "antena1", 0)
    print(f"\nExperiment: {exp.label}")
    print(f"Channels: {exp.ch_names}")
    print(f"Sample rate: {exp.sample_rate:.2e} Hz")
    print(f"CH1 voltage (first 5 samples): {exp.voltage('CH1')[:5]}")

    # --- Iterate over all antena2 experiments ---
    for e in filter_by_antenna(exps, "antena2"):
        t = e.time("CH1")
        v = e.voltage("CH1")
        print(f"{e.label} | duration: {t[-1] - t[0]:.2e} s | Vpeak: {v.max():.4f} V")

    # --- Quick plot ---
    fig = exp.plot()
    plt.show()
