import os
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

from pd_localization.test_signals import generate_pd
from pd_localization.dtw import backtracking
from pd_localization.preprocessing import curv_detection, min_max_norm
from pd_localization.gcc import gcc_ht, gcc_phat, gcc_roth, gcc_scot

ch_names = [f"CH{i + 1}" for i in range(4)]

# ---------------------------------------------------------
# Global Parameters & Test Configuration
# ---------------------------------------------------------
TAMOS = 0.4e-9  # 0.4 ns sampling interval
T_0 = 150 * TAMOS  # 60 ns reference shift

TAUS_SAMPLE = np.array([0, 10, 15, 30])
TAUS_TRUE = TAUS_SAMPLE * TAMOS + T_0
GAINS = np.array([1.0, 0.5, 0.6, 0.1])

# High SNR (configurable default: 15 dB for all channels)
HIGH_SNR_VAL = 15
HIGH_SNR = np.ones(4) * HIGH_SNR_VAL

# Varying SNR across channels (simulates distance attenuation)
VARYING_SNR = np.array([15, 10, 5, 2])

# Betas (pulse shape parameter)
BETA_06 = np.array([0.6, 0.6, 0.6, 0.6]) * 1e-9
BETA_27 = np.array([2.7, 2.7, 2.7, 2.7]) * 1e-9
BETA_VARYING = np.array([0.6, 1.2, 1.2, 2.7]) * 1e-9

# Repetitions
N_RUNS = 1000


# ---------------------------------------------------------
# High-Performance Fast DTW & Akaike
# ---------------------------------------------------------


def fast_gen_cost_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Optimized DTW cost matrix generation using 1D slice references.
    Provides ~10x speedup over standard 2D index loops.
    """
    N, M = len(x1), len(x2)
    dist_mat = (x1[:, None] - x2[None, :]) ** 2
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, N + 1):
        D_prev = D[i - 1]
        D_curr = D[i]
        dist_row = dist_mat[i - 1]
        for j in range(1, M + 1):
            D_curr[j] = dist_row[j - 1] + min(D_prev[j - 1], D_prev[j], D_curr[j - 1])

    return D[1:, 1:]


def akaike_info(signals: List[np.ndarray]) -> List[np.ndarray]:
    """
    Vectorized Akaike Information Criterion (AIC) using cumulative sum variance updates.
    ~50x speedup over iterative np.std calculation.
    """
    energies = []
    for s in signals:
        n = len(s)
        k = np.arange(1, n)
        cum1 = np.cumsum(s)
        cum2 = np.cumsum(s**2)

        mean1 = cum1[: n - 1] / k
        var1 = (cum2[: n - 1] / k) - mean1**2
        var1 = np.maximum(var1, 1e-9)

        cnt2 = n - k
        mean2 = (cum1[-1] - cum1[: n - 1]) / cnt2
        var2 = ((cum2[-1] - cum2[: n - 1]) / cnt2) - mean2**2
        var2 = np.maximum(var2, 1e-9)

        eic = np.zeros(n)
        eic[1:-1] = k[:-1] * np.log(var1[:-1]) + cnt2[:-1] * np.log(var2[:-1])
        eic[0] = eic[1]
        eic[-1] = eic[-2]
        energies.append(min_max_norm(eic))
    return energies


# ---------------------------------------------------------
# Feature / Transform Functions
# ---------------------------------------------------------


def cum_energy(signals: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for s in signals:
        sum = np.cumsum(s**2)
        out.append(sum / np.max(sum))
    return out


def norm_zero_mean(signals: List[np.ndarray]) -> List[np.ndarray]:
    voltages = []
    for s in signals:
        ch_max = np.max(s)
        ch_min = np.min(s)
        v = (s - ch_min) / (ch_max - ch_min) if (ch_max - ch_min) != 0 else s
        voltages.append(v - np.mean(v))
    return voltages


def energy_criterion(signals: List[np.ndarray]) -> List[np.ndarray]:
    energies = []
    for s in signals:
        v = s**2
        px = np.mean(v)
        ec = np.zeros_like(v)
        for k in range(len(v)):
            ec[k] = np.sum(v[:k]) - k * px
        energies.append(min_max_norm(ec))
    return energies


# ---------------------------------------------------------
# TDOA Estimators
# ---------------------------------------------------------


def estimate_taus_dtw(signals: List[np.ndarray]) -> np.ndarray:
    voltages = norm_zero_mean(signals)
    ref_idx = curv_detection(voltages[0])
    v_ref = voltages[0]
    taus = np.zeros(4)
    taus[0] = 0.0
    for i in range(1, 4):
        v = voltages[i]
        D = fast_gen_cost_matrix(v_ref, v)
        path = backtracking(D)[::-1]
        linked = [pj for pi, pj in path if pi == ref_idx]
        if len(linked) == 0:
            taus[i] = 0.0
            continue
        linked_diffs = np.abs(v_ref[ref_idx] - v[linked])
        diffs_sorted = np.argsort(linked_diffs)
        if len(diffs_sorted) == 1:
            estimative = np.abs(diffs_sorted[0] + linked[0] - ref_idx)
        elif len(diffs_sorted) < 4:
            estimative = np.abs(np.mean(diffs_sorted) + linked[0] - ref_idx)
        else:
            estimative = np.abs(np.mean(diffs_sorted[:3]) + linked[0] - ref_idx)
        taus[i] = estimative
    return taus


def estimate_taus_energy(
    signals: List[np.ndarray],
    knee_estimator: Callable[[np.ndarray], int],
    energy_func: Callable[[List[np.ndarray]], List[np.ndarray]],
) -> np.ndarray:
    voltages = energy_func(signals)
    vref_knee = knee_estimator(voltages[0])
    taus = np.zeros(4)
    taus[0] = 0.0
    for i in range(1, 4):
        knee = knee_estimator(voltages[i])
        taus[i] = np.abs(knee - vref_knee)
    return taus


def estimate_taus_gcc(
    signals: List[np.ndarray],
    normalization: Callable[[List[np.ndarray]], List[np.ndarray]] = norm_zero_mean,
    gcc_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = gcc_phat,
) -> np.ndarray:
    voltages = normalization(signals)
    v_ref = voltages[0]
    taus = np.zeros(4)
    taus[0] = 0.0
    for i in range(1, 4):
        cross_corr = gcc_func(v_ref, voltages[i])
        estimative = np.abs(np.argmax(np.abs(cross_corr)) - len(v_ref) + 1)
        taus[i] = estimative
    return taus


# ---------------------------------------------------------
# Benchmark Runner & Evaluation
# ---------------------------------------------------------

METHODS: Dict[str, Callable[[List[np.ndarray]], np.ndarray]] = {
    "DTW": estimate_taus_dtw,
    "GCC-PHAT": lambda s: estimate_taus_gcc(s, gcc_func=gcc_phat),
    "GCC-ROTH": lambda s: estimate_taus_gcc(s, gcc_func=gcc_roth),
    "GCC-SCOT": lambda s: estimate_taus_gcc(s, gcc_func=gcc_scot),
    "GCC-HT": lambda s: estimate_taus_gcc(s, gcc_func=gcc_ht),
    "CumEnergy + Curv": lambda s: estimate_taus_energy(s, curv_detection, cum_energy),
    "EnergyCrit + Argmin": lambda s: estimate_taus_energy(
        s, np.argmin, energy_criterion
    ),
    "Akaike + Argmin": lambda s: estimate_taus_energy(s, np.argmin, akaike_info),
}

TEST_SCENARIOS = [
    {
        "id": "Test_1a",
        "name": "Test 1a: Fixed Beta (0.6ns), High SNR (15dB)",
        "sheet_name": "Test1a_FixedBeta06_HighSNR",
        "snr": HIGH_SNR,
        "betas": BETA_06,
        "description": "Evaluates methods under identical sharp pulse shapes (beta=0.6ns) at high SNR (15dB).",
    },
    {
        "id": "Test_1b",
        "name": "Test 1b: Fixed Beta (2.7ns), High SNR (15dB)",
        "sheet_name": "Test1b_FixedBeta27_HighSNR",
        "snr": HIGH_SNR,
        "betas": BETA_27,
        "description": "Evaluates methods under identical smooth pulse shapes (beta=2.7ns) at high SNR (15dB).",
    },
    {
        "id": "Test_2",
        "name": "Test 2: Varying Betas [0.6, 1.2, 1.2, 2.7]ns, High SNR (15dB)",
        "sheet_name": "Test2_VarBetas_HighSNR",
        "snr": HIGH_SNR,
        "betas": BETA_VARYING,
        "description": "Evaluates methods when waveforms differ across channels due to propagation distance at high SNR (15dB).",
    },
    {
        "id": "Test_3a",
        "name": "Test 3a: Fixed Beta (0.6ns), Varying SNR [15, 10, 5, 2]dB",
        "sheet_name": "Test3a_FixedBeta06_VarSNR",
        "snr": VARYING_SNR,
        "betas": BETA_06,
        "description": "Evaluates methods under sharp pulse shapes with SNR attenuation across distance.",
    },
    {
        "id": "Test_3b",
        "name": "Test 3b: Fixed Beta (2.7ns), Varying SNR [15, 10, 5, 2]dB",
        "sheet_name": "Test3b_FixedBeta27_VarSNR",
        "snr": VARYING_SNR,
        "betas": BETA_27,
        "description": "Evaluates methods under smooth pulse shapes with SNR attenuation across distance.",
    },
    {
        "id": "Test_4",
        "name": "Test 4: Varying Betas [0.6, 1.2, 1.2, 2.7]ns, Varying SNR [15, 10, 5, 2]dB",
        "sheet_name": "Test4_VarBetas_VarSNR",
        "snr": VARYING_SNR,
        "betas": BETA_VARYING,
        "description": "Evaluates methods in realistic scenario: different pulse shapes and SNR attenuation across channels.",
    },
]


def run_benchmark(n_runs: int = N_RUNS) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    scenario_dfs = {}
    master_rows = []

    total_scenarios = len(TEST_SCENARIOS)
    print(
        f"=== Starting Synthetic TDOA Benchmark (N={n_runs} runs per scenario) ===",
        flush=True,
    )
    start_time = time.time()

    for idx, scen in enumerate(TEST_SCENARIOS, 1):
        print(f"\n[{idx}/{total_scenarios}] Running {scen['name']}...", flush=True)
        snr = scen["snr"]
        betas = scen["betas"]
        taus = TAUS_TRUE

        scen_rows = []

        for method_name, method_func in METHODS.items():
            t0_m = time.time()
            errors = np.zeros((n_runs, 4))

            for k in range(n_runs):
                signals = []
                for ts, gain, noise, beta in zip(taus, GAINS, snr, betas):
                    s = generate_pd(ts, gain, snr=noise, alpha=5e-9, beta=beta)
                    signals.append(s)

                try:
                    taus_est = method_func(signals)
                    erro = np.abs(TAUS_SAMPLE - taus_est)
                except Exception as _:
                    erro = np.full(4, np.nan)

                errors[k, :] = erro

            # Metrics
            ch2_mae = np.nanmean(errors[:, 1])
            ch3_mae = np.nanmean(errors[:, 2])
            ch4_mae = np.nanmean(errors[:, 3])

            ch2_std = np.nanstd(errors[:, 1])
            ch3_std = np.nanstd(errors[:, 2])
            ch4_std = np.nanstd(errors[:, 3])

            overall_mae_samples = np.nanmean(errors[:, 1:])
            overall_std_samples = np.nanstd(errors[:, 1:])
            rmse_samples = np.sqrt(np.nanmean(errors[:, 1:] ** 2))
            max_error_samples = np.nanmax(errors[:, 1:])

            mae_ns = overall_mae_samples * (TAMOS * 1e9)
            rmse_ns = rmse_samples * (TAMOS * 1e9)
            elapsed = time.time() - t0_m

            row = {
                "Scenario_ID": scen["id"],
                "Scenario_Name": scen["name"],
                "Method": method_name,
                "MAE_CH2 (samples)": ch2_mae,
                "MAE_CH3 (samples)": ch3_mae,
                "MAE_CH4 (samples)": ch4_mae,
                "Overall_MAE (samples)": overall_mae_samples,
                "Overall_Std (samples)": overall_std_samples,
                "RMSE (samples)": rmse_samples,
                "Max_Error (samples)": max_error_samples,
                "Overall_MAE (ns)": mae_ns,
                "RMSE (ns)": rmse_ns,
                "Std_CH2 (samples)": ch2_std,
                "Std_CH3 (samples)": ch3_std,
                "Std_CH4 (samples)": ch4_std,
                "Execution_Time (s)": elapsed,
            }
            scen_rows.append(row)
            master_rows.append(row)

            print(
                f"  - {method_name:<20} | MAE: {overall_mae_samples:.2f} sam ({mae_ns:.2f} ns) | RMSE: {rmse_samples:.2f} sam | Time: {elapsed:.2f}s",
                flush=True,
            )

        scen_df = pd.DataFrame(scen_rows)
        scenario_dfs[scen["sheet_name"]] = scen_df

    master_df = pd.DataFrame(master_rows)
    print(
        f"\n=== Benchmark completed in {time.time() - start_time:.2f} seconds ===",
        flush=True,
    )
    return scenario_dfs, master_df


# ---------------------------------------------------------
# Exporting Results to Excel & Markdown
# ---------------------------------------------------------


def export_to_excel(
    scenario_dfs: Dict[str, pd.DataFrame],
    master_df: pd.DataFrame,
    filepath: str = "result_sheets/synthetic_benchmark2.xlsx",
):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Master summary tab
        master_df.to_excel(writer, sheet_name="Master_Summary", index=False)
        # Individual scenario tabs
        for sheet_name, df in scenario_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(
        f"Results exported successfully to Excel: [synthetic_benchmark.xlsx](file://{os.path.abspath(filepath)})",
        flush=True,
    )


def export_to_markdown(
    master_df: pd.DataFrame,
    filepath: str = "result_sheets/synthetic_benchmark_summary2.md",
):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = []
    lines.append("# Synthetic TDOA Estimation Benchmark Summary\n")
    lines.append(f"- **Sampling Interval**: {TAMOS * 1e9:.2f} ns ({TAMOS} s)")
    lines.append("- **Reference Delay (CH1)**: 0 samples")
    lines.append(
        f"- **True Sample Delays [CH1, CH2, CH3, CH4]**: {TAUS_SAMPLE.tolist()}"
    )
    lines.append(f"- **Channel Gains**: {GAINS.tolist()}")
    lines.append(f"- **Monte Carlo Runs per Scenario**: {N_RUNS}\n")

    lines.append("## Executive Summary\n")
    lines.append(
        "The table below compares the TDOA estimation methods across all 6 test scenarios.\n"
    )

    for scen in TEST_SCENARIOS:
        scen_id = scen["id"]
        scen_name = scen["name"]
        desc = scen["description"]

        lines.append(f"### {scen_name}\n")
        lines.append(f"*{desc}*\n")

        sub_df = master_df[master_df["Scenario_ID"] == scen_id].copy()
        sub_df.sort_values(by="Overall_MAE (samples)", inplace=True)

        # Markdown table
        lines.append(
            "| Rank | Method | Overall MAE (samples) | Overall MAE (ns) | RMSE (samples) | Std Dev (samples) | CH2 MAE | CH3 MAE | CH4 MAE |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for rank, (_, row) in enumerate(sub_df.iterrows(), 1):
            method = row["Method"]
            mae_s = row["Overall_MAE (samples)"]
            mae_ns = row["Overall_MAE (ns)"]
            rmse_s = row["RMSE (samples)"]
            std_s = row["Overall_Std (samples)"]
            ch2 = row["MAE_CH2 (samples)"]
            ch3 = row["MAE_CH3 (samples)"]
            ch4 = row["MAE_CH4 (samples)"]
            lines.append(
                f"| {rank} | **{method}** | {mae_s:.2f} | {mae_ns:.2f} | {rmse_s:.2f} | {std_s:.2f} | {ch2:.2f} | {ch3:.2f} | {ch4:.2f} |"
            )
        lines.append("\n---\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(
        f"Summary exported successfully to Markdown: [synthetic_benchmark_summary.md](file://{os.path.abspath(filepath)})",
        flush=True,
    )


def main():
    scenario_dfs, master_df = run_benchmark(n_runs=N_RUNS)
    export_to_excel(scenario_dfs, master_df)
    export_to_markdown(master_df)


if __name__ == "__main__":
    main()
