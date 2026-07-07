from typing import Optional

import numpy as np
from icecream import ic
from dataclasses import dataclass, field
from collections.abc import Callable
from scipy.optimize import fsolve

from .dtw import gen_cost_matrix, backtracking, dist
from .experiment_loader import (
    Experiment,
    ANTENNA_POSITIONS,
    SOURCE_POSITIONS,
)
from .preprocessing import cum_energy_exp, deriv_estimation

v_e = 3e8
REF_CHANNEL = {
    "antena1": "CH1",
    "antena2": "CH2",
    "antena3": "CH3",
    "antena4": "CH4",
}

TDOA_SAMPLES = {
    "antena1": [6.66698, 11.1186, 6.66698],
    "antena2": [7.47879, 7.16611, 11.70964],
    "antena3": [12.02122, 7.33526, 7.33526],
    "antena4": [7.47879, 11.70964, 7.16611],
}


@dataclass
class LocalizationResult:
    exp_antenna: str  # e.g. "antena2"
    exp_index: int  # 0-based index within antenna folder
    taus: dict  # {"CH2": float, "CH3": float, "CH4": float} in seconds
    mode: str
    position_estimated: np.ndarray  # [x, y] or [x, y, z] in meters
    t1_estimated: float
    position_true: np.ndarray  # ground truth from SOURCE_POSITIONS
    converged: bool  # fsolve convergence flag
    taus_estimated_samples: dict  # {"CH2": float, ...} in samples
    taus_true_samples: dict  # {"CH2": float, ...} in samples
    tau_errors: dict = field(init=False)
    abs_error: float = field(init=False)  # euclidean error in xyz plane (meters)
    rel_error: float = field(init=False)  # volumetric/area error (%)

    def __post_init__(self):
        est = self.position_estimated
        true = self.position_true
        self.abs_error = float(np.linalg.norm(est - true))
        self.tau_errors = {
            ch: abs(self.taus_estimated_samples[ch] - self.taus_true_samples[ch])
            for ch in self.taus_estimated_samples
        }

        if self.mode == "2d":
            ANTENNA_AREA = 4.0
            self.rel_error = 100 * (np.pi * self.abs_error**2) / ANTENNA_AREA
        elif self.mode == "3d":
            ANTENNA_VOLUME = 4.0 * 0.97
            self.rel_error = 100 * (4 / 3 * np.pi * self.abs_error**3) / ANTENNA_VOLUME
        else:
            self.rel_error = None
            raise ValueError


# ----------------------------------------
# TDOA EQUATIONS
# ----------------------------------------


def tdoa_2d(vars, taus: dict) -> list:
    x, y, t1 = vars

    # remove a equação com o maior tau (menor SNR)
    max_ch = max(taus, key=lambda ch: taus[ch])
    new_taus = {ch: tau for ch, tau in taus.items() if ch != max_ch}

    eqs = []
    for ch, tau in new_taus.items():
        xi = ANTENNA_POSITIONS[ch][0]
        yi = ANTENNA_POSITIONS[ch][1]
        eq = (x - xi) ** 2 + (y - yi) ** 2 - (v_e * (t1 + tau)) ** 2
        eqs.append(eq)
    return eqs


def tdoa_3d(vars, taus: dict) -> list:
    x, y, z, t1 = vars

    eqs = []
    for ch, tau in taus.items():
        xi = ANTENNA_POSITIONS[ch][0]
        yi = ANTENNA_POSITIONS[ch][1]
        zi = ANTENNA_POSITIONS[ch][2]
        eq = (x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2 - (v_e * (t1 + tau)) ** 2
        eqs.append(eq)
    return eqs


# -----------------------------------
# Tau estimation
# -----------------------------------


def true_taus(experiments: list[Experiment]) -> list[dict]:
    refs = {"antena1": 0, "antena2": 1, "antena3": 2, "antena4": 3}
    ch_names = ["CH1", "CH2", "CH3", "CH4"]
    result = []
    for exp in experiments:
        ref_index = refs[exp.antenna]
        ref_ch = ch_names[ref_index]
        travel = exp.travel_times
        ref_time = travel[ref_ch]
        taus = {ch: t - ref_time for ch, t in travel.items()}
        result.append(taus)
    return result


# ------------ FACTORIES -------------


def tau_estim_dtw(
    pre_processing: Callable,
    ref_estimator: Callable,
    distance_func: Callable,
    radius: int = 24,
) -> Callable:
    refs = {"antena1": 0, "antena2": 1, "antena3": 2, "antena4": 3}

    def _estimate(exp: Experiment) -> dict:
        voltages = pre_processing(exp)
        ref_index = refs[exp.antenna]
        ref_arrival = ref_estimator(voltages[ref_index])

        v_ref = voltages[ref_index]
        ch_names = [f"CH{i + 1}" for i in range(len(voltages))]

        taus = {}
        for i, (v, ch) in enumerate(zip(voltages, ch_names)):
            if i == ref_index:
                taus[ch] = 0.0
                continue
            D = gen_cost_matrix(v_ref, v, distance_func, radius=radius)
            path = backtracking(D)[::-1]
            linked = [pj for pi, pj in path if pi == ref_arrival]
            linked_diffs = np.abs(v_ref[ref_arrival] - v[linked])
            estimative = np.abs(np.argmin(linked_diffs) + linked[0] - ref_arrival)
            taus[ch] = estimative / exp.sample_rate
        return taus

    return _estimate


def tau_estim_gcc(pre_processing: Callable, gcc_func: Callable) -> Callable:
    refs = {"antena1": 0, "antena2": 1, "antena3": 2, "antena4": 3}

    def _estimate(exp: Experiment) -> dict:
        voltages = pre_processing(exp)
        ref_index = refs[exp.antenna]

        v_ref = voltages[ref_index]
        ch_names = [f"CH{i + 1}" for i in range(len(voltages))]
        taus = {}
        for i, (v, ch) in enumerate(zip(voltages, ch_names)):
            if i == ref_index:
                taus[ch] = 0.0
                continue
            cross_corr = gcc_func(v_ref, v)
            estimative = np.abs(np.argmax(np.abs(cross_corr)) - len(v_ref) + 1)
            taus[ch] = estimative / exp.sample_rate
        return taus

    return _estimate


def tau_estim_energy(pre_processing: Callable, knee_estimator: Callable) -> Callable:
    refs = {"antena1": 0, "antena2": 1, "antena3": 2, "antena4": 3}

    def _estimate(exp: Experiment) -> dict:
        voltages = pre_processing(exp)
        ref_index = refs[exp.antenna]

        v_ref = voltages[ref_index]
        vref_knee = knee_estimator(v_ref)
        ch_names = [f"CH{i + 1}" for i in range(len(voltages))]
        taus = {}
        for i, (v, ch) in enumerate(zip(voltages, ch_names)):
            if i == ref_index:
                taus[ch] = 0.0
                continue
            knee = knee_estimator(v)
            estimative = np.abs(knee - vref_knee)
            taus[ch] = estimative / exp.sample_rate
        return taus

    return _estimate


def batch_tau_estimates(
    experiments: list[Experiment],
    estimator: Callable,
) -> list[dict]:
    return [estimator(exp) for exp in experiments]


# -----------------------------------
# Localization
# -----------------------------------


def estimate_location(
    exp: Experiment, taus: dict, mode: str = "2d"
) -> LocalizationResult:
    t_mid = np.sqrt(2) / v_e
    if mode == "2d":
        p0_2d = [1.0, 1.0, t_mid]
        sol, info, ier, msg = fsolve(tdoa_2d, p0_2d, full_output=True, args=(taus,))
        true_pos = SOURCE_POSITIONS[exp.antenna][:-1]
    else:
        p0_3d = [1.0, 1.0, 0.5, t_mid]
        sol, info, ier, msg = fsolve(tdoa_3d, p0_3d, full_output=True, args=(taus,))
        true_pos = SOURCE_POSITIONS[exp.antenna]

    refs = {"antena1": 0, "antena2": 1, "antena3": 2, "antena4": 3}
    ch_names = ["CH1", "CH2", "CH3", "CH4"]
    ref_ch = ch_names[refs[exp.antenna]]
    non_ref_chs = [ch for ch in ch_names if ch != ref_ch]

    taus_true_samples = {
        ch: val for ch, val in zip(non_ref_chs, TDOA_SAMPLES[exp.antenna])
    }
    taus_estimated_samples = {
        ch: tau * exp.sample_rate for ch, tau in taus.items() if ch != ref_ch
    }
    return LocalizationResult(
        exp_antenna=exp.antenna,
        exp_index=exp.index,
        taus=taus,
        mode=mode,
        position_estimated=sol[:-1],
        t1_estimated=sol[-1],
        position_true=true_pos,
        converged=bool(ier == 1),
        taus_estimated_samples=taus_estimated_samples,
        taus_true_samples=taus_true_samples,
    )


def batch_localization(
    experiments: list[Experiment],
    taus: list[dict],
    mode: str = "2d",
) -> list[LocalizationResult]:
    results = []
    for exp, t in zip(experiments, taus):
        results.append(estimate_location(exp, t, mode=mode))
    return results
