import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .localizacao import LocalizationResult
from .experiment_loader import ANTENNA_POSITIONS, SOURCE_POSITIONS

# antenna square side and area/volume
ANTENNA_SIDE = 2.0
ANTENNA_AREA = ANTENNA_SIDE**2  # 4.0 m²
ANTENNA_VOLUME = ANTENNA_SIDE**3  # 8.0 m³


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


def to_dataframe(results: list[LocalizationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "exp_antena": r.exp_antenna,
            "exp_index": r.exp_index,
            "x_est": r.position_estimated[0],
            "y_est": r.position_estimated[1],
            "x_true": r.position_true[0],
            "y_true": r.position_true[1],
            "t1_estimated": r.t1_estimated,
            "converged": r.converged,
            "abs_error(m)": r.abs_error,
            "rel_error(%)": r.rel_error,
        }
        if r.mode == "3d":
            row["z_est"] = r.position_estimated[2]
            row["z_true"] = r.position_true[2]
        rows.append(row)
    return pd.DataFrame(rows)


def remove_outliers(
    df: pd.DataFrame, column: str = "abs_error(m)", k: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers per antenna group using IQR filtering on a given column.
    """
    df = df[df["converged"]].copy()
    q1 = df.groupby("exp_antena")[column].transform("quantile", 0.25)
    q3 = df.groupby("exp_antena")[column].transform("quantile", 0.75)
    iqr = q3 - q1
    mask = (df[column] >= q1 - k * iqr) & (df[column] <= q3 + k * iqr)
    return df[mask].reset_index(drop=True)


def remove_out_of_bounds(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["converged"]].copy()
    df = df.query("0 <= x_est <= 2 and 0 <= y_est <= 2")
    if "z_est" in df.columns:
        df = df.query("0 <= z_est <= 1")
    return df


def mean_estimates(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_out_of_bounds(df)
    grouped = df.groupby("exp_antena")
    is_3d = False

    cols = ["x_est", "y_est", "x_true", "y_true"]
    if "z_est" in df.columns and "z_true" in df.columns:
        is_3d = True
        cols.extend(["z_est", "z_true"])

    result = grouped[cols].mean().reset_index()

    if is_3d:
        result["abs_error(m)"] = np.sqrt(
            (result["x_est"] - result["x_true"]) ** 2
            + (result["y_est"] - result["y_true"]) ** 2
            + (result["z_est"] - result["z_true"]) ** 2
        )
        result["rel_error(%)"] = (
            (4 / 3) * np.pi * result["abs_error(m)"] ** 3 / ANTENNA_VOLUME
        )
    else:
        result["abs_error(m)"] = np.sqrt(
            (result["x_est"] - result["x_true"]) ** 2
            + (result["y_est"] - result["y_true"]) ** 2
        )
        result["rel_error(%)"] = np.pi * result["abs_error(m)"] ** 2 / ANTENNA_AREA

    return result


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------


def export_xlsx(df: pd.DataFrame, path: str) -> None:
    means = df.groupby("exp_antena")[["abs_error(m)", "rel_error(%)"]].mean()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for antenna, group in df.groupby("exp_antena"):
            group.drop(columns="exp_antena").to_excel(
                writer, sheet_name=antenna, index=False
            )
        means.to_excel(writer, sheet_name="médias")
    print(f"Excel exportado: {path}")


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    means = df.groupby("exp_antena")[["abs_error(m)", "rel_error(%)"]].mean()
    for antenna, group in df.groupby("exp_antena"):
        print(f"================ {antenna} ================")
        print(
            group[
                [
                    "exp_index",
                    "x_est",
                    "y_est",
                    "x_true",
                    "y_true",
                    "abs_error(m)",
                    "rel_error(%)",
                    "converged",
                ]
            ].to_string(index=False)
        )
    print("===================== médias =====================")
    print(means.to_string())


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------


def plot_location_scatter(
    results: list[LocalizationResult] | pd.DataFrame,
    only_mean_position: bool = False,
    only_converged: bool = True,
    remove_outsiders: bool = True,
) -> plt.Figure:
    if isinstance(results, pd.DataFrame):
        df = results.copy()
        df = df.query("0 <= x_est <= 2 and 0 <= y_est <= 2")
        mode = "3d" if "z_est" in df.columns else "2d"
    else:
        df = to_dataframe(results)
        # df = df.query("0 <= x_est <= 2 and 0 <= y_est <= 2")
        mode = results[0].mode if results else "2d"

    if only_converged and "converged" in df.columns:
        df = df[df["converged"]].copy()

    if remove_outsiders and "converged" in df.columns:
        df = df.query("0 <= x_est <= 2 and 0 <= y_est <= 2")
        if mode == "3d":
            df = df.query("0 <= z_est <= 1")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d") if mode == "3d" else fig.add_subplot(111)

    # antenna positions
    ant_xy = np.array([v for v in ANTENNA_POSITIONS.values()])
    if mode == "3d":
        ax.scatter(
            ant_xy[:, 0],
            ant_xy[:, 1],
            ant_xy[:, 2],
            marker="^",
            s=100,
            c="black",
            zorder=5,
            label="antenas",
        )
        ax.set_xlim([0.0, 2.1])
        ax.set_ylim([0.0, 2.1])
        ax.set_zlim([0.0, 1.1])
    else:
        ax.scatter(
            ant_xy[:, 0],
            ant_xy[:, 1],
            marker="^",
            s=100,
            c="black",
            zorder=5,
            label="antenas",
        )
        # draw antenna boundary square
        corners = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], dtype=float)
        ax.plot(corners[:, 0], corners[:, 1], "k--", linewidth=0.8, alpha=0.4)

    # true source positions
    colors = {"antena1": "C0", "antena2": "C1", "antena3": "C2", "antena4": "C3"}
    for antenna, pos in SOURCE_POSITIONS.items():
        c = colors.get(antenna, "black")
        if mode == "3d":
            ax.scatter(*pos, marker="*", s=200, c=c, zorder=6)
        else:
            ax.scatter(*pos[:2], marker="*", s=200, c=c, zorder=6, label=f"{antenna}")

    if not only_mean_position:
        # estimated positions + error lines
        for _, r in df.iterrows():
            antenna = r["exp_antena"]
            c = colors.get(antenna, "black")
            if mode == "3d":
                ax.scatter(
                    r["x_est"], r["y_est"], r["z_est"], marker="o", s=30, c=c, alpha=0.8
                )
                ax.plot(
                    [r["x_est"], r["x_true"]],
                    [r["y_est"], r["y_true"]],
                    [r["z_est"], r["z_true"]],
                    c=c,
                    linewidth=0.4,
                    alpha=0.3,
                )
            else:
                ax.scatter(r["x_est"], r["y_est"], marker="o", s=30, c=c, alpha=0.8)
                ax.plot(
                    [r["x_est"], r["x_true"]],
                    [r["y_est"], r["y_true"]],
                    c=c,
                    linewidth=0.4,
                    alpha=0.3,
                )

    # mean estimates per antenna
    df_mean = mean_estimates(df)
    first_mean = True
    for _, row in df_mean.iterrows():
        antenna = row["exp_antena"]
        c = colors.get(antenna, "black")
        label = "estimativa média" if first_mean else None
        first_mean = False
        if mode == "3d":
            ax.scatter(
                row["x_est"],
                row["y_est"],
                row["z_est"],
                marker="X",
                s=120,
                c=c,
                zorder=7,
                label=label,
            )
        else:
            ax.scatter(
                row["x_est"],
                row["y_est"],
                marker="X",
                s=120,
                c=c,
                zorder=7,
                label=label,
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if mode == "3d":
        ax.set_zlabel("z (m)")
    ax.set_title("Estimativas de localização de DPs", fontsize=11)
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Error boxplot
# ---------------------------------------------------------------------------


def plot_error_boxplot(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    groups = [group["abs_error"].values for _, group in df.groupby("exp_antena")]
    labels = [antenna for antenna, _ in df.groupby("exp_antena")]

    axes[0].boxplot(groups, labels=labels)
    axes[0].set_title("Erro absoluto por posição (m)")
    axes[0].set_ylabel("erro 2D (m)")
    axes[0].set_xlabel("posição da DP")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    groups_rel = [group["rel_error"].values for _, group in df.groupby("exp_antena")]
    axes[1].boxplot(groups_rel, labels=labels)
    axes[1].set_title("Erro relativo por posição (área do círculo / área total)")
    axes[1].set_ylabel("erro relativo")
    axes[1].set_xlabel("posição da DP")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig
