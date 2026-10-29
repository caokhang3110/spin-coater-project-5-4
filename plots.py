# File: plots.py
# Role: #6 Code C - Graphs, charts, and figures
# Task: Read data from CSV files in data/outputs and generate the 5 paper figures (Figure 2-6, 300 dpi).
#       The 13 single-figure functions plot_fig01..plot_fig13 are kept for reference but no longer called.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# Directory paths
OUTPUT_DIR = "data/outputs"
INPUT_DIR = "data/inputs"
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)


# Step 1: Common plot styling (font sizes, grid, dimensions)
def set_common_style():
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 13


def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}. Run python main.py first."
        )
    return pd.read_csv(path)


# Helper function to apply tight layout and save figures at 300 dpi
def finish_and_save(fig, filename):
    fig.tight_layout()
    save_path = os.path.join(VIS_DIR, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def get_t_ramp_from_summary_or_results(row, target_rpm=3000.0):
    case_name = row["case"]

    # Always compute 99% target speed directly from CSV for exponential profile (~2.995 s)
    # because summary.csv might store the 99.9% value (~3.9 s)
    if "exp" not in case_name and "t_ramp_s" in row and pd.notna(row["t_ramp_s"]):
        return float(row["t_ramp_s"])

    df_case = load_csv(f"{case_name}_results.csv")
    threshold = 0.99 * target_rpm if "exp" in case_name else 0.999 * target_rpm
    reached = df_case[df_case["rpm"] >= threshold]
    if len(reached) > 0:
        return float(reached["time_t"].iloc[0])
    return float(row["t_end_s"])


# Step 2: Plot 10 standard graphs - read directly from CSV files

# Figure 1: Angular velocity omega(t) for 3 acceleration profiles
def plot_fig01():
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_lin["time_t"], df_lin["omega"], "-", linewidth=2, label="Linear")
    ax.plot(df_scv["time_t"], df_scv["omega"], "--", linewidth=2, label="S-curve")
    ax.plot(df_exp["time_t"], df_exp["omega"], ":", linewidth=2, label="Exponential")

    ax.set_title("Angular Velocity $\\omega(t)$ for Three Acceleration Profiles (300 mm, 3000 rpm)")
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Angular Velocity $\\omega$ (rad/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig01_omega_profiles.png")


# Figure 2: Angular acceleration $\alpha(t)$ (top) and angular jerk $j(t)$ on a symmetric logarithmic scale (bottom) for the three acceleration profiles for a 300 mm wafer accelerated to 3000 rpm at 1000 rpm/s ($\Delta t = 0.005\text{ s}$). Because the numerical simulation initializes with $\alpha(0) > 0$, the theoretical infinite jerk spike at $t = 0$ for the Linear and Exponential profiles is not shown. Additionally, the small negative jerk spike ($\approx -100\text{ rad/s}^3$) near $t \approx 3.9\text{–}4.0\text{ s}$ on the Exponential curve occurs when the ramp stage terminates and the small residual acceleration ($\alpha \approx 1\text{ rad/s}^2$) is truncated to zero upon transitioning to the constant-speed hold stage. 
def plot_fig02():
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.2, 6.4))

    # Top panel: angular acceleration alpha
    axes[0].plot(df_lin["time_t"], df_lin["alpha"], "-", linewidth=1.8, label="Linear")
    axes[0].plot(df_scv["time_t"], df_scv["alpha"], "--", linewidth=1.8, label="S-curve")
    axes[0].plot(df_exp["time_t"], df_exp["alpha"], ":", linewidth=2.0, label="Exponential")
    axes[0].set_ylabel("Angular Accel. $\\alpha$ (rad/s$^2$)")
    axes[0].set_title("Angular Acceleration $\\alpha(t)$ and Angular Jerk $j(t)$ vs. Time")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Bottom panel: angular jerk j (symlog scale so S-curve +-209 rad/s^3 is clearly visible)
    axes[1].plot(df_lin["time_t"], df_lin["jerk"], "-", linewidth=1.8, label="Linear")
    axes[1].plot(df_scv["time_t"], df_scv["jerk"], "--", linewidth=1.8, label="S-curve")
    axes[1].plot(df_exp["time_t"], df_exp["jerk"], ":", linewidth=2.0, label="Exponential")
    axes[1].set_yscale("symlog", linthresh=100)
    axes[1].set_xlabel("Time $t$ (s)")
    axes[1].set_ylabel("Angular Jerk $j$ (rad/s$^3$, symlog)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Add note about theoretical infinite jerk at t = 0 for Linear and Exponential profiles
    axes[1].text(
        0.5, -0.24,
        "Note: Simulation starts with initial $\\alpha(0) > 0$; theoretically, jerk at $t = 0$ is infinite for Linear and Exponential profiles.",
        transform=axes[1].transAxes, ha="center", va="top", fontsize=8.5, style="italic"
    )

    finish_and_save(fig, "fig02_alpha_jerk_profiles.png")


# Figure 3: Multi-pane plot for theta(t), omega(t), alpha(t)
def plot_fig03():
    df = load_csv("case_300mm_scurve_1000_results.csv")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7))

    axes[0].plot(df["time_t"], df["theta"], "-", color="tab:blue", linewidth=2, label="$\\theta(t)$")
    axes[0].set_ylabel("Angle $\\theta$ (rad)")
    axes[0].set_title("Multi-Pane Kinematics $\\theta(t)$, $\\omega(t)$, $\\alpha(t)$ (S-curve Profile)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    axes[1].plot(df["time_t"], df["omega"], "--", color="tab:orange", linewidth=2, label="$\\omega(t)$")
    axes[1].set_ylabel("Velocity $\\omega$ (rad/s)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")

    axes[2].plot(df["time_t"], df["alpha"], "-.", color="tab:green", linewidth=2, label="$\\alpha(t)$")
    axes[2].set_xlabel("Time $t$ (s)")
    axes[2].set_ylabel("Accel. $\\alpha$ (rad/s$^2$)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    finish_and_save(fig, "fig03_multipane_theta_omega_alpha.png")


# Figure 4: Phase portrait omega vs. theta
def plot_fig04():
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_lin["theta"], df_lin["omega"], "-", linewidth=2, label="Linear")
    ax.plot(df_scv["theta"], df_scv["omega"], "--", linewidth=2, label="S-curve")
    ax.plot(df_exp["theta"], df_exp["omega"], ":", linewidth=2, label="Exponential")

    ax.set_title("Phase Portrait: Angular Velocity $\\omega$ vs. Angular Position $\\theta$")
    ax.set_xlabel("Angular Position $\\theta$ (rad)")
    ax.set_ylabel("Angular Velocity $\\omega$ (rad/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig04_phase_portrait_omega_theta.png")


# Figure 5: Tangential acceleration a_t and centripetal acceleration a_c at wafer edge vs. t
def plot_fig05():
    df = load_csv("case_300mm_scurve_1000_results.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5.5))

    axes[0].plot(df["time_t"], df["a_t_edge"], "-", color="tab:red", linewidth=2, label="$a_t = \\alpha R$")
    axes[0].set_ylabel("Tangential $a_t$ (m/s$^2$)")
    axes[0].set_title("Wafer Edge Accelerations $a_t(t)$ and $a_c(t)$ (S-curve, $R = 0.15$ m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(df["time_t"], df["a_c_edge"], "--", color="tab:purple", linewidth=2, label="$a_c = \\omega^2 R$")
    axes[1].set_xlabel("Time $t$ (s)")
    axes[1].set_ylabel("Centripetal $a_c$ (m/s$^2$)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig05_edge_at_ac_vs_time.png")


# Figure 6: Ramp-up time t_ramp vs. acceleration rate
def plot_fig06():
    df_sum = load_csv("summary.csv")

    # Candidate linear cases (case_extreme_high_accel_20000 has a 6000 rpm target and will be filtered out)
    target_cases = [
        "case_extreme_low_accel_100",
        "case_300mm_linear_500",
        "case_300mm_linear_1000",
        "case_300mm_linear_2000",
        "case_300mm_linear_5000",
        "case_300mm_linear_10000",
        "case_extreme_high_accel_20000"
    ]

    accel_rates = []
    t_ramps = []

    for c_name in target_cases:
        matched = df_sum[df_sum["case"] == c_name]
        if len(matched) == 0:
            raise ValueError(f"Case '{c_name}' not found in summary.csv")
        row = matched.iloc[0]

        # Check target/max rpm from JSON config or summary.csv to filter only 3000 rpm target cases
        rpm_target = None
        json_path = os.path.join(INPUT_DIR, f"{c_name}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k in ["rpm_target", "target_rpm", "rpm_max", "omega_target_rpm"]:
                if k in cfg:
                    rpm_target = float(cfg[k])
                    break
        if rpm_target is None:
            for k in ["rpm_target", "rpm_max", "target_rpm"]:
                if k in row and pd.notna(row[k]):
                    rpm_target = float(row[k])
                    break
        if rpm_target is None:
            df_c = load_csv(f"{c_name}_results.csv")
            rpm_target = float(df_c["rpm"].max())

        # Skip cases whose target speed is not 3000 rpm (e.g., 20000 rpm/s case with 6000 rpm target)
        if abs(rpm_target - 3000.0) > 100.0:
            continue

        rate_val = float(c_name.split("_")[-1])
        t_val = get_t_ramp_from_summary_or_results(row, target_rpm=3000.0)
        accel_rates.append(rate_val)
        t_ramps.append(t_val)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(accel_rates, t_ramps, "-o", color="tab:blue", linewidth=2, label="Linear Profile (3000 rpm target)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Ramp-Up Time $t_{ramp}$ vs. Acceleration Rate (Linear Profile)")
    ax.set_xlabel("Acceleration Rate (rpm/s)")
    ax.set_ylabel("Ramp-Up Time $t_{ramp}$ (s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig06_tramp_vs_accel_rate.png")


# Figure 7: Edge centripetal acceleration a_c vs. rpm for different wafer sizes
def plot_fig07():
    wafer_cases = [
        (100, "case_100mm_linear_1000_results.csv", "-"),
        (150, "case_150mm_linear_1000_results.csv", "--"),
        (200, "case_200mm_linear_1000_results.csv", "-."),
        (300, "case_300mm_linear_1000_results.csv", ":")
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for d_mm, fname, ls in wafer_cases:
        df_w = load_csv(fname)
        # Filter only the ramp-up region where alpha > 0
        df_ramp = df_w[df_w["alpha"] > 1e-6]
        if len(df_ramp) == 0:
            df_ramp = df_w
        ax.plot(
            df_ramp["rpm"], df_ramp["a_c_edge"],
            linestyle=ls, linewidth=2,
            label=f"Wafer {d_mm} mm ($R = {d_mm / 2000:.3f}$ m)"
        )

    ax.set_title("Edge Centripetal Acceleration $a_c$ vs. Spin Speed for Wafer Sizes")
    ax.set_xlabel("Spin Speed (rpm)")
    ax.set_ylabel("Centripetal Acceleration $a_c$ (m/s$^2$)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig07_ac_vs_rpm_wafer_sizes.png")


# Figure 8: Work-energy theorem error vs. simulation time step (log-log scale)
def plot_fig08():
    df_sum = load_csv("summary.csv")

    # Filter dt convergence study cases for linear and scurve profiles
    df_lin_dt = df_sum[df_sum["case"].str.contains("linear_1000_dt", na=False)].sort_values("dt_s")
    df_scv_dt = df_sum[df_sum["case"].str.contains("scurve_1000_dt", na=False)].sort_values("dt_s")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot energy error in Joules (or relative percentage if updated by Code A)
    err_col = "energy_err_pct" if "energy_err_pct" in df_sum.columns else "energy_err_J"
    ylabel_str = "Relative Energy Error (%)" if err_col == "energy_err_pct" else "Energy Error $|\\int \\tau d\\theta - \\Delta K|$ (J)"

    if len(df_lin_dt) > 0:
        ax.plot(df_lin_dt["dt_s"], np.maximum(df_lin_dt[err_col], 1e-16), "-o", linewidth=2, label="Linear (RK45)")
    if len(df_scv_dt) > 0:
        ax.plot(df_scv_dt["dt_s"], np.maximum(df_scv_dt[err_col], 1e-16), "--s", linewidth=2, label="S-curve (RK45)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Work-Energy Theorem Error vs. Simulation Time Step $\\Delta t$")
    ax.set_xlabel("Time Step $\\Delta t$ (s)")
    ax.set_ylabel(ylabel_str)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig08_energy_error_log.png")


# Figure 9: Relative photoresist film thickness h/h_0 vs. angular velocity omega
def plot_fig09():
    df_ms = load_csv("case_300mm_multistep_results.csv")

    # Reference angular velocity at omega_0 = 500 rpm
    omega_0 = 500.0 * 2.0 * np.pi / 60.0
    df_valid = df_ms[df_ms["rpm"] >= 500.0].copy()
    omega_vals = np.sort(df_valid["omega"].unique())

    # Meyerhofer (1978) analytical scaling law: h / h_0 = (omega / omega_0)^(-1/2)
    h_ratio = (omega_vals / omega_0) ** (-0.5)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Plot solid curve without markers as this is an analytical model, not discrete measurements
    ax.plot(omega_vals, h_ratio, "-", color="tab:blue", linewidth=2.2,
            label="Applied Meyerhofer (1978) Analytical Model: $h/h_0 = (\\omega/\\omega_0)^{-1/2}$")

    ax.set_title("Relative Film Thickness $h/h_0$ vs. Angular Velocity $\\omega$ (Meyerhofer Model)")
    ax.set_xlabel("Angular Velocity $\\omega$ (rad/s)")
    ax.set_ylabel("$h/h_0$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    ax.text(
        0.5, -0.18,
        "Note: Curve represents the theoretical Meyerhofer (1978) scaling model applied to $\\omega$, not direct fluid simulation.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, style="italic"
    )

    finish_and_save(fig, "fig09_film_thickness_vs_omega.png")


# Figure 10: Complete multi-step spin-coating process recipe
def plot_fig10():
    df_ms = load_csv("case_300mm_multistep_results.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5))

    axes[0].plot(df_ms["time_t"], df_ms["rpm"], "-", color="blue", linewidth=2, label="Spin Speed (rpm)")
    axes[0].set_ylabel("Spin Speed (rpm)")
    axes[0].set_title("Multi-Step Spin-Coating Process Recipe (Spread $\\rightarrow$ Spin $\\rightarrow$ Stop)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(df_ms["time_t"], df_ms["torque"], "--", color="darkorange", linewidth=2, label="Motor Torque $\\tau$")
    axes[1].set_xlabel("Time $t$ (s)")
    axes[1].set_ylabel("Torque $\\tau$ (N$\\cdot$m)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig10_multistep_recipe_full.png")


# Step 3: Plot trajectory figure with acceleration vectors

def plot_fig11():
    df_sum = load_csv("summary.csv")
    df = load_csv("case_300mm_linear_1000_results.csv")

    # Get wafer radius R_m from summary.csv for case_300mm_linear_1000
    matched = df_sum[df_sum["case"] == "case_300mm_linear_1000"]
    R = float(matched["R_m"].iloc[0]) if len(matched) > 0 else 0.15

    # Filter only the initial stage time_t <= 0.3 s
    df_sub = df[df["time_t"] <= 0.3].copy()

    t = df_sub["time_t"].to_numpy()
    theta = df_sub["theta"].to_numpy()
    a_t = df_sub["a_t_edge"].to_numpy()
    a_c = df_sub["a_c_edge"].to_numpy()

    # Compute Cartesian coordinates x and y
    x = R * np.cos(theta)
    y = R * np.sin(theta)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    # Plot full circular wafer outline for reference
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), "--", color="lightgray", linewidth=1.2,
            label=f"Wafer Outline ($R = {R:.2f}$ m)")

    # Plot trajectory during the first 0.3 seconds
    ax.plot(x, y, "-", color="tab:blue", linewidth=2.5,
            label="Trajectory of a point on the wafer edge ($t \\leq 0.3$ s)")
    ax.scatter(x[0], y[0], color="green", s=65, zorder=5, label="Start point ($t = 0$)")

    # Select 10 evenly spaced time instants between 0 and 0.3 seconds
    idx = np.linspace(0, len(t) - 1, 10, dtype=int)

    # Unit vectors in tangential (-sin theta, cos theta) and centripetal (-cos theta, -sin theta) directions
    tx = -np.sin(theta[idx])
    ty = np.cos(theta[idx])
    cx = -np.cos(theta[idx])
    cy = -np.sin(theta[idx])

    ax_vec = a_t[idx] * tx + a_c[idx] * cx
    ay_vec = a_t[idx] * ty + a_c[idx] * cy

    # Normalize vectors to a fixed length of 0.04 m so all arrows fit cleanly inside the frame
    mag = np.sqrt(ax_vec**2 + ay_vec**2)
    fixed_len = 0.04
    ux = (ax_vec / mag) * fixed_len
    uy = (ay_vec / mag) * fixed_len

    ax.quiver(
        x[idx], y[idx], ux, uy,
        angles="xy", scale_units="xy", scale=1,
        color="crimson", width=0.006, zorder=6,
        label="Total acceleration vector (arrows show direction only)"
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.21, 0.21)
    ax.set_ylim(-0.21, 0.21)
    ax.set_title(f"Edge-point trajectory and direction of the acceleration\nvector during the first 0.3 s ($R = {R:.2f}$ m)")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    # Place legend outside below the plot so it does not overlap the trajectory
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize="small", frameon=True)

    finish_and_save(fig, "fig11_edge_trajectory_vectors.png")


# Step 4: Bar charts

# Figure 12: Bar chart 1 - Compare t_ramp, jerk_max, a_t_max across 3 profiles
def plot_fig12():
    df_sum = load_csv("summary.csv")

    case_list = [
        ("Linear", "case_300mm_linear_1000"),
        ("S-curve", "case_300mm_scurve_1000"),
        ("Exponential\n(99% target)", "case_300mm_exp_1000")
    ]

    labels = []
    t_ramps = []
    jerk_maxs = []
    at_maxs = []

    for label, c_name in case_list:
        matched = df_sum[df_sum["case"] == c_name]
        if len(matched) == 0:
            raise ValueError(f"Case {c_name} not found in summary.csv")
        row = matched.iloc[0]
        labels.append(label)
        t_ramps.append(get_t_ramp_from_summary_or_results(row, target_rpm=3000.0))
        jerk_maxs.append(float(row["jerk_max"]))
        at_maxs.append(float(row["a_t_max"]))

    x = np.arange(len(labels))
    # Widen figure to (11.2, 4.4) and reduce x-tick font size so labels do not crowd each other
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.4))

    # Subplot 1: t_ramp
    bars0 = axes[0].bar(x, t_ramps, width=0.55, color="steelblue", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9.5)
    axes[0].set_title("Ramp Time $t_{ramp}$")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_ylim(0, max(t_ramps) * 1.18)
    axes[0].bar_label(bars0, fmt="%.2f", padding=3, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: jerk_max (log scale)
    bars1 = axes[1].bar(x, jerk_maxs, width=0.55, color="seagreen", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9.5)
    axes[1].set_yscale("log")
    axes[1].set_title("Max Jerk $j_{max}$")
    axes[1].set_ylabel("Jerk (rad/s$^3$, log scale)")
    axes[1].set_ylim(10, max(jerk_maxs) * 4.0)
    axes[1].bar_label(bars1, fmt="%.0f", padding=3, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: a_t_max
    bars2 = axes[2].bar(x, at_maxs, width=0.55, color="indianred", edgecolor="black")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9.5)
    axes[2].set_title("Max Edge Tangential $a_{t,max}$")
    axes[2].set_ylabel("Acceleration (m/s$^2$)")
    axes[2].set_ylim(0, max(at_maxs) * 1.18)
    axes[2].bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Comparison of Key Metrics Across Three Acceleration Profiles")
    finish_and_save(fig, "fig12_bar_profile_comparison.png")


# Figure 13: Bar chart 2 - Revolutions and step duration breakdown for multi-step process
def plot_fig13():
    df_ms = load_csv("case_300mm_multistep_results.csv")

    # Default step names and durations from the multi-step recipe specification
    durations = [1.0, 5.0, 2.5, 30.0, 3.0]
    step_names = [
        "1. Ramp 500 rpm",
        "2. Hold 500 rpm",
        "3. Ramp 3000 rpm",
        "4. Hold 3000 rpm",
        "5. Spin Down"
    ]

    # Read step durations from data/inputs/case_300mm_multistep.json if available
    json_path = os.path.join(INPUT_DIR, "case_300mm_multistep.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ["steps", "segments", "recipe", "process_steps"]:
            if key in cfg and isinstance(cfg[key], list) and len(cfg[key]) > 0:
                if "duration" in cfg[key][0]:
                    durations = [float(s["duration"]) for s in cfg[key]]
                break

    # Calculate number of revolutions = delta_theta / (2 * pi) at each milestone
    t_milestones = np.cumsum([0.0] + durations)
    t_arr = df_ms["time_t"].to_numpy()
    theta_arr = df_ms["theta"].to_numpy()
    theta_at_milestones = np.interp(t_milestones, t_arr, theta_arr)
    revolutions = np.diff(theta_at_milestones) / (2.0 * np.pi)

    if len(step_names) != len(durations):
        step_names = [f"Step {i+1}" for i in range(len(durations))]

    x = np.arange(len(step_names))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.8, 6.0))

    bars_rev = axes[0].bar(x, revolutions, width=0.5, color="royalblue", edgecolor="black", label="Revolutions (rev)")
    axes[0].set_ylabel("Revolutions $N$ (rev)")
    axes[0].set_title("Revolutions and Step Duration Breakdown for Multi-Step Process")
    axes[0].set_ylim(0, max(revolutions) * 1.18)
    axes[0].bar_label(bars_rev, labels=[f"{v:.1f}" for v in revolutions], padding=3, fontsize=9.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    bars_dur = axes[1].bar(x, durations, width=0.5, color="sandybrown", edgecolor="black", label="Step Duration (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(step_names, rotation=15)
    axes[1].set_xlabel("Process Steps")
    axes[1].set_ylabel("Duration $\\Delta t$ (s)")
    axes[1].set_ylim(0, max(durations) * 1.18)
    axes[1].bar_label(bars_dur, labels=[f"{v:.1f} s" for v in durations], padding=3, fontsize=9.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")

    finish_and_save(fig, "fig13_bar_multistep_breakdown.png")

# Step 5: combine pictures into 5 pictures

def plot_combined_fig1():
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6.3, 4.5))
    datasets = [
        (df_lin, "-", "Linear"),
        (df_scv, "--", "S-curve"),
        (df_exp, ":", "Exponential")
    ]

    # (a) Angular position theta(t)
    for df, ls, lbl in datasets:
        axes[0, 0].plot(df["time_t"], df["theta"], linestyle=ls, linewidth=1.6, label=lbl)
    axes[0, 0].set_ylabel("Angle $\\theta$ (rad)", fontsize=8.5)
    axes[0, 0].set_title("(a) Angular Position $\\theta(t)$", fontsize=9, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # (b) Angular velocity omega(t)
    for df, ls, _ in datasets:
        axes[0, 1].plot(df["time_t"], df["omega"], linestyle=ls, linewidth=1.6)
    axes[0, 1].set_ylabel("Velocity $\\omega$ (rad/s)", fontsize=8.5)
    axes[0, 1].set_title("(b) Angular Velocity $\\omega(t)$", fontsize=9, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # (c) Angular acceleration alpha(t)
    for df, ls, _ in datasets:
        axes[1, 0].plot(df["time_t"], df["alpha"], linestyle=ls, linewidth=1.6)
    axes[1, 0].set_xlabel("Time $t$ (s)", fontsize=8.5)
    axes[1, 0].set_ylabel("Accel. $\\alpha$ (rad/s$^2$)", fontsize=8.5)
    axes[1, 0].set_title("(c) Angular Acceleration $\\alpha(t)$", fontsize=9, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # (d) Angular jerk j(t) using symlog scale so +/-209 rad/s^3 of S-curve is clearly visible
    for df, ls, _ in datasets:
        axes[1, 1].plot(df["time_t"], df["jerk"], linestyle=ls, linewidth=1.6)
    axes[1, 1].set_yscale("symlog", linthresh=100)
    axes[1, 1].set_xlabel("Time $t$ (s)", fontsize=8.5)
    axes[1, 1].set_ylabel("Jerk $j$ (rad/s$^3$, symlog)", fontsize=8.5)
    axes[1, 1].set_title("(d) Angular Jerk $j(t)$", fontsize=9, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)

    # Shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(VIS_DIR, "paper_fig01_kinematics_2x2.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


# Paper Figure 3 (2x2): Phase portrait, Trajectory, a_t/a_c, and Energy error (%) - S-curve only + text note for Linear
def plot_combined_fig2():
    df_sum = load_csv("summary.csv")
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6))

    # (a) Phase portrait omega vs. theta
    axes[0, 0].plot(df_lin["theta"], df_lin["omega"], "-", linewidth=1.5, label="Linear")
    axes[0, 0].plot(df_scv["theta"], df_scv["omega"], "--", linewidth=1.5, label="S-curve")
    axes[0, 0].plot(df_exp["theta"], df_exp["omega"], ":", linewidth=1.7, label="Exponential")
    axes[0, 0].set_xlabel("Angle $\\theta$ (rad)", fontsize=8.5)
    axes[0, 0].set_ylabel("Velocity $\\omega$ (rad/s)", fontsize=8.5)
    axes[0, 0].set_title("(a) Phase Portrait $\\omega(\\theta)$", fontsize=9, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # (b) Edge trajectory and normalized acceleration vectors (t <= 0.3 s, reduced whitespace)
    matched = df_sum[df_sum["case"] == "case_300mm_linear_1000"]
    R = float(matched["R_m"].iloc[0]) if len(matched) > 0 else 0.15
    df_sub = df_lin[df_lin["time_t"] <= 0.3].copy()
    t_sub = df_sub["time_t"].to_numpy()
    th_sub = df_sub["theta"].to_numpy()
    at_sub = df_sub["a_t_edge"].to_numpy()
    ac_sub = df_sub["a_c_edge"].to_numpy()

    x_edge = R * np.cos(th_sub)
    y_edge = R * np.sin(th_sub)
    th_circle = np.linspace(0, 2 * np.pi, 150)
    axes[0, 1].plot(R * np.cos(th_circle), R * np.sin(th_circle), "--", color="lightgray", linewidth=1.0)
    axes[0, 1].plot(x_edge, y_edge, "-", color="tab:blue", linewidth=1.8)
    axes[0, 1].scatter(x_edge[0], y_edge[0], color="green", s=25, zorder=5)

    idx = np.linspace(0, len(t_sub) - 1, 9, dtype=int)
    tx, ty = -np.sin(th_sub[idx]), np.cos(th_sub[idx])
    cx, cy = -np.cos(th_sub[idx]), -np.sin(th_sub[idx])
    ax_v = at_sub[idx] * tx + ac_sub[idx] * cx
    ay_v = at_sub[idx] * ty + ac_sub[idx] * cy
    mag = np.sqrt(ax_v**2 + ay_v**2)
    ux, uy = (ax_v / mag) * 0.038, (ay_v / mag) * 0.038

    axes[0, 1].quiver(x_edge[idx], y_edge[idx], ux, uy, angles="xy", scale_units="xy", scale=1,
                      color="crimson", width=0.008, zorder=6)
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].set_xlim(-0.175, 0.175)
    axes[0, 1].set_ylim(-0.175, 0.175)
    axes[0, 1].set_xticks([-0.15, 0.0, 0.15])
    axes[0, 1].set_yticks([-0.15, 0.0, 0.15])
    axes[0, 1].set_xlabel("$x$ (m)", fontsize=8.5)
    axes[0, 1].set_ylabel("$y$ (m)", fontsize=8.5)
    axes[0, 1].set_title("(b) Trajectory & $\\vec{a}$ Direction ($t\\leq 0.3$s)", fontsize=9, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # (c) Edge accelerations a_t and a_c vs. time (twin y-axis)
    ax_c_left = axes[1, 0]
    ax_c_right = ax_c_left.twinx()
    l1 = ax_c_left.plot(df_scv["time_t"], df_scv["a_t_edge"], "-", color="tab:red", linewidth=1.5, label="$a_t$ (left)")
    l2 = ax_c_right.plot(df_scv["time_t"], df_scv["a_c_edge"] / 1000.0, "--", color="tab:purple", linewidth=1.5, label="$a_c$ (right)")
    ax_c_left.set_xlabel("Time $t$ (s)", fontsize=8.5)
    ax_c_left.set_ylabel("$a_t$ (m/s$^2$)", color="tab:red", fontsize=8.5)
    ax_c_right.set_ylabel("$a_c$ ($\\times 10^3$ m/s$^2$)", color="tab:purple", fontsize=8.5)
    ax_c_right.tick_params(labelsize=8)
    ax_c_left.set_title("(c) Edge $a_t$ and $a_c$ (S-curve)", fontsize=9, fontweight="bold")
    ax_c_left.grid(True, linestyle="--", alpha=0.5)

    # (d) Energy error (%) vs. time step dt (log-log scale) - S-curve only, reading energy_err_J from summary.csv
    df_scv_dt = df_sum[df_sum["case"].str.contains("scurve_1000_dt", na=False)].sort_values("dt_s")

    if len(df_scv_dt) > 0:
        dt_vals = df_scv_dt["dt_s"].to_numpy()
        err_j = df_scv_dt["energy_err_J"].to_numpy()
        omega_max = df_scv_dt["rpm_max"].to_numpy() * 2.0 * np.pi / 60.0
        i_total = df_scv_dt["I_total"].to_numpy()
        k_max = 0.5 * i_total * (omega_max ** 2)
        err_pct = (err_j / k_max) * 100.0
        err_pct = np.maximum(err_pct, 1e-14)
        axes[1, 1].plot(dt_vals, err_pct, "--s", color="tab:orange", markersize=4, linewidth=1.5, label="S-curve")

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Time Step $\\Delta t$ (s)", fontsize=8.5)
    axes[1, 1].set_ylabel("Energy Error (%)", fontsize=8.5)
    axes[1, 1].set_title("(d) Energy Error vs. $\\Delta t$", fontsize=9, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Ghi chú cho trường hợp Linear có sai số bằng 0 (không vẽ được trên thang log)
    axes[1, 1].text(0.05, 0.15, "Linear: exact ($\\approx 0\\%$)", transform=axes[1, 1].transAxes,
                    fontsize=8, fontstyle="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    axes[1, 1].legend(loc="upper left", fontsize=7.5, frameon=True)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)

    # Shared legend at the top
    h_a, l_a = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_a + l1 + l2, l_a + ["$a_t$ (left)", "$a_c$ (right)"],
               loc="upper center", ncol=5, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(VIS_DIR, "paper_fig02_phase_traj_edge_err_2x2.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)

# Paper Figure 4 (1x3): t_ramp vs. alpha (fixed yticks), a_c vs. rpm (own legend), h/h0 vs. omega (distinct color)
def plot_combined_fig3():
    df_sum = load_csv("summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.55))

    # (a) t_ramp vs. acceleration rate with clean scalar y-ticks (0.3, 0.6, 1, 2, 3, 6)
    target_cases = [
        "case_300mm_extreme_low_accel_100",
        "case_300mm_linear_500",
        "case_300mm_linear_1000",
        "case_300mm_linear_2000",
        "case_300mm_linear_5000",
        "case_300mm_linear_10000",
        "case_300mm_extreme_high_accel_20000"
    ]
    rates, tramps = [], []
    for c_name in target_cases:
        m = df_sum[df_sum["case"] == c_name]
        if len(m) > 0:
            rates.append(float(c_name.split("_")[-1]))
            tramps.append(get_t_ramp_from_summary_or_results(m.iloc[0]))

    axes[0].plot(rates, tramps, "-o", color="tab:blue", markersize=3.5, linewidth=1.5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.3, 0.6, 1, 2, 3, 6])
    axes[0].get_yaxis().set_major_formatter(ScalarFormatter())
    axes[0].yaxis.minorticks_off()
    axes[0].set_yticklabels(["0.3", "0.6", "1", "2", "3", "6"], fontsize=8)
    axes[0].set_xlabel("Accel. Rate (rpm/s)", fontsize=8.5)
    axes[0].set_ylabel("$t_{ramp}$ (s)", fontsize=8.5)
    axes[0].set_title("(a) $t_{ramp}$ vs. $\\alpha$", fontsize=9, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) a_c vs. rpm for wafer sizes with dedicated legend inside subplot (b)
    wafer_cases = [
        (100, "case_100mm_linear_1000_results.csv", "-", "tab:blue"),
        (150, "case_150mm_linear_1000_results.csv", "--", "tab:orange"),
        (200, "case_200mm_linear_1000_results.csv", "-.", "tab:green"),
        (300, "case_300mm_linear_1000_results.csv", ":", "tab:red")
    ]
    for d_mm, fname, ls, col in wafer_cases:
        df_w = load_csv(fname)
        df_r = df_w[df_w["alpha"] > 1e-6]
        if len(df_r) == 0:
            df_r = df_w
        axes[1].plot(df_r["rpm"], df_r["a_c_edge"] / 1000.0, linestyle=ls, color=col, linewidth=1.5, label=f"{d_mm} mm")
    axes[1].set_xlabel("Spin Speed (rpm)", fontsize=8.5)
    axes[1].set_ylabel("$a_c$ ($\\times 10^3$ m/s$^2$)", fontsize=8.5)
    axes[1].set_title("(b) $a_c$ vs. Speed", fontsize=9, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=7.5, frameon=True, handlelength=1.8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # (c) Film thickness ratio h/h0 vs. omega using distinct darkmagenta color (not green)
    df_300 = load_csv("case_300mm_linear_1000_results.csv")
    omega_0 = 500.0 * 2.0 * np.pi / 60.0
    df_v = df_300[df_300["rpm"] >= 500.0].copy()
    om_vals = df_v["omega"].to_numpy()
    h_ratio = (om_vals / omega_0) ** (-0.5)

    axes[2].plot(om_vals, h_ratio, "-", color="darkmagenta", linewidth=1.6, label="$h/h_0 = (\\omega/\\omega_0)^{-1/2}$")
    axes[2].set_xlabel("Velocity $\\omega$ (rad/s)", fontsize=8.5)
    axes[2].set_ylabel("Ratio $h / h_0$", fontsize=8.5)
    axes[2].set_title("(c) Film Thickness $h(\\omega)$", fontsize=9, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=7.5, frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    save_path = os.path.join(VIS_DIR, "paper_fig03_parametric_1x3.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


# Paper Figure 5 (1x3 Bar Chart): Profile comparison with value labels on top of bars
def plot_combined_fig4():
    df_sum = load_csv("summary.csv")
    case_list = [
        ("Linear", "case_300mm_linear_1000"),
        ("S-curve", "case_300mm_scurve_1000"),
        ("Exp (99%)", "case_300mm_exp_1000")
    ]

    labels, t_ramps, jerk_maxs, at_maxs = [], [], [], []
    for lbl, c_name in case_list:
        m = df_sum[df_sum["case"] == c_name]
        if len(m) == 0:
            raise ValueError(f"Case {c_name} not found in summary.csv")
        row = m.iloc[0]
        labels.append(lbl)
        t_ramps.append(get_t_ramp_from_summary_or_results(row))
        jerk_maxs.append(float(row["jerk_max"]))
        at_maxs.append(float(row["a_t_max"]))

    x = np.arange(len(labels))
    colors = ["steelblue", "seagreen", "indianred"]
    hatches = ["", "//", ".."]

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.85))

    # (a) Ramp time t_ramp with bar labels (3.0 / 3.5 / 3.0 s)
    for i in range(len(x)):
        b = axes[0].bar(x[i], t_ramps[i], width=0.55, color=colors[i], edgecolor="black", hatch=hatches[i], label=labels[i])
        axes[0].bar_label(b, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, fontsize=8)
    axes[0].set_ylim(0, max(t_ramps) * 1.22)
    axes[0].set_ylabel("Time $t_{ramp}$ (s)", fontsize=8.5)
    axes[0].set_title("(a) Ramp Time $t_{ramp}$", fontsize=9, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Max jerk j_max (log scale) with integer bar labels (10472 / 209 / 741)
    for i in range(len(x)):
        b = axes[1].bar(x[i], jerk_maxs[i], width=0.55, color=colors[i], edgecolor="black", hatch=hatches[i])
        axes[1].bar_label(b, labels=[f"{jerk_maxs[i]:.0f}"], padding=2, fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylim(100, max(jerk_maxs) * 3.2)
    axes[1].set_ylabel("$j_{max}$ (rad/s$^3$, log)", fontsize=8.5)
    axes[1].set_title("(b) Max Jerk $j_{max}$", fontsize=9, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # (c) Max tangential acceleration a_t_max with bar labels (15.7 / 15.7 / 72.5)
    for i in range(len(x)):
        b = axes[2].bar(x[i], at_maxs[i], width=0.55, color=colors[i], edgecolor="black", hatch=hatches[i])
        axes[2].bar_label(b, fmt="%.1f", padding=2, fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, fontsize=8)
    axes[2].set_ylim(0, max(at_maxs) * 1.22)
    axes[2].set_ylabel("$a_{t,max}$ (m/s$^2$)", fontsize=8.5)
    axes[2].set_title("(c) Max Edge $a_{t,max}$", fontsize=9, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.tick_params(axis="y", labelsize=8)

    # Shared legend at the top
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = os.path.join(VIS_DIR, "paper_fig04_profile_bars_1x3.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


# Paper Figure 6 (1x2): Multi-step process curve and step breakdown (slightly more compact height)
def plot_combined_fig5():
    df_ms = load_csv("case_300mm_multistep_results.csv")

    durations = [1.0, 5.0, 2.5, 30.0, 3.0]
    step_names = ["S1:Ramp1", "S2:Hold1", "S3:Ramp2", "S4:Hold2", "S5:Stop"]

    json_path = os.path.join(INPUT_DIR, "case_300mm_multistep.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ["steps", "segments", "recipe", "process_steps"]:
            if key in cfg and isinstance(cfg[key], list) and len(cfg[key]) > 0:
                if "duration" in cfg[key][0]:
                    durations = [float(s["duration"]) for s in cfg[key]]
                break

    t_milestones = np.cumsum([0.0] + durations)
    theta_at_m = np.interp(t_milestones, df_ms["time_t"].to_numpy(), df_ms["theta"].to_numpy())
    revolutions = np.diff(theta_at_m) / (2.0 * np.pi)

    if len(step_names) != len(durations):
        step_names = [f"S{i+1}" for i in range(len(durations))]

    # Slightly reduced height (2.65 in) to save vertical space in the paper
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.65))

    # (a) Multi-step spin speed (rpm) and motor torque vs. time
    ax_a1 = axes[0]
    ax_a2 = ax_a1.twinx()
    l1 = ax_a1.plot(df_ms["time_t"], df_ms["rpm"], "-", color="tab:blue", linewidth=1.6, label="Speed (rpm)")
    l2 = ax_a2.plot(df_ms["time_t"], df_ms["torque"], "--", color="tab:orange", linewidth=1.4, label="Torque $\\tau$ (N$\\cdot$m)")
    ax_a1.set_xlabel("Time $t$ (s)", fontsize=8.5)
    ax_a1.set_ylabel("Spin Speed (rpm)", color="tab:blue", fontsize=8.5)
    ax_a2.set_ylabel("Torque $\\tau$ (N$\\cdot$m)", color="tab:orange", fontsize=8.5)
    ax_a1.tick_params(labelsize=8)
    ax_a2.tick_params(labelsize=8)
    ax_a1.set_title("(a) Multi-Step Speed & Torque", fontsize=9, fontweight="bold")
    ax_a1.grid(True, linestyle="--", alpha=0.5)

    # (b) Step breakdown: revolutions and duration bars
    x = np.arange(len(step_names))
    w = 0.36
    ax_b1 = axes[1]
    ax_b2 = ax_b1.twinx()
    b1 = ax_b1.bar(x - w/2, revolutions, width=w, color="royalblue", edgecolor="black", label="Revolutions (rev)")
    b2 = ax_b2.bar(x + w/2, durations, width=w, color="sandybrown", edgecolor="black", hatch="//", label="Duration (s)")
    ax_b1.set_xticks(x)
    ax_b1.set_xticklabels(step_names, rotation=20, fontsize=8)
    ax_b1.set_xlabel("Process Steps", fontsize=8.5)
    ax_b1.set_ylabel("Revolutions $N$ (rev)", color="royalblue", fontsize=8.5)
    ax_b2.set_ylabel("Duration $\\Delta t$ (s)", color="sandybrown", fontsize=8.5)
    ax_b1.tick_params(axis="y", labelsize=8)
    ax_b2.tick_params(axis="y", labelsize=8)
    ax_b1.set_title("(b) Step Breakdown", fontsize=9, fontweight="bold")
    ax_b1.grid(True, linestyle="--", alpha=0.5)

    # Shared legend at the top
    handles = [l1[0], l2[0], b1[0], b2[0]]
    labels = ["Speed (rpm)", "Torque $\\tau$ (N$\\cdot$m)", "Revolutions (rev)", "Duration (s)"]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    save_path = os.path.join(VIS_DIR, "paper_fig05_multistep_1x2.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)

# Run main program
if __name__ == "__main__":
    set_common_style()

    plot_combined_fig1()
    plot_combined_fig2()
    plot_combined_fig3()
    plot_combined_fig4()
    plot_combined_fig5()

    print("5 paper figures (Figure 2-6) have been generated in visualizations/ folder.")
