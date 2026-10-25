# File: plots.py
# Role: #6 Code C - Graphs, charts, and figures
# Task: Read data from CSV files in data/outputs and generate all 13 figures (300 dpi)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    if "t_ramp_s" in row and pd.notna(row["t_ramp_s"]):
        return float(row["t_ramp_s"])

    case_name = row["case"]
    df_case = load_csv(f"{case_name}_results.csv")
    # For exponential profile, use 99% of target speed (~4.6 * tau_m)
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


# Figure 2: Angular acceleration alpha(t) and angular jerk j(t)
def plot_fig02():
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    # Top panel: angular acceleration alpha
    axes[0].plot(df_lin["time_t"], df_lin["alpha"], "-", linewidth=1.8, label="Linear")
    axes[0].plot(df_scv["time_t"], df_scv["alpha"], "--", linewidth=1.8, label="S-curve")
    axes[0].plot(df_exp["time_t"], df_exp["alpha"], ":", linewidth=2.0, label="Exponential")
    axes[0].set_ylabel("Angular Accel. $\\alpha$ (rad/s$^2$)")
    axes[0].set_title("Angular Acceleration $\\alpha(t)$ and Angular Jerk $j(t)$ vs. Time")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Bottom panel: angular jerk j
    axes[1].plot(df_lin["time_t"], df_lin["jerk"], "-", linewidth=1.8, label="Linear")
    axes[1].plot(df_scv["time_t"], df_scv["jerk"], "--", linewidth=1.8, label="S-curve")
    axes[1].plot(df_exp["time_t"], df_exp["jerk"], ":", linewidth=2.0, label="Exponential")
    axes[1].set_xlabel("Time $t$ (s)")
    axes[1].set_ylabel("Angular Jerk $j$ (rad/s$^3$)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

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

    # Select target cases including extreme low and high acceleration cases
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
        # Extract numerical acceleration rate (rpm/s) from the end of the case name
        rate_val = float(c_name.split("_")[-1])
        t_val = get_t_ramp_from_summary_or_results(row)
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
    omega_vals = df_valid["omega"].to_numpy()

    # Meyerhofer scaling law: h / h_0 = (omega / omega_0)^(-1/2)
    h_ratio = (omega_vals / omega_0) ** (-0.5)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(omega_vals, h_ratio, "-o", markevery=40, linewidth=2,
            label="Meyerhofer Model ($h/h_0 = (\\omega/\\omega_0)^{-1/2}$)")

    ax.set_title("Relative Photoresist Film Thickness $h/h_0$ vs. Angular Velocity $\\omega$")
    ax.set_xlabel("Angular Velocity $\\omega$ (rad/s)")
    ax.set_ylabel("$h/h_0$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig09_film_thickness_vs_omega.png")


# Figure 10: Complete multi-step spin-coating process recipe
def plot_fig10():
    df_ms = load_csv("case_300mm_multistep_results.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5))

    axes[0].plot(df_ms["time_t"], df_ms["rpm"], "-", color="blue", linewidth=2, label="Spin Speed (rpm)")
    axes[0].set_ylabel("Spin Speed (rpm)")
    axes[0].set_title("Multi-Step Spin-Coating Process Recipe (Spread -> Spin -> Stop)")
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
        t_ramps.append(get_t_ramp_from_summary_or_results(row))
        jerk_maxs.append(float(row["jerk_max"]))
        at_maxs.append(float(row["a_t_max"]))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 4.2))

    # Subplot 1: t_ramp
    axes[0].bar(x, t_ramps, color="steelblue", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Ramp Time $t_{ramp}$")
    axes[0].set_ylabel("Time (s)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: jerk_max (log scale)
    axes[1].bar(x, jerk_maxs, color="seagreen", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_yscale("log")
    axes[1].set_title("Max Jerk $j_{max}$")
    axes[1].set_ylabel("Jerk (rad/s$^3$, log scale)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: a_t_max
    axes[2].bar(x, at_maxs, color="indianred", edgecolor="black")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Max Edge Tangential $a_{t,max}$")
    axes[2].set_ylabel("Acceleration (m/s$^2$)")
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
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5))

    axes[0].bar(x, revolutions, width=0.5, color="royalblue", edgecolor="black", label="Revolutions (rev)")
    axes[0].set_ylabel("Revolutions $N$ (rev)")
    axes[0].set_title("Revolutions and Step Duration Breakdown for Multi-Step Process")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].bar(x, durations, width=0.5, color="sandybrown", edgecolor="black", label="Step Duration (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(step_names, rotation=15)
    axes[1].set_xlabel("Process Steps")
    axes[1].set_ylabel("Duration $\\Delta t$ (s)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig13_bar_multistep_breakdown.png")


# Run main program
if __name__ == "__main__":
    set_common_style()

    plot_fig01()
    plot_fig02()
    plot_fig03()
    plot_fig04()
    plot_fig05()
    plot_fig06()
    plot_fig07()
    plot_fig08()
    plot_fig09()
    plot_fig10()
    plot_fig11()
    plot_fig12()
    plot_fig13()

    print("13 figures have been generated in visualizations/ folder.")
