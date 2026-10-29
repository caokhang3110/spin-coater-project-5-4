# Draw 5 figures for the report (Figure2 2-6) from data/outputs

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

OUTPUT_DIR = "data/outputs"
INPUT_DIR = "data/inputs"
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)


def set_common_style():
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 13


def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}. Run python main.py first.")
    return pd.read_csv(path)


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
    threshold = 0.99 * target_rpm if "exp" in case_name else 0.999 * target_rpm
    reached = df_case[df_case["rpm"] >= threshold]
    if len(reached) > 0:
        return float(reached["time_t"].iloc[0])
    return float(row["t_end_s"])


# 5 COMBINED MULTI-PANEL FIGURES FOR WRITTEN PAPER (FIGURES 2 TO 6)

# Paper Figure 2 (2x2): Kinematics theta, omega, alpha, jerk (symlog scale)
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

    for df, ls, lbl in datasets:
        axes[0, 0].plot(df["time_t"], df["theta"], linestyle=ls, linewidth=1.6, label=lbl)
    axes[0, 0].set_ylabel("Angle $\\theta$ (rad)", fontsize=8.5)
    axes[0, 0].set_title("(a) Angular Position $\\theta(t)$", fontsize=9, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    for df, ls, _ in datasets:
        axes[0, 1].plot(df["time_t"], df["omega"], linestyle=ls, linewidth=1.6)
    axes[0, 1].set_ylabel("Velocity $\\omega$ (rad/s)", fontsize=8.5)
    axes[0, 1].set_title("(b) Angular Velocity $\\omega(t)$", fontsize=9, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    for df, ls, _ in datasets:
        axes[1, 0].plot(df["time_t"], df["alpha"], linestyle=ls, linewidth=1.6)
    axes[1, 0].set_xlabel("Time $t$ (s)", fontsize=8.5)
    axes[1, 0].set_ylabel("Accel. $\\alpha$ (rad/s$^2$)", fontsize=8.5)
    axes[1, 0].set_title("(c) Angular Acceleration $\\alpha(t)$", fontsize=9, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    for df, ls, _ in datasets:
        axes[1, 1].plot(df["time_t"], df["jerk"], linestyle=ls, linewidth=1.6)
    axes[1, 1].set_yscale("symlog", linthresh=100)
    axes[1, 1].set_xlabel("Time $t$ (s)", fontsize=8.5)
    axes[1, 1].set_ylabel("Jerk $j$ (rad/s$^3$, symlog)", fontsize=8.5)
    axes[1, 1].set_title("(d) Angular Jerk $j(t)$", fontsize=9, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    finish_and_save(fig, "paper_fig02_kinematics_2x2.png")


# Paper Figure 3 (2x2): Phase portrait, Trajectory, a_t/a_c, and Energy error (%)
def plot_combined_fig2():
    df_sum = load_csv("summary.csv")
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6))

    axes[0, 0].plot(df_lin["theta"], df_lin["omega"], "-", linewidth=1.5, label="Linear")
    axes[0, 0].plot(df_scv["theta"], df_scv["omega"], "--", linewidth=1.5, label="S-curve")
    axes[0, 0].plot(df_exp["theta"], df_exp["omega"], ":", linewidth=1.7, label="Exponential")
    axes[0, 0].set_xlabel("Angle $\\theta$ (rad)", fontsize=8.5)
    axes[0, 0].set_ylabel("Velocity $\\omega$ (rad/s)", fontsize=8.5)
    axes[0, 0].set_title("(a) Phase Portrait $\\omega(\\theta)$", fontsize=9, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

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
    
    axes[1, 1].text(0.55, 0.12, "Linear: exact ($\\approx 0\\%$)", transform=axes[1, 1].transAxes,
                    fontsize=8, fontstyle="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    axes[1, 1].legend(loc="upper left", fontsize=7.5, frameon=True)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)

    h_a, l_a = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_a + l1 + l2, l_a + ["$a_t$ (left)", "$a_c$ (right)"],
               loc="upper center", ncol=5, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    finish_and_save(fig, "paper_fig03_phase_traj_edge_err_2x2.png")


# Paper Figure 4 (1x3): Parametric dependencies (t_ramp, a_c, film thickness)
def plot_combined_fig3():
    df_sum = load_csv("summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.55))

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
    finish_and_save(fig, "paper_fig04_parametric_1x3.png")


# Paper Figure 5 (1x3 Bar Chart): Profile comparison
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

    for i in range(len(x)):
        b = axes[0].bar(x[i], t_ramps[i], width=0.55, color=colors[i], edgecolor="black", hatch=hatches[i], label=labels[i])
        axes[0].bar_label(b, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, fontsize=8)
    axes[0].set_ylim(0, max(t_ramps) * 1.22)
    axes[0].set_ylabel("Time $t_{ramp}$ (s)", fontsize=8.5)
    axes[0].set_title("(a) Ramp Time $t_{ramp}$", fontsize=9, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

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

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    finish_and_save(fig, "paper_fig05_profile_bars_1x3.png")


# Paper Figure 6 (1x2): Multi-step process recipe and breakdown
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

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.65))

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

    handles = [l1[0], l2[0], b1[0], b2[0]]
    labels = ["Speed (rpm)", "Torque $\\tau$ (N$\\cdot$m)", "Revolutions (rev)", "Duration (s)"]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    finish_and_save(fig, "paper_fig06_multistep_1x2.png")

# Generate 5 figures


if __name__ == "__main__":
    set_common_style()

    plot_combined_fig1()
    plot_combined_fig2()
    plot_combined_fig3()
    plot_combined_fig4()
    plot_combined_fig5()

    print("All 5 paper figures (Figure 2 to Figure 6) have been generated successfully.")
