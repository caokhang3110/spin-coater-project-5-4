# File: plots.py
# Vai trò: #6 Code C - Đồ thị, biểu đồ và hình
# Nhiệm vụ: Đọc dữ liệu từ file CSV trong data/outputs và vẽ đủ 13 hình (300 dpi)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Tạo thư mục lưu dữ liệu và hình ảnh nếu chưa có
os.makedirs("data/outputs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


# B1: Hàm định dạng chung cho tất cả biểu đồ (cỡ chữ, lưới, kích thước)
def set_common_style():
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 13


# Hàm phụ trợ để bật lưới và lưu hình 300 dpi cho gọn
def finish_and_save(fig, filename):
    fig.tight_layout()
    save_path = os.path.join("visualizations", filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("Saved:", save_path)

# B3: Vẽ 10 đồ thi (grahps) - only read from file csv

# Hình 1: Vận tốc góc omega(t) của 3 profile tăng tốc
def plot_fig01():
    df_lin = pd.read_csv("data/outputs/case_300mm_linear_1000.csv")
    df_scv = pd.read_csv("data/outputs/case_300mm_scurve_1000.csv")
    df_exp = pd.read_csv("data/outputs/case_300mm_exp_1000.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Dùng kiểu nét khác nhau (-, --, :) để in trắng đen vẫn phân biệt được
    ax.plot(df_lin["time_t"], df_lin["omega"], "-", linewidth=2, label="Linear profile")
    ax.plot(df_scv["time_t"], df_scv["omega"], "--", linewidth=2, label="S-curve profile")
    ax.plot(df_exp["time_t"], df_exp["omega"], ":", linewidth=2, label="Exponential profile")

    ax.set_title("Angular Velocity vs. Time for Three Acceleration Profiles")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Angular Velocity $\\omega$ (rad/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig01_omega_profiles.png")


# Hình 2: Gia tốc góc alpha(t) và độ giật góc jerk(t)
def plot_fig02():
    df_lin = pd.read_csv("data/outputs/case_300mm_linear_1000.csv")
    df_scv = pd.read_csv("data/outputs/case_300mm_scurve_1000.csv")
    df_exp = pd.read_csv("data/outputs/case_300mm_exp_1000.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    # Khung trên vẽ gia tốc góc alpha
    axes[0].plot(df_lin["time_t"], df_lin["alpha"], "-", label="Linear")
    axes[0].plot(df_scv["time_t"], df_scv["alpha"], "--", label="S-curve")
    axes[0].plot(df_exp["time_t"], df_exp["alpha"], ":", label="Exponential")
    axes[0].set_ylabel("Angular Accel. $\\alpha$ (rad/s$^2$)")
    axes[0].set_title("Angular Acceleration and Angular Jerk vs. Time")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Khung dưới vẽ độ giật jerk
    axes[1].plot(df_lin["time_t"], df_lin["jerk"], "-", label="Linear")
    axes[1].plot(df_scv["time_t"], df_scv["jerk"], "--", label="S-curve")
    axes[1].plot(df_exp["time_t"], df_exp["jerk"], ":", label="Exponential")
    axes[1].set_xlabel("Time t (s)")
    axes[1].set_ylabel("Angular Jerk j (rad/s$^3$)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig02_alpha_jerk_profiles.png")


# Hình 3: Hình nhiều khung (multi-pane) cho theta(t), omega(t), alpha(t)
def plot_fig03():
    df = pd.read_csv("data/outputs/case_300mm_scurve_1000.csv")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7))

    axes[0].plot(df["time_t"], df["theta"], "-", color="blue", label="$\\theta(t)$")
    axes[0].set_ylabel("Angle $\\theta$ (rad)")
    axes[0].set_title("Multi-pane Angular Kinematics (S-curve Profile)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    axes[1].plot(df["time_t"], df["omega"], "--", color="darkorange", label="$\\omega(t)$")
    axes[1].set_ylabel("Velocity $\\omega$ (rad/s)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")

    axes[2].plot(df["time_t"], df["alpha"], "-.", color="green", label="$\\alpha(t)$")
    axes[2].set_xlabel("Time t (s)")
    axes[2].set_ylabel("Accel. $\\alpha$ (rad/s$^2$)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    finish_and_save(fig, "fig03_multipane_theta_omega_alpha.png")


# Hình 4: Giản đồ pha (Phase Portrait) omega theo theta
def plot_fig04():
    df_lin = pd.read_csv("data/outputs/case_300mm_linear_1000.csv")
    df_scv = pd.read_csv("data/outputs/case_300mm_scurve_1000.csv")
    df_exp = pd.read_csv("data/outputs/case_300mm_exp_1000.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_lin["theta"], df_lin["omega"], "-", linewidth=2, label="Linear")
    ax.plot(df_scv["theta"], df_scv["omega"], "--", linewidth=2, label="S-curve")
    ax.plot(df_exp["theta"], df_exp["omega"], ":", linewidth=2, label="Exponential")

    ax.set_title("Phase Portrait: Angular Velocity $\\omega$ vs. Angle $\\theta$")
    ax.set_xlabel("Angular Position $\\theta$ (rad)")
    ax.set_ylabel("Angular Velocity $\\omega$ (rad/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig04_phase_portrait_omega_theta.png")


# Hình 5: Gia tốc tiếp tuyến a_t và gia tốc hướng tâm a_c ở mép wafer theo t
def plot_fig05():
    df = pd.read_csv("data/outputs/case_300mm_scurve_1000.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5.5))

    axes[0].plot(df["time_t"], df["a_t_edge"], "-", color="red", label="$a_t = \\alpha R$")
    axes[0].set_ylabel("Tangential Accel. $a_t$ (m/s$^2$)")
    axes[0].set_title("Edge Tangential and Centripetal Accelerations vs. Time (R = 0.15 m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(df["time_t"], df["a_c_edge"], "--", color="purple", label="$a_c = \\omega^2 R$")
    axes[1].set_xlabel("Time t (s)")
    axes[1].set_ylabel("Centripetal Accel. $a_c$ (m/s$^2$)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig05_edge_at_ac_vs_time.png")


# Hình 6: Thời gian tăng tốc t_ramp theo tốc độ tăng tốc
def plot_fig06():
    df_sum = pd.read_csv("data/outputs/summary.csv")
    df_b = df_sum[df_sum["group"] == "GroupB"]

    df_lin = df_b[df_b["profile"] == "Linear"]
    df_scv = df_b[df_b["profile"] == "S-curve"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_lin["accel_rpm_s"], df_lin["t_ramp_s"], "-o", linewidth=2, label="Linear profile")
    ax.plot(df_scv["accel_rpm_s"], df_scv["t_ramp_s"], "--s", linewidth=2, label="S-curve profile")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Ramp-up Time $t_{ramp}$ vs. Acceleration Rate (Target 3000 rpm)")
    ax.set_xlabel("Acceleration Rate (rpm/s)")
    ax.set_ylabel("Ramp-up Time $t_{ramp}$ (s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig06_tramp_vs_accel_rate.png")


# Hình 7: Gia tốc hướng tâm a_c theo rpm cho các cỡ wafer khác nhau
def plot_fig07():
    df_w = pd.read_csv("data/outputs/wafer_size_sweep.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    w100 = df_w[df_w["diameter_mm"] == 100]
    w150 = df_w[df_w["diameter_mm"] == 150]
    w200 = df_w[df_w["diameter_mm"] == 200]
    w300 = df_w[df_w["diameter_mm"] == 300]

    ax.plot(w100["rpm"], w100["a_c_edge"], "-o", markevery=5, label="Wafer 100 mm")
    ax.plot(w150["rpm"], w150["a_c_edge"], "--s", markevery=5, label="Wafer 150 mm")
    ax.plot(w200["rpm"], w200["a_c_edge"], "-.^", markevery=5, label="Wafer 200 mm")
    ax.plot(w300["rpm"], w300["a_c_edge"], ":d", markevery=5, label="Wafer 300 mm")

    ax.set_title("Edge Centripetal Acceleration $a_c$ vs. Spin Speed for Wafer Sizes")
    ax.set_xlabel("Spin Speed (rpm)")
    ax.set_ylabel("Centripetal Acceleration $a_c$ (m/s$^2$)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig07_ac_vs_rpm_wafer_sizes.png")


# Hình 8: Sai số năng lượng theo thời gian (vẽ trên thang đo log)
def plot_fig08():
    df_lin = pd.read_csv("data/outputs/case_300mm_linear_1000.csv")
    df_scv = pd.read_csv("data/outputs/case_300mm_scurve_1000.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_lin["time_t"], df_lin["energy_error_abs"], "-", label="Linear (RK45)")
    ax.plot(df_scv["time_t"], df_scv["energy_error_abs"], "--", label="S-curve (RK45)")

    # Đặt trục y theo thang log theo đúng hướng dẫn Bước 7
    ax.set_yscale("log")
    ax.set_title("Work-Energy Theorem Numerical Error $|\\int \\tau d\\theta - \\Delta K|$")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Absolute Energy Error (J) [Log Scale]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig08_energy_error_log.png")


# Hình 9: Độ dày màng phủ photoresist h theo vận tốc góc omega
def plot_fig09():
    df_h = pd.read_csv("data/outputs/film_thickness_sweep.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_h["omega"], df_h["thickness_um"], "-o", markevery=4, linewidth=2,
            label="Meyerhofer Model ($h \\propto \\omega^{-1/2}$)")

    ax.set_title("Photoresist Film Thickness $h$ vs. Angular Velocity $\\omega$")
    ax.set_xlabel("Steady Angular Velocity $\\omega$ (rad/s)")
    ax.set_ylabel("Final Film Thickness $h$ ($\\mu$m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    finish_and_save(fig, "fig09_film_thickness_vs_omega.png")


# Hình 10: Toàn bộ quy trình spin-coating nhiều bước
def plot_fig10():
    df_ms = pd.read_csv("data/outputs/case_multistep_recipe.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5))

    axes[0].plot(df_ms["time_t"], df_ms["rpm"], "-", color="blue", linewidth=2, label="Spin Speed (rpm)")
    axes[0].set_ylabel("Spin Speed (rpm)")
    axes[0].set_title("Multi-Step Spin-Coating Process Recipe (Spread -> Spin -> Stop)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(df_ms["time_t"], df_ms["torque"], "--", color="darkorange", linewidth=2, label="Motor Torque $\\tau$")
    axes[1].set_xlabel("Time t (s)")
    axes[1].set_ylabel("Torque $\\tau$ (N$\\cdot$m)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig10_multistep_recipe_full.png")


# B4: Vẽ 1 hình quỹ đạo (trajectory) kèm vector gia tốc

def plot_fig11():
    df = pd.read_csv("data/outputs/case_trajectory_short.csv")

    x = df["x_edge"].to_numpy()
    y = df["y_edge"].to_numpy()
    ax_vec = df["ax_edge"].to_numpy()
    ay_vec = df["ay_edge"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(x, y, "-", color="blue", linewidth=2, label="Edge Trajectory $(R\\cos\\theta, R\\sin\\theta)$")
    ax.scatter(x[0], y[0], color="green", s=60, label="Start Point (t = 0)")

    # Lấy một vài điểm cách đều trên quỹ đạo để vẽ mũi tên vectơ gia tốc
    idx = [2, 10, 20, 30, 40, 50, 58]
    ax.quiver(x[idx], y[idx], ax_vec[idx], ay_vec[idx],
              angles="xy", scale_units="xy", scale=180.0, color="red",
              label="Total Acceleration Vector $\\vec{a}$")

    # Giữ tỉ lệ 2 trục bằng nhau để hình tròn không bị méo
    ax.set_aspect("equal")
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(-0.22, 0.22)
    ax.set_title("Wafer Edge Point Trajectory and Acceleration Vectors (R = 0.15 m)")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    finish_and_save(fig, "fig11_edge_trajectory_vectors.png")


# B5: Biểu đồ cột

# Hình 12: Biểu đồ cột 1 - So sánh t_ramp, jerk_max, a_t_max của 3 profile
def plot_fig12():
    df_sum = pd.read_csv("data/outputs/summary.csv")
    df_a = df_sum[df_sum["group"] == "GroupA"]

    profiles = df_a["profile"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4))

    axes[0].bar(profiles, df_a["t_ramp_s"], color="steelblue", edgecolor="black")
    axes[0].set_title("Ramp Time $t_{ramp}$")
    axes[0].set_ylabel("Time (s)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].bar(profiles, df_a["jerk_max"], color="seagreen", edgecolor="black")
    axes[1].set_yscale("log")
    axes[1].set_title("Max Jerk $j_{max}$")
    axes[1].set_ylabel("Jerk (rad/s$^3$, log scale)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].bar(profiles, df_a["a_t_max"], color="indianred", edgecolor="black")
    axes[2].set_title("Max Edge Tangential $a_{t,max}$")
    axes[2].set_ylabel("Acceleration (m/s$^2$)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Comparison of Key Metrics Across Three Acceleration Profiles")
    finish_and_save(fig, "fig12_bar_profile_comparison.png")


# Hình 13: Biểu đồ cột 2 - Số vòng quay và thời gian của từng bước trong quy trình
def plot_fig13():
    df_steps = pd.read_csv("data/outputs/multistep_steps_summary.csv")

    steps = df_steps["step_name"].tolist()
    x = np.arange(len(steps))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5))

    axes[0].bar(x, df_steps["revolutions"], width=0.5, color="royalblue", edgecolor="black", label="Revolutions (rev)")
    axes[0].set_ylabel("Number of Revolutions")
    axes[0].set_title("Revolutions and Duration Breakdown for Multi-Step Process")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].bar(x, df_steps["duration_s"], width=0.5, color="sandybrown", edgecolor="black", label="Step Duration (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(steps, rotation=15)
    axes[1].set_xlabel("Process Steps")
    axes[1].set_ylabel("Duration (s)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig13_bar_multistep_breakdown.png")


# Trường hợp mẫu

def create_sample_csv_files():
    if os.path.exists("data/outputs/case_300mm_linear_1000.csv"):
        return

    print("Creating sample CSV files in data/outputs/ for testing...")
    R = 0.15
    rho_si = 2330.0
    d_si = 0.000775
    m_wafer = rho_si * np.pi * (R**2) * d_si
    I_total = 0.5 * m_wafer * (R**2) + 0.015

    omega_f = 3000.0 * 2 * np.pi / 60.0
    alpha_0 = 1000.0 * 2 * np.pi / 60.0
    t = np.linspace(0, 4.0, 801)

    # 1. Tính dữ liệu mẫu cho profile tuyến tính (Linear)
    omega_lin = np.minimum(alpha_0 * t, omega_f)
    alpha_lin = np.where(t <= 3.0, alpha_0, 0.0)
    theta_lin = cumulative_trapezoid(omega_lin, t, initial=0)
    jerk_lin = np.gradient(alpha_lin, t)

    # 2. Tính dữ liệu mẫu cho trường hợp S-curve
    t_j = 0.6
    T_scv = 3.6
    j_max = alpha_0 / t_j
    alpha_scv = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] < t_j:
            alpha_scv[i] = j_max * t[i]
        elif t[i] < 3.0:
            alpha_scv[i] = alpha_0
        elif t[i] <= T_scv:
            alpha_scv[i] = alpha_0 - j_max * (t[i] - 3.0)
        else:
            alpha_scv[i] = 0.0

    omega_scv = cumulative_trapezoid(alpha_scv, t, initial=0)
    theta_scv = cumulative_trapezoid(omega_scv, t, initial=0)
    jerk_scv = np.gradient(alpha_scv, t)

    # 3. Tính dữ liệu mẫu cho trường hợp hàm mũ (Exponential)
    T_m = 3.0 / 4.60517
    omega_exp = omega_f * (1.0 - np.exp(-t / T_m))
    alpha_exp = (omega_f - omega_exp) / T_m
    theta_exp = cumulative_trapezoid(omega_exp, t, initial=0)
    jerk_exp = np.gradient(alpha_exp, t)

    # Hàm lưu bảng CSV đúng tên cột chuẩn của nhóm
    def save_csv(fname, t_arr, th, om, al, jk, err_factor):
        at = al * R
        ac = (om**2) * R
        atot = np.sqrt(at**2 + ac**2)
        ke = 0.5 * I_total * (om**2)
        tau = I_total * al
        pwr = tau * om
        err = err_factor * (1e-12 + 1e-10 * (t_arr / 4.0)**2)

        df = pd.DataFrame({
            "time_t": t_arr,
            "theta": th,
            "omega": om,
            "rpm": om * 60.0 / (2 * np.pi),
            "alpha": al,
            "jerk": jk,
            "a_t_edge": at,
            "a_c_edge": ac,
            "a_total": atot,
            "kinetic_energy": ke,
            "torque": tau,
            "power": pwr,
            "energy_error_abs": err
        })
        df.to_csv(os.path.join("data/outputs", fname), index=False)

    save_csv("case_300mm_linear_1000.csv", t, theta_lin, omega_lin, alpha_lin, jerk_lin, 1.0)
    save_csv("case_300mm_scurve_1000.csv", t, theta_scv, omega_scv, alpha_scv, jerk_scv, 2.5)
    save_csv("case_300mm_exp_1000.csv", t, theta_exp, omega_exp, alpha_exp, jerk_exp, 1.8)

    # Tạo file summary.csv
    summary_rows = [
        {"group": "GroupA", "profile": "Linear", "accel_rpm_s": 1000, "t_ramp_s": 3.0, "jerk_max": float(np.max(np.abs(jerk_lin))), "a_t_max": alpha_0 * R},
        {"group": "GroupA", "profile": "S-curve", "accel_rpm_s": 1000, "t_ramp_s": 3.6, "jerk_max": j_max, "a_t_max": alpha_0 * R},
        {"group": "GroupA", "profile": "Exponential", "accel_rpm_s": 1000, "t_ramp_s": 3.0, "jerk_max": float(np.max(np.abs(jerk_exp))), "a_t_max": float(np.max(alpha_exp) * R)},
    ]
    for rate in [500, 1000, 2000, 5000, 10000, 20000]:
        t_lin = 3000.0 / rate
        summary_rows.append({"group": "GroupB", "profile": "Linear", "accel_rpm_s": rate, "t_ramp_s": t_lin, "jerk_max": 0, "a_t_max": 0})
        summary_rows.append({"group": "GroupB", "profile": "S-curve", "accel_rpm_s": rate, "t_ramp_s": t_lin * 1.2, "jerk_max": 0, "a_t_max": 0})
    pd.DataFrame(summary_rows).to_csv("data/outputs/summary.csv", index=False)

    # Tạo file wafer_size_sweep.csv
    rpms = np.linspace(500, 6000, 25)
    w_rows = []
    for d_mm in [100, 150, 200, 300]:
        r_m = d_mm / 2000.0
        for r_val in rpms:
            om_val = r_val * 2 * np.pi / 60.0
            w_rows.append({"diameter_mm": d_mm, "rpm": r_val, "a_c_edge": (om_val**2) * r_m})
    pd.DataFrame(w_rows).to_csv("data/outputs/wafer_size_sweep.csv", index=False)

    # Tạo file film_thickness_sweep.csv
    om_sweep = np.linspace(50, 630, 30)
    h_um = 25.0 * (om_sweep ** (-0.5))
    pd.DataFrame({"omega": om_sweep, "thickness_um": h_um}).to_csv("data/outputs/film_thickness_sweep.csv", index=False)

    # Tạo file case_multistep_recipe.csv
    t_ms = np.linspace(0, 42.0, 841)
    rpm_ms = np.zeros_like(t_ms)
    for i in range(len(t_ms)):
        ti = t_ms[i]
        if ti < 1.0:
            rpm_ms[i] = 500.0 * ti
        elif ti < 6.0:
            rpm_ms[i] = 500.0
        elif ti < 8.5:
            rpm_ms[i] = 500.0 + 1000.0 * (ti - 6.0)
        elif ti < 38.5:
            rpm_ms[i] = 3000.0
        elif ti <= 41.5:
            rpm_ms[i] = 3000.0 - 1000.0 * (ti - 38.5)
        else:
            rpm_ms[i] = 0.0
    om_ms = rpm_ms * 2 * np.pi / 60.0
    al_ms = np.gradient(om_ms, t_ms)
    pd.DataFrame({"time_t": t_ms, "rpm": rpm_ms, "torque": I_total * al_ms}).to_csv("data/outputs/case_multistep_recipe.csv", index=False)

    # Tạo file multistep_steps_summary.csv
    pd.DataFrame({
        "step_name": ["1. Ramp 500", "2. Hold 500", "3. Ramp 3000", "4. Hold 3000", "5. Spin Down"],
        "duration_s": [1.0, 5.0, 2.5, 30.0, 3.0],
        "revolutions": [4.17, 41.67, 72.92, 1500.0, 75.0]
    }).to_csv("data/outputs/multistep_steps_summary.csv", index=False)

    # Tạo file case_trajectory_short.csv cho hình quỹ đạo
    t_tr = np.linspace(0, 0.34, 60)
    om_tr = alpha_0 * t_tr
    th_tr = 0.5 * alpha_0 * (t_tr**2)
    x_e = R * np.cos(th_tr)
    y_e = R * np.sin(th_tr)
    at_tr = alpha_0 * R
    ac_tr = (om_tr**2) * R
    ax_e = -at_tr * np.sin(th_tr) - ac_tr * np.cos(th_tr)
    ay_e = at_tr * np.cos(th_tr) - ac_tr * np.sin(th_tr)
    pd.DataFrame({
        "time_t": t_tr, "theta": th_tr, "x_edge": x_e, "y_edge": y_e, "ax_edge": ax_e, "ay_edge": ay_e
    }).to_csv("data/outputs/case_trajectory_short.csv", index=False)


# Chạy chương trình chính
if __name__ == "__main__":
    set_common_style()
    create_sample_csv_files()

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
