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

def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}. Run python main.py first."
        )
    return pd.read_csv(path)

# Hàm phụ trợ để bật lưới và lưu hình 300 dpi cho gọn
def finish_and_save(fig, filename):
    fig.tight_layout()
    save_path = os.path.join("visualizations", filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("Saved:", save_path)

def get_t_ramp_from_summary_or_results(row, target_rpm=3000.0):
    if "t_ramp_s" in row and pd.notna(row["t_ramp_s"]):
        return float(row["t_ramp_s"])

    case_name = row["case"]
    df_case = load_csv(f"{case_name}_results.csv")
    # Không có hàm mũ thì dùng 99% tốc độ (~4.6 * T_m)
    threshold = 0.99 * target_rpm if "exp" in case_name else 0.999 * target_rpm
    reached = df_case[df_case["rpm"] >= threshold]
    if len(reached) > 0:
        return float(reached["time_t"].iloc[0])
    return float(row["t_end_s"])

# B3: Vẽ 10 đồ thi (grahps) - only read from file csv

# Hình 1: Vận tốc góc omega(t) của 3 profile tăng tốc
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


# Hình 2: Gia tốc góc alpha(t) và độ giật góc jerk(t)
def plot_fig02():
    df_lin = load_csv("case_300mm_linear_1000_results.csv")
    df_scv = load_csv("case_300mm_scurve_1000_results.csv")
    df_exp = load_csv("case_300mm_exp_1000_results.csv")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    # Khung trên vẽ gia tốc góc alpha
    axes[0].plot(df_lin["time_t"], df_lin["alpha"], "-", linewidth=1.8, label="Linear")
    axes[0].plot(df_scv["time_t"], df_scv["alpha"], "--", linewidth=1.8, label="S-curve")
    axes[0].plot(df_exp["time_t"], df_exp["alpha"], ":", linewidth=2.0, label="Exponential")
    axes[0].set_ylabel("Angular Accel. $\\alpha$ (rad/s$^2$)")
    axes[0].set_title("Angular Acceleration $\\alpha(t)$ and Angular Jerk $j(t)$ vs. Time")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Khung dưới vẽ độ giật jerk
    axes[1].plot(df_lin["time_t"], df_lin["jerk"], "-", linewidth=1.8, label="Linear")
    axes[1].plot(df_scv["time_t"], df_scv["jerk"], "--", linewidth=1.8, label="S-curve")
    axes[1].plot(df_exp["time_t"], df_exp["jerk"], ":", linewidth=2.0, label="Exponential")
    axes[1].set_xlabel("Time $t$ (s)")
    axes[1].set_ylabel("Angular Jerk $j$ (rad/s$^3$)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    finish_and_save(fig, "fig02_alpha_jerk_profiles.png")


# Hình 3: Hình nhiều khung (multi-pane) cho theta(t), omega(t), alpha(t)
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


# Hình 4: Giản đồ pha (Phase Portrait) omega theo theta
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

# Hình 5: Gia tốc tiếp tuyến a_t và gia tốc hướng tâm a_c ở mép wafer theo t
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

# Hình 6: Thời gian tăng tốc t_ramp theo tốc độ tăng tốc
def plot_fig06():
    df_sum = load_csv("summary.csv")

    # Chọn case
    target_cases = [
        "case_300mm_extreme_low_accel_100",
        "case_300mm_linear_500",
        "case_300mm_linear_1000",
        "case_300mm_linear_2000",
        "case_300mm_linear_5000",
        "case_300mm_linear_10000",
        "case_300mm_extreme_high_accel_20000"
    ]

    accel_rates = []
    t_ramps = []

    for c_name in target_cases:
        matched = df_sum[df_sum["case"] == c_name]
        if len(matched) > 0:
            row = matched.iloc[0]
            # Trích xuất giá trị số của tốc độ tăng vòng quay (rpm/s) từ phần cuối tên trường hợp
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

# Hình 7: Gia tốc hướng tâm a_c theo rpm cho các cỡ wafer khác nhau
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


# Hình 8: Sai số năng lượng theo thời gian (vẽ trên thang đo log)
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


# Hình 9: Độ dày màng phủ photoresist h theo vận tốc góc omega
def plot_fig09():
    df_h = pd.read_csv("case_300mm_multistep_results.csv")

    # Tốc độ góc tại omega_0 = 500 rpm
    omega_0 = 500.0 * 2.0 * np.pi / 60.0
    df_valid = df[df["rpm"] >= 500.0].copy()
    omega_vals = df_valid["omega"].to_numpy()

    # Meyerhofer scaling law: h / h_0 = (omega / omega_0)^(-1/2)
    h_ratio = (omega_vals / omega_0) ** (-0.5)

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
    df_ms = load.csv("case_300mm_multistep_results.csv")

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
    df_sum = load_csv("summary.csv")
    df = load_csv("case_300mm_linear_1000_results.csv")

    # Lấy bán kính đĩa R_m từ summary.csv cho case_300mm_linear_1000
    matched = df_sum[df_sum["case"] == "case_300mm_linear_1000"]
    R = float(matched["R_m"].iloc[0]) if len(matched) > 0 else 0.15

    # B1: Lọc chỉ lấy giai đoạn đầu time_t <= 0.3s
    df_sub = df[df["time_t"] <= 0.3].copy()

    t = df_sub["time_t"].to_numpy()
    theta = df_sub["theta"].to_numpy()
    a_t = df_sub["a_t_edge"].to_numpy()
    a_c = df_sub["a_c_edge"].to_numpy()

    # B2: Nhập x và y
    x = R * np.cos(theta)
    y = R * np.sin(theta)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    # Vẽ đường viền hình tròn của toàn bộ tấm wafer để tham chiếu
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), "--", color="lightgray", linewidth=1.2,
            label=f"Wafer Outline ($R = {R:.2f}$ m)")

    # Vạch quỹ đạo trong 0,3 giây đầu tiên
    ax.plot(x, y, "-", color="tab:blue", linewidth=2.5,
            label="Trajectory of a point on the wafer edge ($t \\leq 0.3$ s)")
    ax.scatter(x[0], y[0], color="green", s=65, zorder=5, label="Start point ($t = 0$)")

    # B3: Chọn 10 thời điểm cách đều nhau trong khoảng từ 0 đến 0,3 giây.
    idx = np.linspace(0, len(t) - 1, 10, dtype=int)

    # B4: Các phương tiếp tuyến (-sin theta, cos theta) và hướng tâm (-cos theta, -sin theta)
    tx = -np.sin(theta[idx])
    ty = np.cos(theta[idx])
    cx = -np.cos(theta[idx])
    cy = -np.sin(theta[idx])

    ax_vec = a_t[idx] * tx + a_c[idx] * cx
    ay_vec = a_t[idx] * ty + a_c[idx] * cy

    # B5: Chuẩn hóa các vectơ về độ dài cố định là 0,04 m để tất cả các mũi tên đều nằm gọn trong khung hình
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
    ax.set_title(f"Edge-point trajectory and direction of the acceleration\nvector during the first 0.3 s (R = {R:.2f} m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9, frameon=True)

    finish_and_save(fig, "fig11_edge_trajectory_vectors.png")


# B5: Biểu đồ cột

# Hình 12: Biểu đồ cột 1 - So sánh t_ramp, jerk_max, a_t_max của 3 profile
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
    axes[0].bar(profiles, df_a["t_ramp_s"], color="steelblue", edgecolor="black")
    axes[0].set_title("Ramp Time $t_{ramp}$")
    axes[0].set_ylabel("Time (s)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: jerk_max (log scale)
    axes[1].bar(profiles, df_a["jerk_max"], color="seagreen", edgecolor="black")
    axes[1].set_yscale("log")
    axes[1].set_title("Max Jerk $j_{max}$")
    axes[1].set_ylabel("Jerk (rad/s$^3$, log scale)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: a_t_max
    axes[2].bar(profiles, df_a["a_t_max"], color="indianred", edgecolor="black")
    axes[2].set_title("Max Edge Tangential $a_{t,max}$")
    axes[2].set_ylabel("Acceleration (m/s$^2$)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Comparison of Key Metrics Across Three Acceleration Profiles")
    finish_and_save(fig, "fig12_bar_profile_comparison.png")


# Hình 13: Biểu đồ cột 2 - Số vòng quay và thời gian của từng bước trong quy trình
def plot_fig13():
 df_ms = load_csv("case_300mm_multistep_results.csv")

    # Các tên gọi và thời lượng bước mặc định từ quy cách công thức gồm nhiều bước
    durations = [1.0, 5.0, 2.5, 30.0, 3.0]
    step_names = [
        "1. Ramp 500 rpm",
        "2. Hold 500 rpm",
        "3. Ramp 3000 rpm",
        "4. Hold 3000 rpm",
        "5. Spin Down"
    ]

    # Đọc các khoảng thời gian từ tệp data/inputs/case_300mm_multistep.json nếu tệp này tồn tại
    json_path = os.path.join(INPUT_DIR, "case_300mm_multistep.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ["steps", "segments", "recipe", "process_steps"]:
            if key in cfg and isinstance(cfg[key], list) and len(cfg[key]) > 0:
                if "duration" in cfg[key][0]:
                    durations = [float(s["duration"]) for s in cfg[key]]
                break

    # Tính số vòng quay = delta_theta / (2 * pi) tại mỗi mốc thời gian
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

'''
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
'''

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
