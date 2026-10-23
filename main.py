import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

from physics_model import calc_physics
from solvers import solve_motion, check_error


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg["R"] <= 0 or cfg["thickness"] <= 0 or cfg["density"] <= 0:
        raise ValueError(f"{path}: Tham số wafer (R, thickness, density) phải > 0")
    if cfg["chuck_mass"] <= 0 or cfg["chuck_R"] <= 0:
        raise ValueError(f"{path}: Tham số mâm hút (chuck_mass, chuck_R) phải > 0")
    if cfg["dt"] <= 0 or cfg["t_max"] <= 0:
        raise ValueError(f"{path}: dt và t_max phải > 0")

    valid_profiles = ["linear", "s_curve", "exponential", "hold"]
    total_time = 0.0
    steps = cfg["steps"]

    for i, step in enumerate(steps):
        prof = step.get("profile", "")
        if prof not in valid_profiles:
            raise ValueError(f"{path} (Bước {i+1}): Sai tên profile '{prof}'")

        rpm0 = step["rpm_start"]
        rpm1 = step["rpm_target"]
        dur = step["duration"]
        delta_rpm = abs(rpm1 - rpm0)

        if rpm0 < 0 or rpm1 < 0:
            raise ValueError(f"{path} (Bước {i+1}): Tốc độ rpm không được âm")
        if dur <= 0:
            raise ValueError(f"{path} (Bước {i+1}): duration phải > 0")

        # 1. Kiểm tra các bước nối liền nhau
        if i > 0:
            prev_prof = steps[i - 1]["profile"]
            prev_target = steps[i - 1]["rpm_target"]
            if prev_prof == "exponential":
                # Sau bước hàm mũ, cho phép lệch tối đa 1%
                if abs(rpm0 - prev_target) > 0.01 * max(1.0, prev_target):
                    raise ValueError(f"{path} (Bước {i+1}): Các bước không nối liền sau hàm mũ")
            else:
                if abs(rpm0 - prev_target) > 1e-6:
                    raise ValueError(f"{path} (Bước {i+1}): Các bước không nối liền ({rpm0} != {prev_target})")

        # 2. Kiểm tra tham số dương và duration khớp với gia tốc
        if prof == "hold":
            if delta_rpm > 1e-6:
                raise ValueError(f"{path} (Bước {i+1}): Bước hold phải có rpm_start == rpm_target")

        elif prof == "linear":
            acc = step.get("accel_rpm_s", 0)
            if acc <= 0:
                raise ValueError(f"{path} (Bước {i+1}): accel_rpm_s phải > 0")
            expected_dur = delta_rpm / acc
            if abs(dur - expected_dur) > 1e-6:
                raise ValueError(
                    f"{path} (Bước {i+1}): duration ({dur}s) không khớp gia tốc (cần {expected_dur:.4f}s)"
                )

        elif prof == "s_curve":
            acc = step.get("accel_rpm_s", 0)
            tj = step.get("t_jerk", 0)
            if acc <= 0 or tj <= 0:
                raise ValueError(f"{path} (Bước {i+1}): accel_rpm_s và t_jerk phải > 0")
            if (delta_rpm / acc) < tj - 1e-6:
                raise ValueError(f"{path} (Bước {i+1}): S-curve không đủ thời gian đạt alpha_max (delta_rpm/acc < t_jerk)")
            expected_dur = (delta_rpm / acc) + tj
            if abs(dur - expected_dur) > 1e-6:
                raise ValueError(
                    f"{path} (Bước {i+1}): duration ({dur}s) không khớp gia tốc S-curve (cần {expected_dur:.4f}s)"
                )

        elif prof == "exponential":
            Tm = step.get("T_m", 0)
            if Tm <= 0:
                raise ValueError(f"{path} (Bước {i+1}): T_m phải > 0")

        total_time = total_time + dur

    if cfg["t_max"] < total_time - 1e-6:
        raise ValueError(f"{path}: t_max ({cfg['t_max']}s) nhỏ hơn tổng thời gian quy trình ({total_time}s)")

    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--case", help="Đường dẫn 1 file JSON để chạy riêng")
args = parser.parse_args()

if args.case:
    files = [args.case]
else:
    files = sorted(glob.glob("data/inputs/*.json"))

os.makedirs("data/outputs", exist_ok=True)
summary_rows = []

for path in files:
    cfg = load_config(path)
    name = os.path.splitext(os.path.basename(path))[0]

    t, theta, omega, alpha = solve_motion(cfg)
    df, I_total = calc_physics(cfg, t, theta, omega, alpha)

    df.to_csv(f"data/outputs/{name}_results.csv", index=False)
    rel_err, energy_err = check_error(cfg, df)

    summary_rows.append({
        "case": name,
        "R_m": cfg["R"],
        "I_total": I_total,
        "profile": cfg["steps"][0]["profile"],
        "dt_s": cfg["dt"],
        "t_end_s": t[-1],
        "revolutions": theta[-1] / (2 * np.pi),
        "rpm_max": df["rpm"].max(),
        "alpha_max": np.abs(alpha).max(),
        "jerk_max": np.abs(df["jerk"]).max(),
        "a_t_max": df["a_t_edge"].max(),
        "a_c_max": df["a_c_edge"].max(),
        "torque_max": np.abs(df["torque"]).max(),
        "power_max": np.abs(df["power"]).max(),
        "rel_err_pct": rel_err * 100,
        "energy_err_J": energy_err
    })
    print(f"Xong: {name} | Jerk max: {np.abs(df['jerk']).max():.1f} | Sai số: {rel_err*100:.4f}%")

if not args.case:
    pd.DataFrame(summary_rows).to_csv("data/outputs/summary.csv", index=False)
    print("Đã lưu bảng tổng hợp tại: data/outputs/summary.csv")