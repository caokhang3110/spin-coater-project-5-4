import argparse
import csv
import glob
import json
import os
import numpy as np

from physics_model import calc_physics
from solvers import solve_motion, check_error

def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg["R"] <= 0 or cfg["thickness"] <= 0 or cfg["density"] <= 0:
        raise ValueError(f"{path}: Wafer parameters (R, thickness, density) must be > 0")
    if cfg["chuck_mass"] <= 0 or cfg["chuck_R"] <= 0:
        raise ValueError(f"{path}: Chuck parameters (chuck_mass, chuck_R) must be > 0")
    if cfg["dt"] <= 0 or cfg["t_max"] <= 0:
        raise ValueError(f"{path}: dt and t_max must be > 0")

    valid_profiles = ["linear", "s_curve", "exponential", "hold"]
    total_time = 0.0
    steps = cfg["steps"]

    for i, step in enumerate(steps):
        prof = step.get("profile", "")
        if prof not in valid_profiles:
            raise ValueError(f"{path} (Step {i+1}): Invalid profile name '{prof}'")

        rpm0 = step["rpm_start"]
        rpm1 = step["rpm_target"]
        dur = step["duration"]
        delta_rpm = abs(rpm1 - rpm0)

        if rpm0 < 0 or rpm1 < 0:
            raise ValueError(f"{path} (Step {i+1}): RPM cannot be negative")
        if dur <= 0:
            raise ValueError(f"{path} (Step {i+1}): duration must be > 0")

        if i > 0:
            prev_prof = steps[i - 1]["profile"]
            prev_target = steps[i - 1]["rpm_target"]
            if prev_prof == "exponential":
                if abs(rpm0 - prev_target) > 0.01 * max(1.0, prev_target):
                    raise ValueError(f"{path} (Step {i+1}): Steps not continuous after exponential profile")
            else:
                if abs(rpm0 - prev_target) > 1e-6:
                    raise ValueError(f"{path} (Step {i+1}): Steps not continuous ({rpm0} != {prev_target})")

        if prof == "hold":
            if delta_rpm > 1e-6:
                raise ValueError(f"{path} (Step {i+1}): Hold step must have rpm_start == rpm_target")

        elif prof == "linear":
            acc = step.get("accel_rpm_s", 0)
            if acc <= 0:
                raise ValueError(f"{path} (Step {i+1}): accel_rpm_s must be > 0")
            expected_dur = delta_rpm / acc
            if abs(dur - expected_dur) > 1e-6:
                raise ValueError(f"{path} (Step {i+1}): duration does not match linear acceleration (expected {expected_dur:.4f}s)")

        elif prof == "s_curve":
            acc = step.get("accel_rpm_s", 0)
            tj = step.get("t_jerk", 0)
            if acc <= 0 or tj <= 0:
                raise ValueError(f"{path} (Step {i+1}): accel_rpm_s and t_jerk must be > 0")
            if (delta_rpm / acc) < tj - 1e-6:
                raise ValueError(f"{path} (Step {i+1}): S-curve insufficient time to reach alpha_max")
            expected_dur = (delta_rpm / acc) + tj
            if abs(dur - expected_dur) > 1e-6:
                raise ValueError(f"{path} (Step {i+1}): duration does not match s_curve acceleration (expected {expected_dur:.4f}s)")

        elif prof == "exponential":
            Tm = step.get("T_m", 0)
            if Tm <= 0:
                raise ValueError(f"{path} (Step {i+1}): T_m must be > 0")

        total_time += dur

    if cfg["t_max"] < total_time - 1e-6:
        raise ValueError(f"{path}: t_max ({cfg['t_max']}s) is less than total recipe time ({total_time}s)")

    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--case", help="Path to a single JSON file to run")
args = parser.parse_args()

files = [args.case] if args.case else sorted(glob.glob("data/inputs/*.json"))
os.makedirs("data/outputs", exist_ok=True)
summary_rows = []

for path in files:
    cfg = load_config(path)
    name = os.path.splitext(os.path.basename(path))[0]

    t, theta, omega, alpha = solve_motion(cfg)
    df, I_total = calc_physics(cfg, t, theta, omega, alpha)
    
    # Sử dụng csv thay vì pandas để xuất kết quả từng kịch bản
    keys = list(df.keys())
    length = len(df[keys[0]])
    with open(f"data/outputs/{name}_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(length):
            writer.writerow([df[k][i] for k in keys])
    
    rel_err, energy_err = check_error(cfg, df)
    
    step0 = cfg["steps"][0]
    if step0["profile"] == "exponential":
        rpm_target = step0["rpm_target"]
        rpm_arr = np.array(df["rpm"])
        idx_ramp_arr = np.where(rpm_arr >= 0.99 * rpm_target)[0]
        if len(idx_ramp_arr) > 0:
            idx_ramp = idx_ramp_arr[0]
            t_ramp = t[idx_ramp]
        else:
            t_ramp = t[-1]
            idx_ramp = -1
    else:
        t_ramp = step0["duration"]
        idx_ramp = np.argmin(np.abs(t - t_ramp))
        
    revolutions_ramp = theta[idx_ramp] / (2 * np.pi)

    summary_rows.append({
        "case": name,
        "R_m": cfg["R"],
        "I_total": I_total,
        "profile": cfg["steps"][0]["profile"],
        "dt_s": cfg["dt"],
        "t_end_s": t[-1],
        "t_ramp_s": t_ramp,
        "revolutions_ramp": revolutions_ramp,
        "revolutions_total": theta[-1] / (2 * np.pi),
        "rpm_max": np.max(df["rpm"]),
        "alpha_max": np.max(np.abs(alpha)),
        "jerk_max": np.max(np.abs(df["jerk"])),
        "a_t_max": np.max(df["a_t_edge"]),
        "a_c_max": np.max(df["a_c_edge"]),
        "torque_max": np.max(np.abs(df["torque"])),
        "power_max": np.max(np.abs(df["power"])),
        "rel_err_pct": rel_err * 100,
        "energy_err_J": energy_err
    })

if args.case:
    row = summary_rows[-1]
    print(f"\n--- Run results for: {args.case} ---")
    for key in ["t_ramp_s", "revolutions_ramp", "jerk_max", "a_t_max", "a_c_max", "torque_max", "rel_err_pct", "energy_err_J"]:
        print(f"{key:>18}: {row[key]:.4g}")
else:
    # Xuất file summary bằng csv
    if summary_rows:
        with open("data/outputs/summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
