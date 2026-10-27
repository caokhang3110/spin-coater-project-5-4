import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

from physics_model import calc_physics
from solvers import solve_motion, check_error
from profile_segments import validate_duration


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
            validate_duration(step, path, i)

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
    df.to_csv(f"data/outputs/{name}_results.csv", index=False)
    
    rel_err, energy_err = check_error(cfg, df)
    
    # Tính số vòng quay và thời gian riêng trong giai đoạn tăng tốc
    step0 = cfg["steps"][0]
    if step0["profile"] == "exponential":
        rpm_target = step0["rpm_target"]
        idx_ramp_arr = np.where(df["rpm"].to_numpy() >= 0.99 * rpm_target)[0]
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

if not args.case:
    pd.DataFrame(summary_rows).to_csv("data/outputs/summary.csv", index=False)
