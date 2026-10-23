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
        raise ValueError(f"{path}: Tham số wafer phải > 0")
    if cfg["dt"] <= 0 or cfg["t_max"] <= 0:
        raise ValueError(f"{path}: dt và t_max phải > 0")
    total_time = 0
    for step in cfg["steps"]:
        if step["rpm_start"] < 0 or step["rpm_target"] < 0:
            raise ValueError(f"{path}: rpm không được âm")
        if step["duration"] <= 0:
            raise ValueError(f"{path}: duration phải > 0")
        total_time = total_time + step["duration"]
    if cfg["t_max"] < total_time - 1e-6:
        raise ValueError(f"{path}: t_max nhỏ hơn tổng thời gian quy trình")
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

    # 1. Giải phương trình vi phân tìm t, theta, omega, alpha
    t, theta, omega, alpha = solve_motion(cfg)

    # 2. Tính các đại lượng vật lý
    df, I_total = calc_physics(cfg, t, theta, omega, alpha)

    # 3. Lưu bảng kết quả chi tiết ra CSV
    df.to_csv(f"data/outputs/{name}_results.csv", index=False)

    # 4. Kiểm tra sai số
    rel_err, energy_err = check_error(cfg, df)

    # 5. Lưu thông tin tóm tắt vào bảng tổng hợp
    summary_rows.append({
        "case": name,
        "R_m": cfg["R"],
        "I_total": I_total,
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
    print(f"Xong: {name}")

if not args.case:
    pd.DataFrame(summary_rows).to_csv("data/outputs/summary.csv", index=False)
    print("Đã lưu: data/outputs/summary.csv")