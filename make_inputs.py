import json
import os

os.makedirs("data/inputs", exist_ok=True)


def save_json(name, R, thickness, dt, t_max, steps):
    data = {
        "R": R,
        "thickness": thickness,
        "density": 2330.0,
        "chuck_mass": 0.17,
        "chuck_R": 0.045,
        "dt": dt,
        "t_max": t_max,
        "steps": steps
    }
    with open(f"data/inputs/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# 1. Ba profile cơ bản (Wafer 300mm -> R = 0.15m)
save_json("case_300mm_linear_1000", 0.15, 0.000775, 0.005, 3.0, [
    {"profile": "linear", "rpm_start": 0, "rpm_target": 3000, "accel_rpm_s": 1000, "duration": 3.0}
])

save_json("case_300mm_scurve_1000", 0.15, 0.000775, 0.005, 3.5, [
    {"profile": "s_curve", "rpm_start": 0, "rpm_target": 3000, "accel_rpm_s": 1000, "t_jerk": 0.5, "duration": 3.5}
])

save_json("case_300mm_exp_1000", 0.15, 0.000775, 0.005, 4.0, [
    {"profile": "exponential", "rpm_start": 0, "rpm_target": 3000, "T_m": 0.65, "duration": 4.0}
])

# 2. Thay đổi tốc độ tăng tốc (500, 2000, 5000 rpm/s)
for acc in [500, 2000, 5000]:
    t_ramp = 3000.0 / acc
    save_json(f"case_300mm_linear_{acc}", 0.15, 0.000775, 0.002, t_ramp, [
        {"profile": "linear", "rpm_start": 0, "rpm_target": 3000, "accel_rpm_s": acc, "duration": t_ramp}
    ])

# 3. Thay đổi cỡ Wafer (100mm, 150mm, 200mm)
save_json("case_100mm_linear_1000", 0.050, 0.000525, 0.005, 3.0, [
    {"profile": "linear", "rpm_start": 0, "rpm_target": 3000, "accel_rpm_s": 1000, "duration": 3.0}
])
save_json("case_150mm_linear_1000", 0.075, 0.000675, 0.005, 3.0, [
    {"profile": "linear", "rpm_start": 0, "rpm_target": 3000, "accel_rpm_s": 1000, "duration": 3.0}
])
save_json("case_200mm_linear_1000", 0.100, 0.000725, 0.005, 3.0, [
    {"profile": "linear", "rpm_start": 0, "rpm_target": 3000, "accel_rpm_s": 1000, "duration": 3.0}
])

# 4. Quy trình nhiều bước (500 rpm giữ 5s -> 3000 rpm giữ 30s -> giảm về 0)
save_json("case_300mm_multistep", 0.15, 0.000775, 0.02, 41.5, [
    {"profile": "linear", "rpm_start": 0, "rpm_target": 500, "accel_rpm_s": 500, "duration": 1.0},
    {"profile": "hold", "rpm_start": 500, "rpm_target": 500, "accel_rpm_s": 0, "duration": 5.0},
    {"profile": "linear", "rpm_start": 500, "rpm_target": 3000, "accel_rpm_s": 1000, "duration": 2.5},
    {"profile": "hold", "rpm_start": 3000, "rpm_target": 3000, "accel_rpm_s": 0, "duration": 30.0},
    {"profile": "linear", "rpm_start": 3000, "rpm_target": 0, "accel_rpm_s": 1000, "duration": 3.0}
])

# 5. Trường hợp cực trị (tăng tốc cực lớn 20000 rpm/s)
save_json("case_extreme_20000", 0.15, 0.000775, 0.001, 0.3, [
    {"profile": "linear", "rpm_start": 0, "rpm_target": 6000, "accel_rpm_s": 20000, "duration": 0.3}
])

print("Đã tạo xong các file JSON!")