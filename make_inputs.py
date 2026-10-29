import json
import os
import glob
os.makedirs("data/inputs", exist_ok=True)
os.makedirs("tests", exist_ok=True)
# Xóa các file JSON cũ trong data/inputs để tránh lẫn file cũ
for old_file in glob.glob("data/inputs/*.json"):
    os.remove(old_file)

WAFERS = {
    100: {"R": 0.050, "thickness": 0.000525},
    150: {"R": 0.075, "thickness": 0.000675},
    200: {"R": 0.100, "thickness": 0.000725},
    300: {"R": 0.150, "thickness": 0.000775}
}


def save_json(folder, name, size_mm, dt, ramp_step, add_hold=True):
    w = WAFERS[size_mm]
    steps = [ramp_step]
    t_max = ramp_step["duration"]

    if add_hold:
        target = ramp_step["rpm_target"]
        steps.append({
            "profile": "hold",
            "rpm_start": target,
            "rpm_target": target,
            "duration": 2.0
        })
        t_max = round(t_max + 2.0, 6)

    data = {
        "R": w["R"],
        "thickness": w["thickness"],
        "density": 2330.0,
        "chuck_mass": 2.778,
        "chuck_R": 0.15,
        "dt": dt,
        "t_max": t_max,
        "steps": steps
    }
    with open(f"{folder}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =====================================================================
# 1. NHÓM A: So sánh 3 profile & Thay đổi t_jerk, T_m (300mm, 3000 rpm)
# =====================================================================
save_json("data/inputs", "case_300mm_linear_1000", 300, 0.005, {
    "profile": "linear", "rpm_start": 0.0, "rpm_target": 3000.0, "accel_rpm_s": 1000.0, "duration": 3.0
})

for tj, tag in [(0.2, "tj02"), (0.5, "tj05"), (1.0, "tj10")]:
    dur = 3.0 + tj
    name = "case_300mm_scurve_1000" if tj == 0.5 else f"case_300mm_scurve_1000_{tag}"
    save_json("data/inputs", name, 300, 0.005, {
        "profile": "s_curve", "rpm_start": 0.0, "rpm_target": 3000.0,
        "accel_rpm_s": 1000.0, "t_jerk": tj, "duration": dur
    })

for Tm, tag in [(0.3, "Tm03"), (0.65, "Tm065"), (1.0, "Tm10")]:
    dur = round(6.0 * Tm, 2)  # Đạt > 99.7% tốc độ mục tiêu để nối mượt sang bước hold
    name = "case_300mm_exp_1000" if Tm == 0.65 else f"case_300mm_exp_{tag}"
    save_json("data/inputs", name, 300, 0.005, {
        "profile": "exponential", "rpm_start": 0.0, "rpm_target": 3000.0, "T_m": Tm, "duration": dur
    })

# S-curve Case B (triangular): delta_rpm = 300 < accel * t_jerk = 1000 * 0.5 = 500,
# so a_max is never reached; duration = 2 * sqrt(300 * 0.5 / 1000) = 0.7746 s
save_json("data/inputs", "case_300mm_scurve_300_caseB", 300, 0.001, {
    "profile": "s_curve", "rpm_start": 0.0, "rpm_target": 300.0,
    "accel_rpm_s": 1000.0, "t_jerk": 0.5, "duration": 0.7746
})

# =====================================================================
# 2. NHÓM B: Thay đổi gia tốc (500, 2000, 5000, 10000 rpm/s)
# =====================================================================
for acc in [500, 2000, 5000, 10000]:
    t_ramp = 3000.0 / acc
    save_json("data/inputs", f"case_300mm_linear_{acc}", 300, 0.002, {
        "profile": "linear", "rpm_start": 0.0, "rpm_target": 3000.0, "accel_rpm_s": float(acc), "duration": t_ramp
    })

# =====================================================================
# 3. NHÓM C: Thay đổi cỡ Wafer (100, 150, 200 mm)
# =====================================================================
for size in [100, 150, 200]:
    save_json("data/inputs", f"case_{size}mm_linear_1000", size, 0.005, {
        "profile": "linear", "rpm_start": 0.0, "rpm_target": 3000.0, "accel_rpm_s": 1000.0, "duration": 3.0
    })

# =====================================================================
# 4. NHÓM D: Quy trình nhiều bước (Multi-step Recipe)
# =====================================================================
multistep_data = {
    "R": 0.15, "thickness": 0.000775, "density": 2330.0,
    "chuck_mass": 2.778, "chuck_R": 0.15,
    "dt": 0.02, "t_max": 41.5,
    "steps": [
        {"profile": "linear", "rpm_start": 0.0, "rpm_target": 500.0, "accel_rpm_s": 500.0, "duration": 1.0},
        {"profile": "hold", "rpm_start": 500.0, "rpm_target": 500.0, "duration": 5.0},
        {"profile": "linear", "rpm_start": 500.0, "rpm_target": 3000.0, "accel_rpm_s": 1000.0, "duration": 2.5},
        {"profile": "hold", "rpm_start": 3000.0, "rpm_target": 3000.0, "duration": 30.0},
        {"profile": "linear", "rpm_start": 3000.0, "rpm_target": 0.0, "accel_rpm_s": 1000.0, "duration": 3.0}
    ]
}
with open("data/inputs/case_300mm_multistep.json", "w", encoding="utf-8") as f:
    json.dump(multistep_data, f, indent=2)

# =====================================================================
# 5. NHÓM E: Các trường hợp cực trị (Extreme Cases)
# =====================================================================
# E1: Gia tốc rất lớn (20000 rpm/s lên 6000 rpm)
save_json("data/inputs", "case_extreme_high_accel_20000", 300, 0.001, {
    "profile": "linear", "rpm_start": 0.0, "rpm_target": 6000.0, "accel_rpm_s": 20000.0, "duration": 0.3
})

# E2: Giảm tốc khẩn cấp (3000 -> 0 rpm trong 0.3s, tốc độ giảm 10000 rpm/s)
save_json("data/inputs", "case_emergency_stop", 300, 0.001, {
    "profile": "linear", "rpm_start": 3000.0, "rpm_target": 0.0, "accel_rpm_s": 10000.0, "duration": 0.3
})

# E3: Gia tốc rất nhỏ (100 rpm/s lên 3000 rpm mất 30s)
save_json("data/inputs", "case_extreme_low_accel_100", 300, 0.02, {
    "profile": "linear", "rpm_start": 0.0, "rpm_target": 3000.0, "accel_rpm_s": 100.0, "duration": 30.0
})

# =====================================================================
# 6. NHÓM F: Khảo sát bước thời gian dt (Linear và S-curve)
# =====================================================================
for dt_val, tag in [(0.05, "dt005"), (0.01, "dt001"), (0.005, "dt0005"), (0.001, "dt0001")]:
    save_json("data/inputs", f"case_300mm_linear_1000_{tag}", 300, dt_val, {
        "profile": "linear", "rpm_start": 0.0, "rpm_target": 3000.0, "accel_rpm_s": 1000.0, "duration": 3.0
    })
    save_json("data/inputs", f"case_300mm_scurve_1000_{tag}", 300, dt_val, {
        "profile": "s_curve", "rpm_start": 0.0, "rpm_target": 3000.0,
        "accel_rpm_s": 1000.0, "t_jerk": 0.5, "duration": 3.5
    })

# =====================================================================
# 7. TẠO 4 FILE TEST CỐ Ý SAI TRONG THƯ MỤC tests
# =====================================================================
# Test 1: accel 1500, duration vẫn 3s
save_json("tests", "bad_duration", 300, 0.005, {
    "profile": "linear", "rpm_start": 0.0, "rpm_target": 3000.0, "accel_rpm_s": 1500.0, "duration": 3.0
}, add_hold=False)

# Test 2: sai tên profile "scurve"
save_json("tests", "bad_profile_name", 300, 0.005, {
    "profile": "scurve", "rpm_start": 0.0, "rpm_target": 3000.0, "accel_rpm_s": 1000.0, "duration": 3.0
}, add_hold=False)

# Test 3: T_m = 0
save_json("tests", "bad_Tm_zero", 300, 0.005, {
    "profile": "exponential", "rpm_start": 0.0, "rpm_target": 3000.0, "T_m": 0.0, "duration": 3.0
}, add_hold=False)

# Test 4: Bước 1 kết thúc 500 rpm, bước 2 bắt đầu 400 rpm
bad_connect = {
    "R": 0.15, "thickness": 0.000775, "density": 2330.0, "chuck_mass": 2.778, "chuck_R": 0.15,
    "dt": 0.005, "t_max": 3.0,
    "steps": [
        {"profile": "linear", "rpm_start": 0.0, "rpm_target": 500.0, "accel_rpm_s": 500.0, "duration": 1.0},
        {"profile": "linear", "rpm_start": 400.0, "rpm_target": 1000.0, "accel_rpm_s": 300.0, "duration": 2.0}
    ]
}
with open("tests/bad_step_connect.json", "w", encoding="utf-8") as f:
    json.dump(bad_connect, f, indent=2)

print("Đã tạo xong toàn bộ file JSON trong data/inputs/ và 4 file test lỗi trong tests/!")