import numpy as np
from physics_model import rpm_to_rads


def validate_duration(step, path="", step_idx=0):
    prof = step.get("profile", "")
    rpm0 = step["rpm_start"]
    rpm1 = step["rpm_target"]
    dur = step["duration"]
    delta_rpm = abs(rpm1 - rpm0)
    prefix = f"{path} (Step {step_idx + 1})" if path else f"Step {step_idx + 1}"

    if prof == "hold":
        if delta_rpm > 1e-6:
            raise ValueError(f"{prefix}: Hold step must have rpm_start == rpm_target")

    elif prof == "linear":
        acc = step.get("accel_rpm_s", 0)
        if acc <= 0:
            raise ValueError(f"{prefix}: accel_rpm_s must be > 0")
        expected_dur = delta_rpm / acc
        if abs(dur - expected_dur) > 1e-6:
            raise ValueError(f"{prefix}: duration does not match linear acceleration (expected {expected_dur:.4f}s)")

    elif prof == "s_curve":
        acc = step.get("accel_rpm_s", 0)
        tj = step.get("t_jerk", 0)
        if acc <= 0 or tj <= 0:
            raise ValueError(f"{prefix}: accel_rpm_s and t_jerk must be > 0")
        if delta_rpm < acc * tj - 1e-6:
            t_prime = np.sqrt(delta_rpm * tj / acc)
            expected_dur = 2.0 * t_prime
        else:
            expected_dur = (delta_rpm / acc) + tj
        if abs(dur - expected_dur) > 1e-4:
            raise ValueError(f"{prefix}: duration does not match s_curve acceleration (expected {expected_dur:.4f}s)")

    elif prof == "exponential":
        Tm = step.get("T_m", 0)
        if Tm <= 0:
            raise ValueError(f"{prefix}: T_m must be > 0")


def make_segments(steps):
    if isinstance(steps, dict):
        steps = steps.get("steps", [steps])

    segments = []
    for s in steps:
        prof = s["profile"]
        w0 = rpm_to_rads(s["rpm_start"])
        wf = rpm_to_rads(s["rpm_target"])
        dur = float(s["duration"])
        sign = 1.0 if wf >= w0 else -1.0
        delta_w = abs(wf - w0)

        if prof == "hold" or delta_w < 1e-12:
            segments.append({"dur": dur, "type": "const", "a": 0.0})

        elif prof == "linear":
            a = sign * rpm_to_rads(s["accel_rpm_s"])
            segments.append({"dur": dur, "type": "const", "a": a})

        elif prof == "s_curve":
            a_max = rpm_to_rads(s["accel_rpm_s"])
            tj = float(s["t_jerk"])
            j = a_max / tj

            if delta_w < a_max * tj - 1e-12:
                t_prime = np.sqrt(delta_w * tj / a_max)
                a_prime = j * t_prime
                segments.append({"dur": t_prime, "type": "ramp", "a0": 0.0, "j": sign * j})
                segments.append({"dur": t_prime, "type": "ramp", "a0": sign * a_prime, "j": -sign * j})
            else:
                t_hold = dur - 2.0 * tj
                segments.append({"dur": tj, "type": "ramp", "a0": 0.0, "j": sign * j})
                if t_hold > 1e-12:
                    segments.append({"dur": t_hold, "type": "const", "a": sign * a_max})
                segments.append({"dur": tj, "type": "ramp", "a0": sign * a_max, "j": -sign * j})

        elif prof == "exponential":
            segments.append({"dur": dur, "type": "exp", "wf": wf, "Tm": float(s["T_m"])})

    return segments
