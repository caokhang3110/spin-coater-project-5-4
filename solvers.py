import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from physics_model import rpm_to_rads
def make_segments(steps):
    segments = []
    for s in steps:
        prof = s["profile"]
        w0 = rpm_to_rads(s["rpm_start"])
        wf = rpm_to_rads(s["rpm_target"])
        dur = s["duration"]
        sign = 1.0 if wf >= w0 else -1.0

        if prof == "hold":
            segments.append({"dur": dur, "type": "const", "a": 0.0})

        elif prof == "linear":
            a = sign * rpm_to_rads(s["accel_rpm_s"])
            segments.append({"dur": dur, "type": "const", "a": a})

        elif prof == "s_curve":
            a_max = rpm_to_rads(s["accel_rpm_s"])
            tj = s["t_jerk"]
            j = a_max / tj
            t_hold = dur - 2 * tj

            segments.append({"dur": tj, "type": "ramp", "a0": 0.0, "j": sign * j})
            if t_hold > 0:
                segments.append({"dur": t_hold, "type": "const", "a": sign * a_max})
            segments.append({"dur": tj, "type": "ramp", "a0": sign * a_max, "j": -sign * j})

        elif prof == "exponential":
            segments.append({"dur": dur, "type": "exp", "wf": wf, "Tm": s["T_m"]})

    return segments


def solve_motion(cfg):
    dt = cfg["dt"]
    segments = make_segments(cfg["steps"])

    t0 = 0.0
    w_start = rpm_to_rads(cfg["steps"][0]["rpm_start"])
    y0 = [0.0, w_start]

    T, Theta, Omega, Alpha = [], [], [], []

    for i, seg in enumerate(segments):
        t1 = t0 + seg["dur"]
        n = max(2, round(seg["dur"] / dt))
        te = np.linspace(t0, t1, n + 1)

        if seg["type"] == "const":
            def rhs(t, y):
                return [y[1], seg["a"]]
            s = solve_ivp(rhs, (t0, t1), y0, method="RK45", t_eval=te, rtol=1e-9, atol=1e-12)
            a_seg = np.full_like(s.t, seg["a"])

        elif seg["type"] == "ramp":
            def rhs(t, y):
                a_now = seg["a0"] + seg["j"] * (t - t0)
                return [y[1], a_now]
            s = solve_ivp(rhs, (t0, t1), y0, method="RK45", t_eval=te, rtol=1e-9, atol=1e-12)
            a_seg = seg["a0"] + seg["j"] * (s.t - t0)

        elif seg["type"] == "exp":
            def rhs(t, y):
                a_now = (seg["wf"] - y[1]) / seg["Tm"]
                return [y[1], a_now]
            s = solve_ivp(rhs, (t0, t1), y0, method="RK45", t_eval=te, rtol=1e-9, atol=1e-12)
            a_seg = (seg["wf"] - s.y[1]) / seg["Tm"]

        if not s.success:
            raise RuntimeError(s.message)

        # Bỏ điểm đầu của đoạn sau để không bị trùng thời điểm nối
        start_idx = 0 if i == 0 else 1
        T.append(s.t[start_idx:])
        Theta.append(s.y[0][start_idx:])
        Omega.append(s.y[1][start_idx:])
        Alpha.append(a_seg[start_idx:])

        t0 = t1
        y0 = [s.y[0][-1], s.y[1][-1]]

    return np.concatenate(T), np.concatenate(Theta), np.concatenate(Omega), np.concatenate(Alpha)


def check_error(cfg, df):
    t = df["time_t"].to_numpy()
    omega = df["omega"].to_numpy()
    torque = df["torque"].to_numpy()
    KE = df["kinetic_energy"].to_numpy()

    # Kiểm tra định lý công - động năng: W = tích phân (tau * omega dt) so với dK
    W = cumulative_trapezoid(torque * omega, t, initial=0)
    dK = KE - KE[0]
    energy_err = np.max(np.abs(W - dK))

    # So với nghiệm giải tích omega = a * t (cho profile tuyến tính 1 bước)
    rel_err = 0.0
    step0 = cfg["steps"][0]
    if len(cfg["steps"]) == 1 and step0["profile"] == "linear":
        w0 = rpm_to_rads(step0["rpm_start"])
        wf = rpm_to_rads(step0["rpm_target"])
        sign = 1.0 if wf >= w0 else -1.0
        a = sign * rpm_to_rads(step0["accel_rpm_s"])

        w_exact = w0 + a * t
        mask = np.abs(w_exact) > 1e-12
        rel_err = np.max(np.abs(omega[mask] - w_exact[mask]) / np.abs(w_exact[mask]))

    return float(rel_err), float(energy_err)