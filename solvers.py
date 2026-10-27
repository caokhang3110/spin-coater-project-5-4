import numpy as np
from scipy.integrate import solve_ivp
from physics_model import rpm_to_rads
from profile_segments import make_segments, validate_duration
from error_checker import check_error


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

        start_idx = 0 if i == 0 else 1
        T.append(s.t[start_idx:])
        Theta.append(s.y[0][start_idx:])
        Omega.append(s.y[1][start_idx:])
        Alpha.append(a_seg[start_idx:])

        t0 = t1
        y0 = [s.y[0][-1], s.y[1][-1]]

    return np.concatenate(T), np.concatenate(Theta), np.concatenate(Omega), np.concatenate(Alpha)
