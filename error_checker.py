import numpy as np
from scipy.integrate import cumulative_trapezoid
from profile_segments import make_segments
from physics_model import rpm_to_rads


def evaluate_exact_trajectories(t, segments, w_init, th_init=0.0):
    theta_ana = np.zeros_like(t, dtype=float)
    omega_ana = np.zeros_like(t, dtype=float)
    th0, w0 = float(th_init), float(w_init)
    t0 = 0.0

    for seg in segments:
        t1 = t0 + seg["dur"]
        stype = seg["type"]
        mask = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)
        dt_arr = t[mask] - t0
        dt_seg = seg["dur"]

        if stype == "const":
            a0 = seg["a"]
            omega_ana[mask] = w0 + a0 * dt_arr
            theta_ana[mask] = th0 + w0 * dt_arr + 0.5 * a0 * (dt_arr ** 2)
            th0 = th0 + w0 * dt_seg + 0.5 * a0 * (dt_seg ** 2)
            w0 = w0 + a0 * dt_seg
        elif stype == "ramp":
            a0 = seg["a0"]
            j = seg["j"]
            omega_ana[mask] = w0 + a0 * dt_arr + 0.5 * j * (dt_arr ** 2)
            theta_ana[mask] = th0 + w0 * dt_arr + 0.5 * a0 * (dt_arr ** 2) + (1.0 / 6.0) * j * (dt_arr ** 3)
            th0 = th0 + w0 * dt_seg + 0.5 * a0 * (dt_seg ** 2) + (1.0 / 6.0) * j * (dt_seg ** 3)
            w0 = w0 + a0 * dt_seg + 0.5 * j * (dt_seg ** 2)
        elif stype == "exp":
            wf = seg["wf"]
            Tm = seg["Tm"]
            exp_arr = np.exp(-dt_arr / Tm)
            omega_ana[mask] = wf + (w0 - wf) * exp_arr
            theta_ana[mask] = th0 + wf * dt_arr + (w0 - wf) * Tm * (1.0 - exp_arr)
            exp_end = np.exp(-dt_seg / Tm)
            th0 = th0 + wf * dt_seg + (w0 - wf) * Tm * (1.0 - exp_end)
            w0 = wf + (w0 - wf) * exp_end

        t0 = t1

    return theta_ana, omega_ana


def check_error(cfg, df):
    t_col = "time_t" if "time_t" in df else ("t" if "t" in df else df.columns[0])
    t = np.asarray(df[t_col], dtype=float)
    theta_num = np.asarray(df["theta"], dtype=float)
    omega_num = np.asarray(df["omega"], dtype=float)

    if "kinetic_energy" in df:
        KE = np.asarray(df["kinetic_energy"], dtype=float)
    elif "KE" in df:
        KE = np.asarray(df["KE"], dtype=float)
    else:
        KE = np.zeros_like(omega_num)

    i = np.argmax(np.abs(omega_num))
    I_val = 2.0 * KE[i] / (omega_num[i] ** 2)

    steps = cfg.get("steps", []) if isinstance(cfg, dict) else cfg
    segments = make_segments(steps)
    w0_init = rpm_to_rads(steps[0]["rpm_start"]) if len(steps) > 0 else omega_num[0]
    th0_init = float(theta_num[0]) if len(theta_num) > 0 else 0.0

    theta_ana, omega_ana = evaluate_exact_trajectories(t, segments, w0_init, th0_init)

    max_th = max(np.max(np.abs(theta_ana)), 1e-12)
    max_w = max(np.max(np.abs(omega_ana)), 1e-12)

    err_th = np.max(np.abs(theta_num - theta_ana)) / max_th
    err_w = np.max(np.abs(omega_num - omega_ana)) / max_w
    rel_err = float(max(err_th, err_w)) * 100.0

    work_total = np.zeros_like(t, dtype=float)
    work_accumulated = 0.0
    t0 = 0.0

    for seg in segments:
        t1 = t0 + seg["dur"]
        stype = seg["type"]
        mask = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)

        if np.sum(mask) > 1:
            t_seg = t[mask]
            w_seg = omega_num[mask]
            dt_seg = t_seg - t0

            if stype == "const":
                alpha_seg = np.full_like(t_seg, seg["a"])
            elif stype == "ramp":
                alpha_seg = seg["a0"] + seg["j"] * dt_seg
            elif stype == "exp":
                alpha_seg = (seg["wf"] - w_seg) / seg["Tm"]
            else:
                alpha_seg = np.zeros_like(t_seg)

            p_seg = (I_val * alpha_seg) * w_seg
            w_seg_integral = cumulative_trapezoid(p_seg, t_seg, initial=0.0) + work_accumulated
            work_total[mask] = w_seg_integral
            work_accumulated = float(w_seg_integral[-1])

        t0 = t1

    delta_ke = KE - KE[0]
    energy_err_joules = float(np.max(np.abs(work_total - delta_ke)))

    return rel_err, energy_err_joules
