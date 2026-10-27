import numpy as np
from scipy.integrate import cumulative_trapezoid


# Extract initial angular velocity in rad/s from step configuration
def get_omega_start(step_cfg):
    if "rpm_start" in step_cfg:
        return float(step_cfg["rpm_start"]) * 2.0 * np.pi / 60.0
    if "omega_initial" in step_cfg:
        return float(step_cfg["omega_initial"])
    return 0.0


# Extract target angular velocity in rad/s from step configuration
def get_omega_target(step_cfg):
    if "rpm_target" in step_cfg:
        return float(step_cfg["rpm_target"]) * 2.0 * np.pi / 60.0
    if "omega_final" in step_cfg:
        return float(step_cfg["omega_final"])
    if "rpm_final" in step_cfg:
        return float(step_cfg["rpm_final"]) * 2.0 * np.pi / 60.0
    return 0.0


# Extract constant acceleration in rad/s^2 from step configuration
def get_alpha_const(step_cfg, cfg):
    if "accel_rpm_s" in step_cfg:
        return float(step_cfg["accel_rpm_s"]) * 2.0 * np.pi / 60.0
    if "alpha_constant" in step_cfg:
        return float(step_cfg["alpha_constant"])
    if isinstance(cfg, dict) and "alpha_constant" in cfg:
        return float(cfg["alpha_constant"])
    return 100.0


# Extract total moment of inertia matching the specific wafer case
def get_inertia(cfg):
    if isinstance(cfg, dict):
        if "I_total" in cfg:
            return float(cfg["I_total"])
        if "I" in cfg:
            return float(cfg["I"])
        if "wafer" in cfg and isinstance(cfg["wafer"], dict) and "I_total" in cfg["wafer"]:
            return float(cfg["wafer"]["I_total"])
        if "R" in cfg:
            r = float(cfg["R"])
            return 0.5 * 2330.0 * np.pi * (r ** 4) * 0.001
    return 0.032688


# Build piecewise kinematic segments for analytical verification and work integration
def make_recipe_segments(cfg):
    segments = []
    t_curr = 0.0
    if isinstance(cfg, list):
        steps = cfg
    elif isinstance(cfg, dict):
        steps = cfg.get("steps", [cfg])
    else:
        steps = []

    for step in steps:
        dur = float(step.get("duration", step.get("t_max", 0.0)))
        t_end = t_curr + dur
        w0 = get_omega_start(step)
        wf = get_omega_target(step)
        alpha_c = get_alpha_const(step, cfg)
        t_j = float(step.get("t_j", step.get("t_jerk", 0.5)))
        profile = step.get("profile", step.get("system", "linear")).lower().replace("_", "")

        delta_w = abs(wf - w0)
        sign = 1.0 if wf >= w0 else -1.0

        if delta_w < 1e-12 or profile == "hold":
            segments.append({
                "type": "const",
                "t0": t_curr,
                "t1": t_end,
                "a0": 0.0,
                "j": 0.0,
                "wf": wf,
                "Tm": 1.0
            })
        elif profile == "linear":
            t_ramp = delta_w / alpha_c if alpha_c > 0 else 0.0
            t_ramp_clamped = min(t_ramp, dur)
            segments.append({
                "type": "const",
                "t0": t_curr,
                "t1": t_curr + t_ramp_clamped,
                "a0": sign * alpha_c,
                "j": 0.0,
                "wf": wf,
                "Tm": 1.0
            })
            if t_curr + t_ramp_clamped < t_end - 1e-12:
                segments.append({
                    "type": "const",
                    "t0": t_curr + t_ramp_clamped,
                    "t1": t_end,
                    "a0": 0.0,
                    "j": 0.0,
                    "wf": wf,
                    "Tm": 1.0
                })
        elif profile in ["exp", "exponential"]:
            Tm = float(step.get("T_m", step.get("T_exp", delta_w / alpha_c if alpha_c > 0 else 1.0)))
            t_ramp = float(step.get("t_ramp", dur))
            t_ramp_clamped = min(t_ramp, dur)
            segments.append({
                "type": "exp",
                "t0": t_curr,
                "t1": t_curr + t_ramp_clamped,
                "a0": 0.0,
                "j": 0.0,
                "wf": wf,
                "Tm": Tm
            })
            if t_curr + t_ramp_clamped < t_end - 1e-12:
                segments.append({
                    "type": "const",
                    "t0": t_curr + t_ramp_clamped,
                    "t1": t_end,
                    "a0": 0.0,
                    "j": 0.0,
                    "wf": wf,
                    "Tm": 1.0
                })
        elif profile in ["scurve", "s_curve"]:
            j_nom = sign * alpha_c / t_j
            if delta_w < alpha_c * t_j:
                t_prime = np.sqrt(delta_w * t_j / alpha_c)
                a_prime = sign * np.sqrt((alpha_c / t_j) * delta_w)
                t1 = min(t_curr + t_prime, t_end)
                t2 = min(t_curr + 2.0 * t_prime, t_end)
                segments.append({"type": "ramp", "t0": t_curr, "t1": t1, "a0": 0.0, "j": j_nom, "wf": wf, "Tm": 1.0})
                if t2 > t1 + 1e-12:
                    segments.append({"type": "ramp", "t0": t1, "t1": t2, "a0": a_prime, "j": -j_nom, "wf": wf, "Tm": 1.0})
                if t_end > t2 + 1e-12:
                    segments.append({"type": "const", "t0": t2, "t1": t_end, "a0": 0.0, "j": 0.0, "wf": wf, "Tm": 1.0})
            else:
                t_ramp = (delta_w / alpha_c) + t_j
                a_val = sign * alpha_c
                t1 = min(t_curr + t_j, t_end)
                t2 = min(t_curr + t_ramp - t_j, t_end)
                t3 = min(t_curr + t_ramp, t_end)
                segments.append({"type": "ramp", "t0": t_curr, "t1": t1, "a0": 0.0, "j": j_nom, "wf": wf, "Tm": 1.0})
                if t2 > t1 + 1e-12:
                    segments.append({"type": "const", "t0": t1, "t1": t2, "a0": a_val, "j": 0.0, "wf": wf, "Tm": 1.0})
                if t3 > t2 + 1e-12:
                    segments.append({"type": "ramp", "t0": t2, "t1": t3, "a0": a_val, "j": -j_nom, "wf": wf, "Tm": 1.0})
                if t_end > t3 + 1e-12:
                    segments.append({"type": "const", "t0": t3, "t1": t_end, "a0": 0.0, "j": 0.0, "wf": wf, "Tm": 1.0})

        t_curr += dur

    return segments


# Evaluate exact analytical angular position, velocity, and acceleration across segments
def evaluate_exact_trajectories(t, segments, w_init, th_init=0.0):
    theta_ana = np.zeros_like(t, dtype=float)
    omega_ana = np.zeros_like(t, dtype=float)
    th0, w0 = float(th_init), float(w_init)

    for seg in segments:
        t0, t1 = seg["t0"], seg["t1"]
        stype = seg["type"]
        mask = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)
        dt_arr = t[mask] - t0
        dt_seg = t1 - t0

        if stype == "const":
            a0 = seg["a0"]
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

    return theta_ana, omega_ana


# Calculate relative kinematic error and piecewise work-energy error matching main signature
def check_error(cfg, df):
    t_col = "time_t" if "time_t" in df else ("t" if "t" in df else df.columns[0])
    t = np.asarray(df[t_col], dtype=float)
    theta_num = np.asarray(df["theta"], dtype=float)
    omega_num = np.asarray(df["omega"], dtype=float)

    I_val = get_inertia(cfg)
    if "kinetic_energy" in df:
        KE = np.asarray(df["kinetic_energy"], dtype=float)
    elif "KE" in df:
        KE = np.asarray(df["KE"], dtype=float)
    else:
        KE = 0.5 * I_val * (omega_num ** 2)

    segments = make_recipe_segments(cfg)
    steps = cfg.get("steps", []) if isinstance(cfg, dict) else (cfg if isinstance(cfg, list) else [])
    w0_init = get_omega_start(steps[0]) if len(steps) > 0 else omega_num[0]
    th0_init = float(theta_num[0]) if len(theta_num) > 0 else 0.0

    theta_ana, omega_ana = evaluate_exact_trajectories(t, segments, w0_init, th0_init)

    max_th = max(np.max(np.abs(theta_ana)), 1e-12)
    max_w = max(np.max(np.abs(omega_ana)), 1e-12)

    err_th_pct = (np.max(np.abs(theta_num - theta_ana)) / max_th) * 100.0
    err_w_pct = (np.max(np.abs(omega_num - omega_ana)) / max_w) * 100.0
    rel_err_pct = float(max(err_th_pct, err_w_pct))

    work_total = np.zeros_like(t, dtype=float)
    work_accumulated = 0.0

    for seg in segments:
        t0, t1 = seg["t0"], seg["t1"]
        stype = seg["type"]
        mask = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)

        if np.sum(mask) > 1:
            t_seg = t[mask]
            w_seg = omega_num[mask]
            dt_seg = t_seg - t0

            if stype == "const":
                alpha_seg = np.full_like(t_seg, seg["a0"])
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

    delta_ke = KE - KE[0]
    energy_err_joules = float(np.max(np.abs(work_total - delta_ke)))

    return rel_err_pct, energy_err_joules
