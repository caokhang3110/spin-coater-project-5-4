import numpy as np
from scipy.integrate import cumulative_trapezoid
from profile_segments import make_segments, get_omega_rad


# Calculate moment of inertia of the disk from configuration parameters
def get_moment_of_inertia(cfg):
    if 'I' in cfg and cfg['I'] is not None:
        return float(cfg['I'])
    R = float(cfg.get('R', 0.15))
    d = float(cfg.get('d', cfg.get('h', 0.001)))
    rho = float(cfg.get('rho', 2330.0))
    mass = rho * np.pi * (R ** 2) * d
    return 0.5 * mass * (R ** 2)


# Evaluate analytical angular position, velocity, and acceleration across segments
def evaluate_piecewise_analytical(t, segments, omega_initial, theta_initial=0.0):
    theta_ana = np.zeros_like(t, dtype=float)
    omega_ana = np.zeros_like(t, dtype=float)
    alpha_ana = np.zeros_like(t, dtype=float)

    th0 = float(theta_initial)
    w0 = float(omega_initial)

    for seg in segments:
        t0 = seg['t0']
        t1 = seg['t1']
        seg_type = seg['type']
        mask = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)
        dt_arr = t[mask] - t0
        dt_seg = t1 - t0

        if seg_type == 'const':
            a0 = seg['alpha0']
            omega_ana[mask] = w0 + a0 * dt_arr
            theta_ana[mask] = th0 + w0 * dt_arr + 0.5 * a0 * (dt_arr ** 2)
            alpha_ana[mask] = a0

            th0 = th0 + w0 * dt_seg + 0.5 * a0 * (dt_seg ** 2)
            w0 = w0 + a0 * dt_seg

        elif seg_type == 'ramp':
            a0 = seg['alpha0']
            j = seg['jerk']
            omega_ana[mask] = w0 + a0 * dt_arr + 0.5 * j * (dt_arr ** 2)
            theta_ana[mask] = th0 + w0 * dt_arr + 0.5 * a0 * (dt_arr ** 2) + (1.0 / 6.0) * j * (dt_arr ** 3)
            alpha_ana[mask] = a0 + j * dt_arr

            th0 = th0 + w0 * dt_seg + 0.5 * a0 * (dt_seg ** 2) + (1.0 / 6.0) * j * (dt_seg ** 3)
            w0 = w0 + a0 * dt_seg + 0.5 * j * (dt_seg ** 2)

        elif seg_type == 'exp':
            w_target = seg['omega_target']
            T_m = seg['T_m']
            exp_arr = np.exp(-dt_arr / T_m)
            omega_ana[mask] = w_target + (w0 - w_target) * exp_arr
            theta_ana[mask] = th0 + w_target * dt_arr + (w0 - w_target) * T_m * (1.0 - exp_arr)
            alpha_ana[mask] = ((w_target - w0) / T_m) * exp_arr

            exp_end = np.exp(-dt_seg / T_m)
            th0 = th0 + w_target * dt_seg + (w0 - w_target) * T_m * (1.0 - exp_end)
            w0 = w_target + (w0 - w_target) * exp_end

    return theta_ana, omega_ana, alpha_ana


# Compute kinematic relative error and piecewise work-energy error percentages
def check_error(cfg, df):
    t = np.asarray(df['t'], dtype=float)
    theta_num = np.asarray(df['theta'], dtype=float)
    omega_num = np.asarray(df['omega'], dtype=float)

    I_val = get_moment_of_inertia(cfg)
    if 'KE' in df:
        KE = np.asarray(df['KE'], dtype=float)
    else:
        KE = 0.5 * I_val * (omega_num ** 2)

    omega_0 = get_omega_rad(cfg, 'omega_initial', 'rpm_initial', omega_num[0])
    theta_0 = float(theta_num[0]) if len(theta_num) > 0 else 0.0

    segments = make_segments(cfg)
    theta_ana, omega_ana, _ = evaluate_piecewise_analytical(t, segments, omega_0, theta_0)

    max_theta = max(np.max(np.abs(theta_ana)), 1e-12)
    max_omega = max(np.max(np.abs(omega_ana)), 1e-12)

    err_theta_pct = (np.max(np.abs(theta_num - theta_ana)) / max_theta) * 100.0
    err_omega_pct = (np.max(np.abs(omega_num - omega_ana)) / max_omega) * 100.0
    rel_err_pct = float(max(err_theta_pct, err_omega_pct))

    work_total = np.zeros_like(t, dtype=float)
    work_accumulated = 0.0
    w_start_seg = omega_0

    for seg in segments:
        t0 = seg['t0']
        t1 = seg['t1']
        seg_type = seg['type']
        mask = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)

        if np.sum(mask) > 1:
            t_seg = t[mask]
            w_seg = omega_num[mask]
            dt_seg = t_seg - t0

            if seg_type == 'const':
                alpha_seg = np.full_like(t_seg, seg['alpha0'])
                w_start_seg = w_start_seg + seg['alpha0'] * (t1 - t0)
            elif seg_type == 'ramp':
                alpha_seg = seg['alpha0'] + seg['jerk'] * dt_seg
                w_start_seg = w_start_seg + seg['alpha0'] * (t1 - t0) + 0.5 * seg['jerk'] * ((t1 - t0) ** 2)
            elif seg_type == 'exp':
                w_target = seg['omega_target']
                T_m = seg['T_m']
                alpha_seg = (w_target - w_seg) / T_m
                w_start_seg = w_target + (w_start_seg - w_target) * np.exp(-(t1 - t0) / T_m)
            else:
                alpha_seg = np.zeros_like(t_seg)

            power_seg = (I_val * alpha_seg) * w_seg
            work_seg = cumulative_trapezoid(power_seg, t_seg, initial=0.0) + work_accumulated
            work_total[mask] = work_seg
            work_accumulated = float(work_seg[-1])

    delta_ke = KE - KE[0]
    max_ke = max(np.max(np.abs(KE)), 1e-12)
    energy_err_pct = float((np.max(np.abs(work_total - delta_ke)) / max_ke) * 100.0)

    return {
        'rel_err_pct': rel_err_pct,
        'err_theta_pct': float(err_theta_pct),
        'err_omega_pct': float(err_omega_pct),
        'energy_err_pct': energy_err_pct
    }