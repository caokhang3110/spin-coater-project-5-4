import numpy as np


# Convert angular velocity from RPM to rad/s or extract rad/s directly from configuration
def get_omega_rad(step_cfg, key_omega, key_rpm, default_val=0.0):
    if key_omega in step_cfg and step_cfg[key_omega] is not None:
        return float(step_cfg[key_omega])
    if key_rpm in step_cfg and step_cfg[key_rpm] is not None:
        return float(step_cfg[key_rpm]) * 2.0 * np.pi / 60.0
    return float(default_val)


# Compute minimum required ramp duration for a single motion step including triangular S-curve
def compute_min_step_duration(omega_0, omega_f, alpha_c, profile_type, t_j=0.5):
    delta_omega = abs(omega_f - omega_0)
    if delta_omega < 1e-12:
        return 0.0
    p_type = profile_type.lower().replace('_', '').replace('-', '')
    if p_type == 'linear':
        return delta_omega / alpha_c
    elif p_type == 'scurve':
        if delta_omega < alpha_c * t_j:
            t_prime = np.sqrt(delta_omega * t_j / alpha_c)
            return 2.0 * t_prime
        return (delta_omega / alpha_c) + t_j
    return 0.0


# Validate whether configured step duration is sufficient for the target velocity change
def validate_duration(cfg):
    if 'steps' in cfg and isinstance(cfg['steps'], list):
        omega_curr = get_omega_rad(cfg, 'omega_initial', 'rpm_initial', 0.0)
        for step in cfg['steps']:
            omega_next = get_omega_rad(step, 'omega_final', 'rpm_final', step.get('omega', omega_curr))
            alpha_c = float(step.get('alpha_constant', cfg.get('alpha_constant', 100.0)))
            t_j = float(step.get('t_j', step.get('t_jerk', cfg.get('t_j', 0.5))))
            p_type = step.get('profile', step.get('system', 'linear'))
            duration = float(step.get('duration', step.get('t_max', 0.0)))
            min_dur = compute_min_step_duration(omega_curr, omega_next, alpha_c, p_type, t_j)
            if duration + 1e-9 < min_dur:
                raise ValueError(f"Step duration ({duration:.4f}s) is shorter than minimum ramp time ({min_dur:.4f}s).")
            omega_curr = omega_next
        return True

    omega_0 = get_omega_rad(cfg, 'omega_initial', 'rpm_initial', 0.0)
    omega_f = get_omega_rad(cfg, 'omega_final', 'rpm_final', 0.0)
    alpha_c = float(cfg.get('alpha_constant', 100.0))
    t_j = float(cfg.get('t_j', cfg.get('t_jerk', 0.5)))
    p_type = cfg.get('system', cfg.get('profile', 'linear'))
    t_max = float(cfg.get('t_max', cfg.get('duration', 0.0)))
    min_dur = compute_min_step_duration(omega_0, omega_f, alpha_c, p_type, t_j)
    if t_max + 1e-9 < min_dur:
        raise ValueError(f"Total duration ({t_max:.4f}s) is shorter than minimum ramp time ({min_dur:.4f}s).")
    return True


# Build elementary kinematic segments (const, ramp, exp) for a single motion step
def build_single_step_segments(t_start, duration, omega_0, omega_f, alpha_c, profile_type, t_j=0.5, T_m=None):
    t_end = t_start + duration
    delta_omega = abs(omega_f - omega_0)
    sign = 1.0 if omega_f >= omega_0 else -1.0
    p_type = profile_type.lower().replace('_', '').replace('-', '')
    segments = []

    if delta_omega < 1e-12 or p_type == 'hold':
        segments.append({
            'type': 'const',
            't0': t_start,
            't1': t_end,
            'alpha0': 0.0,
            'jerk': 0.0,
            'omega_target': omega_f,
            'T_m': 1.0
        })
        return segments

    if p_type == 'linear':
        t_ramp = min(delta_omega / alpha_c, duration)
        segments.append({
            'type': 'const',
            't0': t_start,
            't1': t_start + t_ramp,
            'alpha0': sign * alpha_c,
            'jerk': 0.0,
            'omega_target': omega_f,
            'T_m': 1.0
        })
        if t_start + t_ramp < t_end - 1e-12:
            segments.append({
                'type': 'const',
                't0': t_start + t_ramp,
                't1': t_end,
                'alpha0': 0.0,
                'jerk': 0.0,
                'omega_target': omega_f,
                'T_m': 1.0
            })

    elif p_type in ['exp', 'exponential']:
        T_c = float(T_m) if T_m is not None else (delta_omega / alpha_c if alpha_c > 0 else 1.0)
        segments.append({
            'type': 'exp',
            't0': t_start,
            't1': t_end,
            'alpha0': 0.0,
            'jerk': 0.0,
            'omega_target': omega_f,
            'T_m': T_c
        })

    elif p_type == 'scurve':
        j_val = sign * (alpha_c / t_j)
        if delta_omega < alpha_c * t_j:
            t_prime = np.sqrt(delta_omega * t_j / alpha_c)
            a_peak = j_val * t_prime
            t1 = min(t_start + t_prime, t_end)
            t2 = min(t_start + 2.0 * t_prime, t_end)
            segments.append({
                'type': 'ramp',
                't0': t_start,
                't1': t1,
                'alpha0': 0.0,
                'jerk': j_val,
                'omega_target': omega_f,
                'T_m': 1.0
            })
            if t2 > t1 + 1e-12:
                segments.append({
                    'type': 'ramp',
                    't0': t1,
                    't1': t2,
                    'alpha0': a_peak,
                    'jerk': -j_val,
                    'omega_target': omega_f,
                    'T_m': 1.0
                })
            if t_end > t2 + 1e-12:
                segments.append({
                    'type': 'const',
                    't0': t2,
                    't1': t_end,
                    'alpha0': 0.0,
                    'jerk': 0.0,
                    'omega_target': omega_f,
                    'T_m': 1.0
                })
        else:
            t_ramp = (delta_omega / alpha_c) + t_j
            a_const = sign * alpha_c
            t1 = min(t_start + t_j, t_end)
            t2 = min(t_start + t_ramp - t_j, t_end)
            t3 = min(t_start + t_ramp, t_end)
            segments.append({
                'type': 'ramp',
                't0': t_start,
                't1': t1,
                'alpha0': 0.0,
                'jerk': j_val,
                'omega_target': omega_f,
                'T_m': 1.0
            })
            if t2 > t1 + 1e-12:
                segments.append({
                    'type': 'const',
                    't0': t1,
                    't1': t2,
                    'alpha0': a_const,
                    'jerk': 0.0,
                    'omega_target': omega_f,
                    'T_m': 1.0
                })
            if t3 > t2 + 1e-12:
                segments.append({
                    'type': 'ramp',
                    't0': t2,
                    't1': t3,
                    'alpha0': a_const,
                    'jerk': -j_val,
                    'omega_target': omega_f,
                    'T_m': 1.0
                })
            if t_end > t3 + 1e-12:
                segments.append({
                    'type': 'const',
                    't0': t3,
                    't1': t_end,
                    'alpha0': 0.0,
                    'jerk': 0.0,
                    'omega_target': omega_f,
                    'T_m': 1.0
                })

    return segments


# Generate full list of piecewise segments from single-step or multi-step configuration
def make_segments(cfg):
    validate_duration(cfg)
    if 'steps' in cfg and isinstance(cfg['steps'], list):
        all_segments = []
        t_curr = 0.0
        omega_curr = get_omega_rad(cfg, 'omega_initial', 'rpm_initial', 0.0)
        for step in cfg['steps']:
            duration = float(step.get('duration', step.get('t_max', 0.0)))
            omega_next = get_omega_rad(step, 'omega_final', 'rpm_final', step.get('omega', omega_curr))
            alpha_c = float(step.get('alpha_constant', cfg.get('alpha_constant', 100.0)))
            t_j = float(step.get('t_j', step.get('t_jerk', cfg.get('t_j', 0.5))))
            T_m = step.get('T_m', step.get('T_exp', cfg.get('T_m', None)))
            p_type = step.get('profile', step.get('system', 'linear'))
            segs = build_single_step_segments(t_curr, duration, omega_curr, omega_next, alpha_c, p_type, t_j, T_m)
            all_segments.extend(segs)
            t_curr += duration
            omega_curr = omega_next
        return all_segments

    t_max = float(cfg.get('t_max', cfg.get('duration', 5.0)))
    omega_0 = get_omega_rad(cfg, 'omega_initial', 'rpm_initial', 0.0)
    omega_f = get_omega_rad(cfg, 'omega_final', 'rpm_final', 0.0)
    alpha_c = float(cfg.get('alpha_constant', 100.0))
    t_j = float(cfg.get('t_j', cfg.get('t_jerk', 0.5)))
    T_m = cfg.get('T_m', cfg.get('T_exp', None))
    p_type = cfg.get('system', cfg.get('profile', 'linear'))
    return build_single_step_segments(0.0, t_max, omega_0, omega_f, alpha_c, p_type, t_j, T_m)