import json
import matplotlib.pyplot as plt
import numpy as np
from physic_quantities import calc_moment_of_inertia, calc_physical_quantities
from scipy.integrate import cumulative_trapezoid, solve_ivp


# Solve angular motion for linear acceleration profile (constant alpha + hold step)
def solve_linear_profile(
    omega_initial, omega_final, alpha_constant, t_max, dt
):
  sign = 1.0 if omega_final >= omega_initial else -1.0
  a_values = sign * alpha_constant
  t_ramp = abs(omega_final - omega_initial) / alpha_constant

  # Ramp step (0 <= t <= t_ramp)
  t1 = np.arange(0, t_ramp, dt)
  if len(t1) == 0 or not np.isclose(t1[-1], t_ramp):
    t1 = np.append(t1, t_ramp)
  sol1 = solve_ivp(
      lambda t, y: [y[1], a_values],
      (0, t_ramp),
      [0.0, omega_initial],
      method='RK45',
      t_eval=t1,
      max_step=dt,
      rtol=1e-9,
      atol=1e-12,
  )

  # Hold step (t_ramp < t <= t_max)
  if t_max > t_ramp:
    t2 = np.arange(t_ramp, t_max, dt)
    if len(t2) == 0 or not np.isclose(t2[-1], t_max):
      t2 = np.append(t2, t_max)
    sol2 = solve_ivp(
        lambda t, y: [y[1], 0.0],
        (t_ramp, t_max),
        [sol1.y[0, -1], sol1.y[1, -1]],
        method='RK45',
        t_eval=t2,
        max_step=dt,
        rtol=1e-9,
        atol=1e-12,
    )
    t = np.concatenate([t1, t2[1:]])
    theta = np.concatenate([sol1.y[0], sol2.y[0, 1:]])
    omega = np.concatenate([sol1.y[1], sol2.y[1, 1:]])
  else:
    t, theta, omega = t1, sol1.y[0], sol1.y[1]

  alpha = np.where(t <= t_ramp, a_values, 0.0)
  segments = [t1, t2] if t_max > t_ramp else [t1]
  return t, theta, omega, alpha, t_ramp, segments


# Solve angular motion for exponential relaxation profile with time constant T_m
def solve_exp_profile(
    omega_initial,
    omega_final,
    alpha_constant,
    t_max,
    dt,
    T_m=None,
    t_ramp=None,
):
  T_constant = (
      T_m
      if T_m is not None
      else abs(omega_final - omega_initial) / alpha_constant
  )
  if t_ramp is None:
    t_ramp = t_max

  def rhs_exp(t, y):
    theta, omega = y
    alpha = (omega_final - omega) / T_constant
    return [omega, alpha]

  # Exponential ramp step (0 <= t <= t_ramp)
  t1 = np.arange(0, t_ramp, dt)
  if len(t1) == 0 or not np.isclose(t1[-1], t_ramp):
    t1 = np.append(t1, t_ramp)
  sol1 = solve_ivp(
      rhs_exp,
      (0, t_ramp),
      [0.0, omega_initial],
      method='RK45',
      t_eval=t1,
      max_step=dt,
      rtol=1e-9,
      atol=1e-12,
  )

  # Hold step (t_ramp < t <= t_max)
  if t_max > t_ramp:
    t2 = np.arange(t_ramp, t_max, dt)
    if len(t2) == 0 or not np.isclose(t2[-1], t_max):
      t2 = np.append(t2, t_max)
    sol2 = solve_ivp(
        lambda t, y: [y[1], 0.0],
        (t_ramp, t_max),
        [sol1.y[0, -1], sol1.y[1, -1]],
        method='RK45',
        t_eval=t2,
        max_step=dt,
        rtol=1e-9,
        atol=1e-12,
    )
    t = np.concatenate([t1, t2[1:]])
    theta = np.concatenate([sol1.y[0], sol2.y[0, 1:]])
    omega = np.concatenate([sol1.y[1], sol2.y[1, 1:]])
  else:
    t, theta, omega = t1, sol1.y[0], sol1.y[1]

  alpha = np.where(t <= t_ramp, (omega_final - omega) / T_constant, 0.0)
  segments = [t1, t2] if t_max > t_ramp else [t1]
  return t, theta, omega, alpha, t_ramp, segments


# Solve angular motion for S-curve profile (handles both Trapezoidal and Triangular S-curves)
def solve_scurve_profile(
    omega_initial, omega_final, alpha_constant, t_max, dt, t_j=0.5
):
  sign = 1.0 if omega_final >= omega_initial else -1.0
  delta_omega = abs(omega_final - omega_initial)
  j_nom = sign * alpha_constant / t_j

  # Check if delta_omega < alpha_constant * t_j (Triangular S-curve)
  is_triangular = delta_omega < (alpha_constant * t_j)

  if is_triangular:
    t_prime = np.sqrt(delta_omega / (alpha_constant / t_j))
    A_prime = sign * np.sqrt((alpha_constant / t_j) * delta_omega)
    t_ramp = 2.0 * t_prime

    def alpha_scurve(t_val):
      if t_val <= t_prime:
        return j_nom * t_val
      elif t_val <= t_ramp:
        return A_prime - j_nom * (t_val - t_prime)
      else:
        return 0.0

    knots = [t_prime, t_ramp, t_max]
  else:
    A = sign * alpha_constant
    t_ramp = delta_omega / alpha_constant + t_j

    def alpha_scurve(t_val):
      if t_val <= t_j:
        return j_nom * t_val
      elif t_val <= t_ramp - t_j:
        return A
      elif t_val <= t_ramp:
        return A - j_nom * (t_val - (t_ramp - t_j))
      else:
        return 0.0

    knots = [t_j, t_ramp - t_j, t_ramp, t_max]

  def rhs_scurve(t_val, y):
    return [y[1], alpha_scurve(t_val)]

  t = np.arange(0, t_max, dt)
  t = np.unique(np.sort(np.concatenate([t, [k for k in knots if k <= t_max]])))

  sol = solve_ivp(
      rhs_scurve,
      (0, t_max),
      [0.0, omega_initial],
      method='RK45',
      t_eval=t,
      max_step=dt,
      rtol=1e-9,
      atol=1e-12,
  )
  theta = sol.y[0]
  omega = sol.y[1]
  alpha = np.array([alpha_scurve(ti) for ti in t])
  return t, theta, omega, alpha, t_ramp, knots


def check_error(
    t,
    theta,
    omega,
    P,
    KE,
    omega_initial,
    omega_final,
    alpha_constant,
    t_max,
    system,
    t_ramp,
    t_j=0.5,
    T_m=None,
    I=0.01,
):
  sys_type = system.lower().replace('_', '')
  sign = 1.0 if omega_final >= omega_initial else -1.0
  delta_omega = abs(omega_final - omega_initial)

  theta_ana = np.zeros_like(t)
  omega_ana = np.zeros_like(t)

  # 1. Tính nghiệm giải tích chính xác
  if sys_type == 'linear':
    a_val = sign * alpha_constant
    m1 = t <= t_ramp
    m2 = t > t_ramp
    t1 = t[m1]
    t2 = t[m2] - t_ramp

    omega_ana[m1] = omega_initial + a_val * t1
    theta_ana[m1] = omega_initial * t1 + 0.5 * a_val * t1**2

    theta_end_ramp = omega_initial * t_ramp + 0.5 * a_val * t_ramp**2
    omega_ana[m2] = omega_final
    theta_ana[m2] = theta_end_ramp + omega_final * t2

  elif sys_type in ['exp', 'exponential']:
    T_c = (
        T_m
        if T_m is not None
        else (delta_omega / alpha_constant if alpha_constant != 0 else 1.0)
    )
    m1 = t <= t_ramp
    m2 = t > t_ramp
    t1 = t[m1]
    t2 = t[m2] - t_ramp

    omega_ana[m1] = omega_final + (omega_initial - omega_final) * np.exp(
        -t1 / T_c
    )
    theta_ana[m1] = (
        omega_final * t1
        + (omega_initial - omega_final) * T_c * (1.0 - np.exp(-t1 / T_c))
    )

    omega_end_ramp = omega_final + (omega_initial - omega_final) * np.exp(
        -t_ramp / T_c
    )
    theta_end_ramp = (
        omega_final * t_ramp
        + (omega_initial - omega_final) * T_c * (1.0 - np.exp(-t_ramp / T_c))
    )
    omega_ana[m2] = omega_end_ramp
    theta_ana[m2] = theta_end_ramp + omega_end_ramp * t2

  elif sys_type in ['scurve', 's_curve']:
    j = sign * alpha_constant / t_j
    is_triangular = delta_omega < (alpha_constant * t_j)

    if is_triangular:
      t_prime = np.sqrt(delta_omega / (alpha_constant / t_j))
      A_prime = sign * np.sqrt((alpha_constant / t_j) * delta_omega)

      p1 = t <= t_prime
      p2 = (t > t_prime) & (t <= 2 * t_prime)
      p3 = t > 2 * t_prime

      dt1 = t[p1]
      omega_ana[p1] = omega_initial + 0.5 * j * dt1**2
      theta_ana[p1] = omega_initial * dt1 + (1.0 / 6.0) * j * dt1**3

      w1 = omega_initial + 0.5 * j * t_prime**2
      th1 = omega_initial * t_prime + (1.0 / 6.0) * j * t_prime**3

      dt2 = t[p2] - t_prime
      omega_ana[p2] = w1 + A_prime * dt2 - 0.5 * j * dt2**2
      theta_ana[p2] = (
          th1 + w1 * dt2 + 0.5 * A_prime * dt2**2 - (1.0 / 6.0) * j * dt2**3
      )

      dt3 = t[p3] - 2 * t_prime
      omega_ana[p3] = omega_final
      th2 = (
          th1
          + w1 * t_prime
          + 0.5 * A_prime * t_prime**2
          - (1.0 / 6.0) * j * t_prime**3
      )
      theta_ana[p3] = th2 + omega_final * dt3
    else:
      A = sign * alpha_constant
      p1 = t <= t_j
      p2 = (t > t_j) & (t <= t_ramp - t_j)
      p3 = (t > t_ramp - t_j) & (t <= t_ramp)
      p4 = t > t_ramp

      dt1 = t[p1]
      omega_ana[p1] = omega_initial + 0.5 * j * dt1**2
      theta_ana[p1] = omega_initial * dt1 + (1.0 / 6.0) * j * dt1**3

      w1 = omega_initial + 0.5 * j * t_j**2
      th1 = omega_initial * t_j + (1.0 / 6.0) * j * t_j**3

      dt2 = t[p2] - t_j
      omega_ana[p2] = w1 + A * dt2
      theta_ana[p2] = th1 + w1 * dt2 + 0.5 * A * dt2**2

      dt_p2 = t_ramp - 2.0 * t_j
      w2 = w1 + A * dt_p2
      th2 = th1 + w1 * dt_p2 + 0.5 * A * dt_p2**2

      dt3 = t[p3] - (t_ramp - t_j)
      omega_ana[p3] = w2 + A * dt3 - 0.5 * j * dt3**2
      theta_ana[p3] = (
          th2 + w2 * dt3 + 0.5 * A * dt3**2 - (1.0 / 6.0) * j * dt3**3
      )

      th3 = th2 + w2 * t_j + 0.5 * A * t_j**2 - (1.0 / 6.0) * j * t_j**3

      dt4 = t[p4] - t_ramp
      omega_ana[p4] = omega_final
      theta_ana[p4] = th3 + omega_final * dt4
  else:
    theta_ana[:] = np.nan

  max_th = (
      np.max(np.abs(theta_ana)) if np.max(np.abs(theta_ana)) > 1e-12 else 1.0
  )
  max_w = max(np.max(np.abs(omega_ana)), delta_omega, 1e-12)

  err_th = (np.max(np.abs(theta - theta_ana)) / max_th) * 100.0
  err_w = (np.max(np.abs(omega - omega_ana)) / max_w) * 100.0
  rel_err = max(err_th, err_w)

  # 3. Energy errors
  if sys_type in ['scurve', 's_curve']:
    is_triangular = delta_omega < (alpha_constant * t_j)
    if is_triangular:
      t_prime = np.sqrt(delta_omega / (alpha_constant / t_j))
      boundaries = [0.0, t_prime, 2 * t_prime, t_max]
    else:
      boundaries = [0.0, t_j, t_ramp - t_j, t_ramp, t_max]
  else:
    boundaries = [0.0, t_ramp, t_max]

  boundaries = sorted(list(set([b for b in boundaries if b <= t_max])))

  def get_alpha_segment(t_val, seg_idx):
    if sys_type == 'linear':
      return sign * alpha_constant if seg_idx == 0 else 0.0
    elif sys_type in ['exp', 'exponential']:
      T_c = (
          T_m
          if T_m is not None
          else (delta_omega / alpha_constant if alpha_constant != 0 else 1.0)
      )
      if seg_idx == 0:
        w_t = omega_final + (omega_initial - omega_final) * np.exp(-t_val / T_c)
        return (omega_final - w_t) / T_c
      else:
        return 0.0
    elif sys_type in ['scurve', 's_curve']:
      j_val = sign * alpha_constant / t_j
      is_triangular = delta_omega < (alpha_constant * t_j)
      if is_triangular:
        t_prime = np.sqrt(delta_omega / (alpha_constant / t_j))
        A_prime = sign * np.sqrt((alpha_constant / t_j) * delta_omega)
        if seg_idx == 0:
          return j_val * t_val
        elif seg_idx == 1:
          return A_prime - j_val * (t_val - t_prime)
        else:
          return 0.0
      else:
        A_val = sign * alpha_constant
        if seg_idx == 0:
          return j_val * t_val
        elif seg_idx == 1:
          return A_val
        elif seg_idx == 2:
          return A_val - j_val * (t_val - (t_ramp - t_j))
        else:
          return 0.0
    return 0.0

  W = np.zeros_like(t)
  W_accumulated = 0.0

  for idx in range(len(boundaries) - 1):
    t_start, t_end = boundaries[idx], boundaries[idx + 1]
    mask = (t >= t_start) & (t <= t_end)
    if np.sum(mask) > 1:
      t_seg = t[mask]
      w_seg = omega[mask]
      alpha_seg = np.array([get_alpha_segment(ti, idx) for ti in t_seg])
      P_seg = (I * alpha_seg) * w_seg

      W_seg = cumulative_trapezoid(P_seg, t_seg, initial=0) + W_accumulated
      W[mask] = W_seg
      W_accumulated = W_seg[-1]

  delta_K = KE - KE[0]
  max_KE = np.max(KE) if np.max(KE) > 1e-12 else 1.0
  energy_err_pct = (np.max(np.abs(W - delta_K)) / max_KE) * 100.0

  return rel_err, energy_err_pct


# Json reader
with open('input.json', 'r') as f:
  data = json.load(f)

R = data['R']
d = data['d']
rho = data['rho']
omega_initial = data['omega_initial']
omega_final = data['omega_final']
alpha_constant = data['alpha_constant']
t_max = data['t_max']
dt = data['dt']
system = data.get('system', data.get('profile', 'linear'))
t_j = data.get('t_j', data.get('t_jerk', 0.5))
T_m = data.get('T_m', data.get('T_exp', 0.65))

# Calculate moment of inertia
m, I = calc_moment_of_inertia(R, d, rho)
sys_type = system.lower().replace('_', '')

if sys_type == 'linear':
  t, theta, omega, alpha, t_ramp, _ = solve_linear_profile(
      omega_initial, omega_final, alpha_constant, t_max, dt
  )
elif sys_type in ['scurve', 's_curve']:
  t, theta, omega, alpha, t_ramp, _ = solve_scurve_profile(
      omega_initial, omega_final, alpha_constant, t_max, dt, t_j=t_j
  )
else:
  t, theta, omega, alpha, t_ramp, _ = solve_exp_profile(
      omega_initial, omega_final, alpha_constant, t_max, dt, T_m=T_m
  )

# Errors
rpm, jerk, at, ac, atotal, torque, KE, P = calc_physical_quantities(
    t, theta, omega, alpha, R, I
)
rel_err, energy_err_pct = check_error(
    t,
    theta,
    omega,
    P,
    KE,
    omega_initial,
    omega_final,
    alpha_constant,
    t_max,
    system,
    t_ramp,
    t_j=t_j,
    T_m=T_m,
    I=I,
)

print(f'Rel Error (%): {rel_err:.6e}%')
print(f'Energy Error (%): {energy_err_pct:.6f}%')