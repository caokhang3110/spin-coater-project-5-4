import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import json
import matplotlib.pyplot as plt
from physic_quantities import calc_moment_of_inertia, calc_physical_quantities

# Solve angular motion for linear acceleration profile (constant alpha + hold step)
def solve_linear_profile(omega_inital, omega_final, alpha_constant, t_max, dt):
    if omega_inital <= omega_final:
        a_values = alpha_constant
    else:
        a_values = -alpha_constant
    t_ramp = abs(omega_final - omega_inital) / alpha_constant
    
    # Ramp step (0 <= t <= t_ramp)
    t1 = np.arange(0, t_ramp, dt)
    if len(t1) == 0 or not np.isclose(t1[-1], t_ramp):
        t1 = np.append(t1, t_ramp)
    sol1 = solve_ivp(lambda t, y: [y[1], a_values], (0, t_ramp), [0.0, omega_inital],
                     method="RK45", t_eval=t1, max_step=dt, rtol=1e-9, atol=1e-12)
    
    # Hold step (t_ramp < t <= t_max)
    if t_max > t_ramp:
        t2 = np.arange(t_ramp, t_max, dt)
        if len(t2) == 0 or not np.isclose(t2[-1], t_max):
            t2 = np.append(t2, t_max)
        sol2 = solve_ivp(lambda t, y: [y[1], 0.0], (t_ramp, t_max), [sol1.y[0, -1], sol1.y[1, -1]],
                         method="RK45", t_eval=t2, max_step=dt, rtol=1e-9, atol=1e-12)
        t = np.concatenate([t1, t2[1:]])
        theta = np.concatenate([sol1.y[0], sol2.y[0, 1:]])
        omega = np.concatenate([sol1.y[1], sol2.y[1, 1:]])
    else:
        t, theta, omega = t1, sol1.y[0], sol1.y[1]
    alpha = np.where(t <= t_ramp, a_values, 0.0)
    return t, theta, omega, alpha, t_ramp

# Solve angular motion for exponential relaxation profile with time constant T_m
def solve_exp_profile(omega_initial, omega_final, alpha_constant, t_max, dt, T_m=None, t_ramp=None):
    T_constant = T_m if T_m is not None else abs(omega_final - omega_initial) / alpha_constant
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
    sol1 = solve_ivp(rhs_exp, (0, t_ramp), [0.0, omega_initial],
                     method="RK45", t_eval=t1, max_step=dt, rtol=1e-9, atol=1e-12)
                     
    # Hold step (t_ramp < t <= t_max)
    if t_max > t_ramp:
        t2 = np.arange(t_ramp, t_max, dt)
        if len(t2) == 0 or not np.isclose(t2[-1], t_max):
            t2 = np.append(t2, t_max)
        sol2 = solve_ivp(lambda t, y: [y[1], 0.0], (t_ramp, t_max), [sol1.y[0, -1], sol1.y[1, -1]],
                         method="RK45", t_eval=t2, max_step=dt, rtol=1e-9, atol=1e-12)
        t = np.concatenate([t1, t2[1:]])
        theta = np.concatenate([sol1.y[0], sol2.y[0, 1:]])
        omega = np.concatenate([sol1.y[1], sol2.y[1, 1:]])
    else:
        t, theta, omega = t1, sol1.y[0], sol1.y[1]
    alpha = np.where(t <= t_ramp, (omega_final - omega) / T_constant, 0.0)
    return t, theta, omega, alpha, t_ramp

# Solve angular motion for 3-phase S-curve profile (jerk-up, constant accel, jerk-down)
def solve_scurve_profile(omega_initial, omega_final, alpha_constant, t_max, dt, t_j=0.5):
    sign = 1.0 if omega_final >= omega_initial else -1.0
    A = sign * alpha_constant
    j = A / t_j
    t_ramp = abs(omega_final - omega_initial) / alpha_constant + t_j
    
    # Piecewise angular acceleration for S-curve
    def alpha_scurve(t_val):
        if t_val <= t_j:
            return j * t_val
        elif t_val <= t_ramp - t_j:
            return A
        elif t_val <= t_ramp:
            return A - j * (t_val - (t_ramp - t_j))
        else:
            return 0.0
    def rhs_scurve(t_val, y):
        return [y[1], alpha_scurve(t_val)]
        
    # Include transition knots in time array for accurate integration
    t = np.arange(0, t_max, dt)
    knots = [t_j, t_ramp - t_j, t_ramp, t_max]
    t = np.unique(np.sort(np.concatenate([t, [k for k in knots if k <= t_max]])))
    sol = solve_ivp(rhs_scurve, (0, t_max), [0.0, omega_initial],
                    method="RK45", t_eval=t, max_step=dt, rtol=1e-9, atol=1e-12)
    theta = sol.y[0]
    omega = sol.y[1]
    alpha = np.array([alpha_scurve(ti) for ti in t])
    return t, theta, omega, alpha, t_ramp

# Check analytical relative error (Error 1) and work-energy theorem error in % (Error 2)
def check_error(t, theta, omega, P, KE, omega_initial, omega_final, alpha_constant, t_max, system, t_ramp, t_j=0.5, T_m=None):
    theta_ana = np.zeros_like(t)
    omega_ana = np.zeros_like(t)
    sign = 1.0 if omega_final >= omega_initial else -1.0
    
    # Masks for ramp step (m1) and hold step (m2)
    m1 = t <= t_ramp
    m2 = t > t_ramp
    t1 = t[m1]
    t2 = t[m2] - t_ramp

    # Error 1: Sequential analytical solution (ramp step + hold step) across time array t
    if system == "linear":
        a_val = sign * alpha_constant
        omega_ana[m1] = omega_initial + a_val * t1
        theta_ana[m1] = omega_initial * t1 + 0.5 * a_val * t1**2
        theta_end_ramp = 0.5 * (omega_initial + omega_final) * t_ramp
        omega_ana[m2] = omega_final
        theta_ana[m2] = theta_end_ramp + omega_final * t2

    elif system == "exp":
        T_c = T_m if T_m is not None else abs(omega_final - omega_initial) / alpha_constant
        omega_ana[m1] = omega_final + (omega_initial - omega_final) * np.exp(-t1 / T_c)
        theta_ana[m1] = omega_final * t1 + (omega_initial - omega_final) * T_c * (1.0 - np.exp(-t1 / T_c))
        omega_end_ramp = omega_final + (omega_initial - omega_final) * np.exp(-t_ramp / T_c)
        theta_end_ramp = omega_final * t_ramp + (omega_initial - omega_final) * T_c * (1.0 - np.exp(-t_ramp / T_c))
        omega_ana[m2] = omega_end_ramp
        theta_ana[m2] = theta_end_ramp + omega_end_ramp * t2

    elif system == "scurve":
        A = sign * alpha_constant
        j = A / t_j
        w1 = omega_initial + 0.5 * A * t_j
        th1 = omega_initial * t_j + (1.0 / 6.0) * A * t_j**2
        dt_p2 = t_ramp - 2.0 * t_j
        w2 = w1 + A * dt_p2
        th2 = th1 + w1 * dt_p2 + 0.5 * A * dt_p2**2

        # NumPy boolean masks for the 3 S-curve phases
        p1 = t <= t_j
        p2 = (t > t_j) & (t <= t_ramp - t_j)
        p3 = (t > t_ramp - t_j) & (t <= t_ramp)

        # Phase 1: Jerk-up (0 <= t <= t_j)
        omega_ana[p1] = omega_initial + 0.5 * j * t[p1]**2
        theta_ana[p1] = omega_initial * t[p1] + (1.0 / 6.0) * j * t[p1]**3

        # Phase 2: Constant acceleration (t_j < t <= t_ramp - t_j)
        tau = t[p2] - t_j
        omega_ana[p2] = w1 + A * tau
        theta_ana[p2] = th1 + w1 * tau + 0.5 * A * tau**2

        # Phase 3: Jerk-down (t_ramp - t_j < t <= t_ramp)
        s = t[p3] - (t_ramp - t_j)
        omega_ana[p3] = w2 + A * s - 0.5 * j * s**2
        theta_ana[p3] = th2 + w2 * s + 0.5 * A * s**2 - (1.0 / 6.0) * j * s**3

        # Hold step after ramp (t > t_ramp)
        theta_end_ramp = 0.5 * (omega_initial + omega_final) * t_ramp
        omega_ana[m2] = omega_final
        theta_ana[m2] = theta_end_ramp + omega_final * t2

    else:
        theta_ana[:] = np.nan

    # Compute maximum relative error (%) between numerical and analytical solutions
    if np.all(np.isnan(theta_ana)):
        rel_err = np.nan
    else:
        valid_th = np.abs(theta_ana) >= 1e-6
        valid_w = np.abs(omega_ana) >= 1e-6
        err_th = np.max(np.abs(theta[valid_th] - theta_ana[valid_th]) / np.abs(theta_ana[valid_th])) * 100
        err_w = np.max(np.abs(omega[valid_w] - omega_ana[valid_w]) / np.abs(omega_ana[valid_w])) * 100
        rel_err = max(err_th, err_w)

    # Error 2: Integrate power P separately on each step and compute work-energy error (%)
    W = np.zeros_like(t)
    mask1 = t <= t_ramp
    mask2 = t >= t_ramp
    W[mask1] = cumulative_trapezoid(P[mask1], t[mask1], initial=0)
    if np.sum(mask2) > 1:
        # Set power at junction t = t_ramp to 0 for the hold step (since alpha = 0 in hold)
        P_hold = P[mask2].copy()
        P_hold[0] = 0.0
        W[mask2] = W[mask1][-1] + cumulative_trapezoid(P_hold, t[mask2], initial=0)

    delta_K = KE - KE[0]
    energy_err_pct = np.max(np.abs(W - delta_K)) / np.max(KE) * 100
    return rel_err, energy_err_pct

# Load simulation parameters from JSON configuration
f = open("input.json", "r")
data = json.load(f)
f.close()
R = data["R"]
d = data["d"]
rho = data["rho"]
omega_initial = data["omega_initial"]
omega_final = data["omega_final"]
alpha_constant = data["alpha_constant"]
t_max = data["t_max"]
dt = data["dt"]
system = data["system"]
t_j = data.get("t_j", 0.5)
T_m = data.get("T_m", None)

# Compute wafer moment of inertia and solve motion for the selected profile
m, I = calc_moment_of_inertia(R, d, rho)
if system == "linear":
    t, theta, omega, alpha, t_ramp = solve_linear_profile(omega_initial, omega_final, alpha_constant, t_max, dt)
elif system == "scurve":
    t, theta, omega, alpha, t_ramp = solve_scurve_profile(omega_initial, omega_final, alpha_constant, t_max, dt, t_j=t_j)
else:
    t, theta, omega, alpha, t_ramp = solve_exp_profile(omega_initial, omega_final, alpha_constant, t_max, dt, T_m=T_m)

# Compute derived physical quantities and verify numerical errors
rpm, jerk, at, ac, atotal, torque, KE, P = calc_physical_quantities(t, theta, omega, alpha, R, I)
rel_err, energy_err_pct = check_error(t, theta, omega, P, KE, omega_initial, omega_final, alpha_constant, t_max, system, t_ramp, t_j=t_j, T_m=T_m)
print("rel_err (%):", rel_err)
print("energy_err_pct (%):", energy_err_pct)

# Plot kinematic and dynamic quantities over time
plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(t, rpm)
plt.title("RPM vs Time")
plt.xlabel("Time (s)")
plt.ylabel("RPM")
plt.subplot(3, 2, 2)
plt.plot(t, jerk)
plt.title("Jerk vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Jerk (rad/s^3)")
plt.subplot(3, 2, 3)
plt.plot(t, at)
plt.title("Tangential Acceleration vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Tangential Acceleration (m/s^2)")
plt.subplot(3, 2, 4)
plt.plot(t, ac)
plt.title("Centripetal Acceleration vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Centripetal Acceleration (m/s^2)")
plt.subplot(3, 2, 5)
plt.plot(t, torque)
plt.title("Torque vs Time")
plt.xlabel("Time(s)")
plt.ylabel("Torque(N.m)")
plt.subplot(3, 2, 6)
plt.plot(t, P)
plt.title("Power vs Time")
plt.xlabel("Time(s)")
plt.ylabel("Power(W)")
plt.tight_layout()
plt.show()