import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import json
import matplotlib.pyplot as plt
from physic_quantities import calc_moment_of_inertia, calc_physical_quantities
def solve_linear_profile(omega_inital, omega_final, alpha_constant, t_max, dt):
    if omega_inital <= omega_final:
        a_values = alpha_constant
    else :
        a_values = - alpha_constant
    t_ramp = abs(omega_final-omega_inital)/alpha_constant
    def rhs(t,y):
        theta, omega = y
        if t <= t_ramp:
            alpha = a_values
        else:
            alpha = 0.0
        return [omega, alpha]
    t = np.arange(0, t_max + dt/2, dt)
    sol = solve_ivp(rhs, (0, t_max), [0, omega_inital], method="RK45", t_eval=t, max_step=dt)  
    theta = sol.y[0]
    omega = sol.y[1]
    alpha = np.gradient(omega, t)
    return t, theta, omega, alpha 
def solve_exp_profile(omega_initial, omega_final, alpha_constant, t_max, dt):
    T_constant = abs(omega_final - omega_initial)/alpha_constant
    def rhs(t,y):
        theta, omega = y
        alpha = abs(omega_final-omega)/T_constant
        return [omega, alpha]
    t = np.arange(0, t_max + dt/2, dt)
    sol = solve_ivp(rhs, (0, t_max), [0, omega_initial], method="RK45", t_eval=t, max_step=dt)
    theta = sol.y[0]
    omega = sol.y[1]
    alpha = np.gradient(omega, t)
    return t, theta, omega, alpha    
# Check analytical error (Error 1) and work-energy error in percentage (Error 2)
def check_error(t, theta, P, KE, omega_initial, omega_final, alpha_constant, t_max, system):
    t_ramp = abs(omega_final - omega_initial) / alpha_constant
    # Error 1: Sequential analytical solution (ramp step + hold step)
    if system == "linear":
        theta_ramp = 0.5 * (omega_initial + omega_final) * t_ramp
        theta_hold = omega_final * (t_max - t_ramp)
        theta_ana = theta_ramp + theta_hold
        rel_err = abs(theta[-1] - theta_ana) / abs(theta_ana) * 100
    else:
        rel_err = np.nan 
    # Error 2: Integrate power P separately on each step and convert error to %
    W = np.zeros_like(t)
    m1 = t <= t_ramp
    m2 = t >= t_ramp
    W[m1] = cumulative_trapezoid(P[m1], t[m1], initial=0)
    W[m2] = W[m1][-1] + cumulative_trapezoid(P[m2], t[m2], initial=0)
    delta_K = KE - KE[0]
    energy_err_pct = np.max(np.abs(W - delta_K)) / np.max(KE) * 100
    return rel_err, energy_err_pct
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
m, I = calc_moment_of_inertia(R, d, rho)
if system == "linear":
    t, theta, omega, alpha = solve_linear_profile(omega_initial, omega_final, alpha_constant, t_max, dt)
else:
    t, theta, omega, alpha = solve_exp_profile(omega_initial, omega_final, alpha_constant, t_max, dt)
rpm, jerk, at, ac, atotal, torque, KE, P = calc_physical_quantities(t, theta, omega, alpha, R, I)
# Call check_error and print results
rel_err, energy_err_pct = check_error(t, theta, P, KE, omega_initial, omega_final, alpha_constant, t_max, system)
print("rel_err (%):", rel_err)
print("energy_err_pct (%):", energy_err_pct)
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
plt.subplot(3,2,5)
plt.plot(t, torque)
plt.title("Torque vs Time")
plt.xlabel("Time(s)")
plt.ylabel("Torque(N.m)")
plt.subplot(3,2,6)
plt.plot(t, P)
plt.title("Power vs Time")
plt.xlabel("Time(s)")
plt.ylabel("Power(W)")
plt.tight_layout()
plt.show()