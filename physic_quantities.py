import numpy as np
# Convert rotational speed from revolutions per minute (RPM) to radians per second (rad/s)
def rpm_to_rads(rpm):
    rads = rpm*2*np.pi/60
    return rads
# Convert angular velocity from radians per second (rad/s) to revolutions per minute (RPM).
def rads_to_rpm(rads):
    rpm = rads*60/ (2*np.pi)
    return rpm 
# Calculate the mass and polar moment of inertia for a uniform solid cylinder or disk.
def calc_moment_of_inertia(R, d, rho):
    m = rho*np.pi*d*R**2
    I = 0.5*m*R**2
    return m, I
# Calculate other physical quantities
def calc_physical_quantities(t, theta, omega, alpha, R, I):
    rpm = omega*60 / (2*np.pi)
    jerk = np.gradient(alpha, t)
    at = alpha*R
    ac = omega**2*R
    atotal = np.sqrt(at**2 + ac**2)
    torque = I*alpha
    KE = 0.5*I*omega**2
    P = torque*omega
    return rpm, jerk, at, ac, atotal, torque, KE, P
