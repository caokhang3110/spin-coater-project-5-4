import numpy as np
import pandas as pd


def rpm_to_rads(rpm):
    return rpm * 2 * np.pi / 60


def calc_physics(cfg, t, theta, omega, alpha):
    R = cfg["R"]

    # Khối lượng và mô-men quán tính (I = 1/2 m R^2)
    m_wafer = cfg["density"] * np.pi * (R ** 2) * cfg["thickness"]
    I_wafer = 0.5 * m_wafer * (R ** 2)
    I_chuck = 0.5 * cfg["chuck_mass"] * (cfg["chuck_R"] ** 2)
    I_total = I_wafer + I_chuck

    # Tính các đại lượng trên mảng numpy 
    rpm = omega * 60 / (2 * np.pi)
    jerk = np.gradient(alpha, t)

    a_t_edge = np.abs(alpha) * R
    a_c_edge = (omega ** 2) * R
    a_total = R * np.sqrt(alpha ** 2 + omega ** 4)

    kinetic_energy = 0.5 * I_total * (omega ** 2)
    torque = I_total * alpha
    power = torque * omega

    df = pd.DataFrame({
        "time_t": t,
        "theta": theta,
        "omega": omega,
        "rpm": rpm,
        "alpha": alpha,
        "jerk": jerk,
        "a_t_edge": a_t_edge,
        "a_c_edge": a_c_edge,
        "a_total": a_total,
        "kinetic_energy": kinetic_energy,
        "torque": torque,
        "power": power
    })

    return df, I_total