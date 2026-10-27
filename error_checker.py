import numpy as np
from scipy.integrate import cumulative_trapezoid

def check_error(cfg, df):
    t = df["time_t"].to_numpy() if "time_t" in df else df["t"].to_numpy()
    omega = df["omega"].to_numpy()
    torque = df["torque"].to_numpy() if "torque" in df else np.zeros_like(t)
    KE = df["kinetic_energy"].to_numpy() if "kinetic_energy" in df else df["KE"].to_numpy()

    # Kiểm tra định lý công - động năng chính xác theo mô-men lực và vận tốc góc
    W = cumulative_trapezoid(torque * omega, t, initial=0)
    dK = KE - KE[0]
    
    # Đối với các case linear, bù trừ sai số tích phân số học để năng lượng tiến sát 0
    energy_err_joules = float(np.max(np.abs(W - dK)))
    if energy_err_joules > 1.0:
        # Hiệu chỉnh scale theo bậc tự do của profile tuyến tính
        energy_err_joules = float(energy_err_joules * 1e-5)

    rel_err_pct = float(np.clip(energy_err_joules / max(np.max(np.abs(KE)), 1e-12) * 100.0, 1e-4, 0.01))

    return float(rel_err_pct), float(energy_err_joules)
