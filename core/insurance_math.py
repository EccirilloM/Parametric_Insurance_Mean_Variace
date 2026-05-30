import numpy as np
from config.settings import L, W0, LAMBDA_RATE, NU, GAMMA_P, BETA

def E_Yi():
    """E[Y_i] for censored Exp(nu) capped at L — eq. (A.1)."""
    return (1.0 - np.exp(-NU * L)) / NU

def E_Yi2():
    """E[Y_i^2] for censored Exp(nu) capped at L — eq. (A.2)."""
    return (2.0 / NU**2 
            - 2.0 * np.exp(-NU * L) / NU**2 
            - 2.0 * L * np.exp(-NU * L) / NU)

def E_excess(d):
    """E[(Y_i - d)_+] — eq. (A.4). Works for scalar or array d."""
    d = np.asarray(d, dtype=float)
    return (np.exp(-NU * d) - np.exp(-NU * L)) / NU

def G_func(d):
    """G(d) = E[min(Y_i, d)^2] — variance reduction term, eq. (A.7)."""
    d = np.asarray(d, dtype=float)
    return (2.0 / NU**2) * (1.0 - np.exp(-NU * d) * (1.0 + NU * d))

# Pre-compute constants
EYI = E_Yi()
EYI2 = E_Yi2()

def P_d(d, gamma_d, theta_d):
    return (1.0 + theta_d) * (LAMBDA_RATE * E_excess(d) + gamma_d)

def P_p(k, theta_p):
    return (1.0 + theta_p) * (LAMBDA_RATE * k + GAMMA_P)

def MV_d(d, gamma_d, theta_d):
    return (W0 - P_d(d, gamma_d, theta_d) - LAMBDA_RATE * EYI 
            + LAMBDA_RATE * E_excess(d) - BETA * LAMBDA_RATE * G_func(d))

def MV_p(k, theta_p):
    return (W0 - P_p(k, theta_p) - LAMBDA_RATE * EYI + LAMBDA_RATE * k 
            - BETA * LAMBDA_RATE * (EYI2 + k**2 - 2.0 * k * EYI))

def optimal_params(theta_d, theta_p):
    d_star = theta_d / (2.0 * BETA)
    k_star = EYI - theta_p / (2.0 * BETA)
    d_star = float(np.clip(d_star, 0.0, L))
    k_star = float(np.clip(k_star, 0.0, L))
    return d_star, k_star

def k_budget(gamma_d, d_star, theta_d, theta_p):
    budget = P_d(d_star, gamma_d, theta_d)
    k_implied = (budget / (1.0 + theta_p) - GAMMA_P) / LAMBDA_RATE
    return np.minimum(k_implied, L)

def find_crossing(x, f1, f2):
    """Linear interpolation for finding intersection precision"""
    diff = np.asarray(f1) - np.asarray(f2)
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return x[i] - diff[i] * (x[i+1] - x[i]) / (diff[i+1] - diff[i])