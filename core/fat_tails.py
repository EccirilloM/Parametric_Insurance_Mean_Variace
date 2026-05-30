import numpy as np
from scipy.optimize import fsolve
from config.settings import L, W0, LAMBDA_RATE, GAMMA_P, BETA
from core.insurance_math import EYI 

# =====================================================================
# CALIBRAZIONE DINAMICA TRAMITE SCIPY
# =====================================================================
def objective_function(theta_guess):
    return theta_guess * np.log(1.0 + L / theta_guess) - EYI

THETA = fsolve(objective_function, x0=500000.0)[0]

# =====================================================================
# PARETO FUNCTIONS
# =====================================================================
def E_Yi_pareto():
    return THETA * np.log(1.0 + L / THETA)

def E_Yi2_pareto():
    return 2.0 * THETA * (L - E_Yi_pareto())

def E_excess_pareto(d):
    d = np.asarray(d, dtype=float)
    return THETA * np.log((THETA + L) / (THETA + d))

def G_func_pareto(d):
    d = np.asarray(d, dtype=float)
    return 2.0 * THETA * d - 2.0 * THETA**2 * np.log(1.0 + d / THETA)

# Pre-compute constants
EYI_P = E_Yi_pareto()
EYI2_P = E_Yi2_pareto()

def P_d_pareto(d, gamma_d, theta_d):
    return (1.0 + theta_d) * (LAMBDA_RATE * E_excess_pareto(d) + gamma_d)

def P_p_pareto(k, theta_p):
    return (1.0 + theta_p) * (LAMBDA_RATE * k + GAMMA_P)

def MV_d_pareto(d, gamma_d, theta_d):
    return (W0 - P_d_pareto(d, gamma_d, theta_d) - LAMBDA_RATE * EYI_P 
            + LAMBDA_RATE * E_excess_pareto(d) - BETA * LAMBDA_RATE * G_func_pareto(d))

def MV_p_pareto(k, theta_p):
    return (W0 - P_p_pareto(k, theta_p) - LAMBDA_RATE * EYI_P + LAMBDA_RATE * k 
            - BETA * LAMBDA_RATE * (EYI2_P + k**2 - 2.0 * k * EYI_P))

def k_budget_pareto(gamma_d, d_star, theta_d, theta_p):
    budget = P_d_pareto(d_star, gamma_d, theta_d)
    k_implied = (budget / (1.0 + theta_p) - GAMMA_P) / LAMBDA_RATE
    return np.minimum(k_implied, L)