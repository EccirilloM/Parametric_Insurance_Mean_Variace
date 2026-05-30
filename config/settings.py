# FIXED MODEL PARAMETERS
L = 1_000_000               # shed value [EUR]
W0 = 0.2 * L                # initial wealth
LAMBDA_RATE = 1.0 / 40      # Poisson hazard rate
NU = 1.0 / 700_000          # severity parameter for censored Exp [1/EUR]
GAMMA_P = 0.0               # parametric fixed cost
BETA = 1.0 / W0             # risk-aversion coefficient