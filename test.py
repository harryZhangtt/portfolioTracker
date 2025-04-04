import numpy as np
import sympy as sp

# Define the symbols for the coefficients and for ln(L) and ln(K)
beta1, beta2, beta3, beta4, beta5, ln_L, ln_K = sp.symbols('beta1 beta2 beta3 beta4 beta5 ln_L ln_K')

# Define the elasticity expressions:
# Elasticity of Y with respect to L: E_L = beta1 + 2*beta2*ln(L) + beta5*ln(K)
E_L_expr = beta1 + 2 * beta2 * ln_L + beta5 * ln_K

# Elasticity of Y with respect to K: E_K = beta3 + 2*beta4*ln(K) + beta5*ln(L)
E_K_expr = beta3 + 2 * beta4 * ln_K + beta5 * ln_L

# Use the provided values for ln(L) and ln(K)
ln_L_val = np.log(1712)
ln_K_val = np.log(318)

# Substitute the numeric values into the expressions
E_L_sub = E_L_expr.subs({ln_L: ln_L_val, ln_K: ln_K_val})
E_K_sub = E_K_expr.subs({ln_L: ln_L_val, ln_K: ln_K_val})

print("Elasticity of Y with respect to L:")
sp.pprint(E_L_sub)

print("\nElasticity of Y with respect to K:")
sp.pprint(E_K_sub)
