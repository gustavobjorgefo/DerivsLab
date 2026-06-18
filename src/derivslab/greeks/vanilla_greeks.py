# src/derivslab/greeks/vanilla_greeks.py

import numpy as np
from scipy.stats import norm

def delta_bs(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")