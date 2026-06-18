# src/derivslab/volatility/svi.py

import numpy as np
from scipy.optimize import minimize
from .base import VolatilityModel


class SVIModel(VolatilityModel):
    """
    SVI (Stochastic Volatility Inspired) model for implied volatility smiles.
    Based on Gatheral (2004).
    """

    def __init__(self, T: float, params=None):
        self.T = T
        self.params = params or {"a": 0.01, "b": 0.1, "rho": 0.0, "m": 0.0, "sigma": 0.1}

    def total_variance(self, k: np.ndarray, params=None):
        """Compute total implied variance w(k)"""
        p = params or self.params
        a, b, rho, m, sigma = p.values()
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def implied_vol(self, k: np.ndarray, params=None):
        """Return implied vol smile σ_imp(k)"""
        w = self.total_variance(k, params)
        return np.sqrt(w / self.T)

    def calibrate(self, strikes, forwards, maturities, market_ivs):
        """
        Calibrate SVI parameters via least squares.
        strikes, forwards, maturities, market_ivs -> arrays of same length
        """
        k = np.log(strikes / forwards)
        w_mkt = (market_ivs ** 2) * maturities

        def objective(x):
            p = {"a": x[0], "b": x[1], "rho": x[2], "m": x[3], "sigma": x[4]}
            w_model = self.total_variance(k, p)
            return np.mean((w_model - w_mkt)**2)

        x0 = list(self.params.values())
        bounds = [(1e-6, None), (1e-6, 2.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 2.0)]
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        if res.success:
            self.params = dict(zip(self.params.keys(), res.x))
        else:
            raise RuntimeError(f"SVI calibration failed: {res.message}")

        return self.params

    def compute(self, strikes, forward):
        """Return the calibrated implied vols σ(K)"""
        k = np.log(strikes / forward)
        return self.implied_vol(k)