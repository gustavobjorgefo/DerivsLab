# src/derivslab/volatility/garch.py

import numpy as np
from typing import Optional
from scipy.optimize import minimize
from .base import VolatilityModel

class GARCH11Volatility(VolatilityModel):
    """
    GARCH(1,1) volatility estimator.

    Follows the standard GARCH(1,1) model:
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters
    ----------
    omega : float
        Constant term.
    alpha : float
        Coefficient for lagged squared returns.
    beta : float
        Coefficient for lagged variance.
    trading_days : int, default=252
        Annualization factor for volatility.
    """

    def __init__(self, omega: float = 1e-6, alpha: float = 0.05, beta: float = 0.9, trading_days: int = 252):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.trading_days = trading_days

    def compute(self, returns: np.ndarray, sigma0: Optional[float] = None) -> float:
        n = len(returns)
        if n == 0:
            raise ValueError("Returns array is empty.")

        if sigma0 is None:
            sigma2 = np.var(returns)
        else:
            sigma2 = sigma0**2

        for t in range(n):
            epsilon2 = returns[t] ** 2
            sigma2 = self.omega + self.alpha * epsilon2 + self.beta * sigma2

        sigma_annual = np.sqrt(sigma2 * self.trading_days)
        return sigma_annual
    
    def calibrate(self, returns: np.ndarray):
        """
        Calibrate omega, alpha, beta using Maximum Likelihood Estimation.
        """
        def neg_log_likelihood(params):
            omega, alpha, beta = params
            n = len(returns)
            sigma2 = np.var(returns)
            ll = 0.0
            for t in range(n):
                sigma2 = omega + alpha * returns[t]**2 + beta * sigma2
                ll += np.log(sigma2) + (returns[t]**2)/sigma2
            return 0.5 * ll  # negative log-likelihood

        bounds = [(1e-12, None), (0, 1-1e-12), (0, 1-1e-12)]

        # constraint: alpha + beta < 1
        cons = ({'type': 'ineq', 'fun': lambda x: 0.9999 - (x[1] + x[2])})

        res = minimize(neg_log_likelihood, x0=[self.omega, self.alpha, self.beta],
                       bounds=bounds, constraints=cons)
        
        self.omega, self.alpha, self.beta = res.x
        return self.omega, self.alpha, self.beta