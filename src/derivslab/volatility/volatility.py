# src/derivslab/volatility/volatility.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Literal, Callable
from scipy.stats import norm


class VolatilityEstimator:
    """
    Methods to estimate and invert volatility.
    """

    @staticmethod
    def historical_volatility(returns: np.ndarray, trading_days: int = 252) -> float:
        """
        Computes annualized historical volatility.
        """
        return np.std(returns, ddof=1) * np.sqrt(trading_days)
    

    @staticmethod
    def ewma_volatility(returns: np.ndarray, lam: float = 0.94, trading_days: int = 252) -> float:
        """
        Exponentially Weighted Moving Average (EWMA) volatility.
        """
        weights = np.array([(1 - lam) * (lam ** i) for i in range(len(returns))][::-1])
        weights /= weights.sum()
        variance = np.sum(weights * returns**2)
        return np.sqrt(variance * trading_days)
    

    @staticmethod
    def garch_volatility():
        """
        Placeholder for a GARCH model volatility estimation.
        (to be implemented with arch or statsmodels).
        """
        raise NotImplementedError("GARCH model not implemented yet.")
    

    @staticmethod
    def implied_vol(
        market_price: float,
        model_func: Callable[[float], float],
        initial_vol: float = 0.2,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Estimates implied volatility via Newton-Raphson root-finding.
        
        Parameters
        ----------
        market_price : float
            Observed option market price.
        model_func : Callable[[float], float]
            A pricing function that takes volatility as argument 
            and returns the option price under the chosen model.
            Example: lambda vol: BlackScholes(..., vol=vol).price()
        initial_vol : float
            Initial guess for volatility.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations allowed.
        """
        sigma = initial_vol
        for _ in range(max_iter):
            price = model_func(sigma)

            # Numerical derivative (vega) by finite difference
            eps = 1e-5
            price_up = model_func(sigma + eps)
            vega = (price_up - price) / eps

            diff = price - market_price

            if abs(diff) < tol:
                return sigma
            if abs(vega) < 1e-8:  # avoid division by zero
                break

            sigma -= diff / vega

        return np.nan  # did not converge