# src/derivslab/volatility/implied.py

import numpy as np
from typing import Callable
from .base import VolatilityModel


class ImpliedVolatilityEstimator(VolatilityModel):
    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter

    def compute(self, market_price: float, model_func: Callable[[float], float], initial_vol: float = 0.2) -> float:
        sigma = initial_vol
        for _ in range(self.max_iter):
            price = model_func(sigma)
            eps = 1e-5
            price_up = model_func(sigma + eps)
            vega = (price_up - price) / eps
            diff = price - market_price
            if abs(diff) < self.tol:
                return sigma
            if abs(vega) < 1e-8:
                break
            sigma -= diff / vega
        return np.nan