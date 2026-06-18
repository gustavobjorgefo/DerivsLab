# src/derivslab/volatility/ewma.py

from .base import VolatilityModel
import numpy as np


class EWMAVolatilityEstimator(VolatilityModel):
    def __init__(self, lam: float = 0.94, trading_days: int = 252):
        self.lam = lam
        self.trading_days = trading_days

    def compute(self, returns: np.ndarray) -> float:
        weights = np.array([(1 - self.lam) * (self.lam ** i) for i in range(len(returns))][::-1])
        weights /= weights.sum()
        variance = np.sum(weights * returns**2)
        return np.sqrt(variance * self.trading_days)