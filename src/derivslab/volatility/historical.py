# src/derivslab/volatility/historical.py

import numpy as np

from .base import VolatilityModel


class HistoricalVolatilityEstimator(VolatilityModel):
    def __init__(self, trading_days: int = 252):
        self.trading_days = trading_days

    def compute(self, returns: np.ndarray) -> float:
        return np.std(returns, ddof=1) * np.sqrt(self.trading_days)
