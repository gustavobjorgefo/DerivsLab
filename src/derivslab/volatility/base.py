# src/derivslab/volatility/base.py

from abc import ABC, abstractmethod
import numpy as np

class VolatilityModel(ABC):
    @abstractmethod
    def compute(self, returns: np.ndarray) -> float:
        """Compute annualized volatility"""
        pass