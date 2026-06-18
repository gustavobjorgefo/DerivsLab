# src/derivslab/simulation/path_generator.py

import numpy as np
import pandas as pd

class GBMPathGenerator:
    """
    Generates paths for an underlying asset using Geometric Brownian Motion.
    """

    def __init__(self, S0: float, mu: float, sigma: float, T: float, steps: int, n_paths: int, seed: int = None):
            self.S0 = S0
            self.mu = mu
            self.sigma = sigma
            self.T = T
            self.steps = steps
            self.n_paths = n_paths
            self.dt = T / steps
            self.rng = np.random.default_rng(seed)

    def generate_paths(self) -> np.ndarray:
        """
        Returns:
            paths: np.ndarray of shape (n_paths, steps+1)
        """
        paths = np.zeros((self.n_paths, self.steps + 1))
        paths[:, 0] = self.S0

        for t in range(1, self.steps + 1):
            Z = self.rng.standard_normal(self.n_paths)
            paths[:, t] = paths[:, t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)

        return paths
    
class CorrelatedGBMPathGenerator:
    """
    Generates correlated GBM paths for two assets.
    """

    def __init__(self,
                 S0_1: float, S0_2: float,
                 mu1: float, mu2: float,
                 sigma1: float, sigma2: float,
                 rho: float,
                 T: float, steps: int, n_paths: int,
                 seed: int = None):
        
        self.S0_1 = S0_1
        self.S0_2 = S0_2
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.T = T
        self.steps = steps
        self.n_paths = n_paths
        self.dt = T / steps
        self.rng = np.random.default_rng(seed)

    def generate_paths(self):
        paths1 = np.zeros((self.n_paths, self.steps + 1))
        paths2 = np.zeros((self.n_paths, self.steps + 1))
        paths1[:, 0] = self.S0_1
        paths2[:, 0] = self.S0_2

        for t in range(1, self.steps + 1):
            Z1 = self.rng.standard_normal(self.n_paths)
            eps = self.rng.standard_normal(self.n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * eps

            paths1[:, t] = paths1[:, t-1] * np.exp(
                (self.mu1 - 0.5 * self.sigma1**2) * self.dt +
                self.sigma1 * np.sqrt(self.dt) * Z1
            )

            paths2[:, t] = paths2[:, t-1] * np.exp(
                (self.mu2 - 0.5 * self.sigma2**2) * self.dt +
                self.sigma2 * np.sqrt(self.dt) * Z2
            )

        return paths1, paths2
    

class MultiAssetGBMPathGenerator:
    """
    Generates correlated GBM paths for N assets using a covariance matrix.
    """
    
    def __init__(self,
                S0: list[float],
                mu: list[float],
                sigma: list[float],
                cov_matrix: np.ndarray,
                T: float,
                steps: int,
                n_paths: int,
                seed: int = None):
        """
        Args:
        S0: list of initial prices for each asset
        mu: list of drifts (expected returns)
        sigma: list of volatilities
        cov_matrix: NxN covariance matrix of Brownian increments
        T: total simulation time (years)
        steps: number of time steps
        n_paths: number of Monte Carlo paths
        seed: random seed (optional)
        """

        self.S0 = np.array(S0)
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.cov = np.array(cov_matrix)
        self.T = T
        self.steps = steps
        self.n_paths = n_paths
        self.dt = T / steps
        self.rng = np.random.default_rng(seed)

        # Check dimensions
        n_assets = len(S0)
        if self.cov.shape != (n_assets, n_assets):
            raise ValueError("cov_matrix must be of shape (n_assets, n_assets)")
        
        # Precompute Cholesky factor for correlated shocks
        self.L = np.linalg.cholesky(self.cov)

    def generate_paths(self) -> np.ndarray:
        """
        Returns:
        paths: np.ndarray of shape (n_assets, n_paths, steps+1)
        """
        n_assets = len(self.S0)
        paths = np.zeros((n_assets, self.n_paths, self.steps + 1))
        paths[:, :, 0] = self.S0[:, None]

        drift = (self.mu - 0.5 * self.sigma ** 2)[:, None] * self.dt
        diffusion_scale = self.sigma[:, None] * np.sqrt(self.dt)

        for t in range(1, self.steps + 1):
            # Generate independent standard normals
            Z = self.rng.standard_normal((n_assets, self.n_paths))
            # Correlate them via Cholesky factor
            correlated_Z = self.L @ Z
            # GBM update (vectorized across assets and paths)
            paths[:, :, t] = paths[:, :, t - 1] * np.exp(
                drift + diffusion_scale * correlated_Z
            )

        return paths

    
if __name__ == '__main__':
     gbm_generator = GBMPathGenerator(
          S0=100.0,
          mu=0.05,
          sigma=0.25,
          T=0.50,
          steps=126,
          n_paths=10000,
          seed=None
     )

     paths = gbm_generator.generate_paths()
     print(pd.DataFrame(paths))