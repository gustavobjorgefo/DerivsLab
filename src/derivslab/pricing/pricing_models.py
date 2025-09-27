# src/derivslab/pricing/pricing_models.py

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm


@dataclass
class BlackModel:
    """
    Black model (1976), used for pricing options on futures or forward contracts.
    
    Typical usage:
        - Options on futures (commodities, rates, indices).
    
    Parameters
    ----------
    option_type : 'call' or 'put'
    forward : Forward price (F)
    strike : Strike price (K)
    maturity : Time to maturity in years (T)
    volatility : Volatility of forward price (σ)
    discount_factor : Discount factor (DF = e^(-rT))
    """

    option_type: Literal["call", "put"]
    forward: float
    strike: float
    maturity: float
    volatility: float
    discount_factor: float

    def price(self) -> float:
        """
        Compute the option price using Black's model formula.
        """

        d1 = (math.log(self.forward / self.strike) + 0.5 * self.volatility**2 * self.maturity) / (
            self.volatility * math.sqrt(self.maturity)
        )
        d2 = d1 - self.volatility * math.sqrt(self.maturity)

        if self.option_type == "call":
            return self.discount_factor * (self.forward * norm.cdf(d1) - self.strike * norm.cdf(d2))
        
        elif self.option_type == "put":
            return self.discount_factor * (self.strike * norm.cdf(-d2) - self.forward * norm.cdf(-d1))
        
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")
        

@dataclass
class BlackScholesModel:
    """
    Black-Scholes model (1973), for European options on non-dividend-paying stocks.
    
    Typical usage:
        - Equity options without dividends.
    
    Parameters
    ----------
    option_type : 'call' or 'put'
    spot : Spot price of the underlying asset (S)
    strike : Strike price (K)
    maturity : Time to maturity in years (T)
    volatility : Volatility of asset price (σ)
    risk_free_rate : Risk-free rate (r)
    """

    option_type: Literal["call", "put"]
    spot: float
    strike: float
    maturity: float
    volatility: float
    risk_free_rate: float

    def price(self) -> float:
        """
        Compute the option price using Black-Scholes formula.
        """

        d1 = (math.log(self.spot / self.strike) + (self.risk_free_rate + 0.5 * self.volatility**2) * self.maturity) / (
            self.volatility * math.sqrt(self.maturity)
        )

        d2 = d1 - self.volatility * math.sqrt(self.maturity)

        if self.option_type == "call":
            return self.spot * norm.cdf(d1) - self.strike * math.exp(-self.risk_free_rate * self.maturity) * norm.cdf(d2)
        
        elif self.option_type == "put":
            return self.strike * math.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-d2) - self.spot * norm.cdf(-d1)
        
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")
        

@dataclass
class BlackScholesMertonModel:
    """
    Black-Scholes-Merton model (1973), for European options on dividend-paying assets.
    
    Typical usage:
        - Equity options with continuous dividend yield.
        - FX options (where dividend yield = foreign risk-free rate).
    
    Parameters
    ----------
    option_type : 'call' or 'put'
    spot : Spot price of the underlying asset (S)
    strike : Strike price (K)
    maturity : Time to maturity in years (T)
    volatility : Volatility of asset price (σ)
    risk_free_rate : Domestic risk-free rate (r)
    dividend_yield : Continuous dividend yield (q)
    """

    option_type: Literal["call", "put"]
    spot: float
    strike: float
    maturity: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float

    def price(self) -> float:
        """
        Compute the option price using Black-Scholes-Merton formula.
        """

        d1 = (math.log(self.spot / self.strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility**2) * self.maturity) / (
            self.volatility * math.sqrt(self.maturity)
        )

        d2 = d1 - self.volatility * math.sqrt(self.maturity)

        if self.option_type == "call":
            return self.spot * math.exp(-self.dividend_yield * self.maturity) * norm.cdf(d1) - self.strike * math.exp(
                -self.risk_free_rate * self.maturity
            ) * norm.cdf(d2)
        
        elif self.option_type == "put":
            return self.strike * math.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-d2) - self.spot * math.exp(
                -self.dividend_yield * self.maturity
            ) * norm.cdf(-d1)
        
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")
        

