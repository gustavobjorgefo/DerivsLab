
import math
import pytest
from datetime import datetime

from derivslab.pricing.pricing_models import BlackModel, BlackScholesModel, BlackScholesMertonModel
from derivslab.instruments.option import VanillaOption


if __name__ == '__main__':

    strike_price = 100.0
    expiration_date = datetime(2026, 1, 16)

    option_call  = VanillaOption(
        option_type='call',
        strike_price=strike_price,
        expiration_date=expiration_date,
        underlying='ABCD'
    )

    option_put  = VanillaOption(
        option_type='put',
        strike_price=strike_price,
        expiration_date=expiration_date,
        underlying='ABCD'
    )

    print("Call payoff @120:", option_call.payoff(120))
    print("Call payoff @80:", option_call.payoff(80))
    print("Put payoff @120:", option_put.payoff(120))
    print("Put payoff @80:", option_put.payoff(80))
    
    
    maturity = (expiration_date - datetime(2025, 7, 18)).days / 365
    spot_price = 105.0
    risk_free_rate = 0.05
    volatility = 0.25

    forward = spot_price * math.exp(risk_free_rate * maturity)
    discount_factor = math.exp(-risk_free_rate * maturity)

    black_model = BlackModel()
    
    print(f'Call price: {black_model.price(
        option_type="call",
        forward=forward,
        strike=strike_price,
        maturity=maturity,
        volatility=volatility,
        discount_factor=discount_factor
    )}')

    print(f'Put price: {black_model.price(
        option_type="put",
        forward=forward,
        strike=strike_price,
        maturity=maturity,
        volatility=volatility,
        discount_factor=discount_factor
    )}')

    call_price = option_call.price(
        model=black_model,
        forward=forward,
        maturity=maturity,
        volatility=volatility,
        discount_factor=discount_factor
    )

    put_price = option_put.price(
        model=black_model,
        forward=forward,
        maturity=maturity,
        volatility=volatility,
        discount_factor=discount_factor
    )
    print(f"Integration test (via option.price) call price: {call_price:.4f}")
    print(f"Integration test (via option.price) put price: {put_price:.4f}")