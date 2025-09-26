# option.py
# 
# VanillaOption, ExoticOption (barrier, asian, etc.)

from datetime import date, datetime
from typing import Union, Literal


class VanillaOption:

    def __init__(
        self,
        option_type: str,
        strike_price: float,
        expiration_date: Union[date, datetime],
        underlying: str,
        option_style: str = "european",
        tick_size: float = 0.01,
        contract_size: int = 100
    ):
        
        """
        A basic vanilla option (call or put).

        Parameters
        ----------
        option_type : str
            'call' or 'put'

        strike_price : float
            Exercise price of the option

        expiration_date : datetime.date
            Expiration date of the option

        underlying : str
            The underlying asset symbol (e.g., 'AAPL', 'PETR4')

        option_style : str, default='european'
            'european' or 'american'

        tick_size : float, default=0.01
            Minimum price increment

        contract_size : int, default=100
            Number of units per contract
        """

        self.option_type = option_type.lower()
        self.strike_price = strike_price
        self.expiration_date = expiration_date
        self.underlying = underlying
        self.option_style = option_style.lower()
        self.tick_size = tick_size
        self.contract_size = contract_size

    def __repr__(self):
        return (
            f"VanillaOption("
            f"type={self.option_type}, "
            f"strike_price={self.strike_price}, "
            f"expiration_date={self.expiration_date}, "
            f"underlying={self.underlying}, "
            f"option_style={self.option_style}, "
            f"tick_size={self.tick_size}, "
            f"contract_size={self.contract_size}"
            f")"
        )


from datetime import date
from typing import Literal

class BarrierOption(VanillaOption):
    
    def __init__(
        self,
        option_type: str,
        strike_price: float,
        expiration_date: Union[date, datetime],
        underlying: str,
        barrier: float,
        barrier_type: Literal['up-and-in', 'up-and-out', 'down-and-in', 'down-and-out'],
        monitoring: Literal['continuous', 'discrete'] = 'continuous',
        option_style: str = "european",
        tick_size: float = 0.01,
        contract_size: int = 100
    ):
        
        """
        A barrier option (exotic), inheriting from VanillaOption.

        Parameters
        ----------
        option_type : str
            'call' or 'put'

        strike_price : float
            Strike price of the option

        expiration_date : date
            Expiration date of the option

        underlying : str
            Symbol or identifier of the underlying asset

        barrier : float
            Barrier level for the option

        barrier_type : str
            One of:
                'up-and-in'   - activated if underlying rises above barrier
                'up-and-out'  - deactivated if underlying rises above barrier
                'down-and-in' - activated if underlying falls below barrier
                'down-and-out'- deactivated if underlying falls below barrier

        monitoring : str, optional
            'continuous' - barrier monitored continuously (default)
            'discrete'   - barrier monitored at discrete intervals (e.g., daily)

        option_style : str, optional
            'european' or 'american' (default = 'european')

        tick_size : float, optional
            Minimum price increment (default = 0.01)

        contract_size : int, optional
            Number of units per contract (default = 100)
        """

        # Initialize the vanilla option part
        super().__init__(
            option_type=option_type,
            strike_price=strike_price,
            expiration_date=expiration_date,
            underlying=underlying,
            option_style=option_style,
            tick_size=tick_size,
            contract_size=contract_size
        )

        # Barrier-specific attributes
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.monitoring = monitoring

    def __repr__(self):
        return (
            f"BarrierOption(type={self.option_type}, strike_price={self.strike_price}, "
            f"expiration_date={self.expiration_date}, underlying={self.underlying}, "
            f"barrier={self.barrier}, barrier_type={self.barrier_type}, "
            f"monitoring={self.monitoring}, option_style={self.option_style}, "
            f"tick_size={self.tick_size}, contract_size={self.contract_size})"
        )
