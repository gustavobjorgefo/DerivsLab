# src/derivslab/instruments/option.py

# VanillaOption, ExoticOption (barrier, asian, etc.)

from dataclasses import dataclass
from datetime import date
from typing import Literal

@dataclass
class OptionContract:
        """
        """
        instrument_id: str
        underlying: str
        option_type: str        # call / put
        style: str              # european / american
        strike: float
        expiry: date
        instrument_type: str    # = "EquityVanilla"
        tick_size: float = 0.01
        contract_size: int = 100
        # qty: int
        # book: str

@dataclass
class BarrierOptionContract(OptionContract):
    """
    """
    instrument_id: str
    underlying: str
    option_type: str        # call / put
    style: str              # european / american
    expiry: date
    strike: float
    barrier: float
    barrier_type: Literal['upIn', 'upOut', 'downIn', 'downOut']
    monitoring: Literal['continuous', 'discrete']
    instrument_type: str    # = "EquityVanilla"
    tick_size: float = 0.01
    contract_size: int = 100       
    # qty: int
    # book: str