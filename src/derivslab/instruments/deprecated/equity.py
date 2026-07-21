# src/derivslab/instruments/equity.py

from dataclasses import dataclass
from datetime import date

@dataclass
class EquityContract:
        """
        """

        instrument_id: str
        underlying: str
        ticker: str
        expiry: date
        instrument_type: str    # = "EquityContract"
        tick_size: float = 0.01
        contract_size: int = 100
        # qty: int
        # book: str