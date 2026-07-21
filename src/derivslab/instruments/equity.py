"""Equity instrument.

An equity is the trivial instrument in the system: its payoff is simply
its own spot price, it never expires, and it needs no pricing model
beyond reading the spot from ``MarketState``. It still implements the
full ``Instrument`` interface so that a ``Book`` can hold equities and
options side by side and aggregate greeks the same way for both — a
hedge position is just an ``EquityInstrument`` with quantity attached at
the ``Position`` level, not a special case.
"""

from __future__ import annotations

from datetime import date
from math import inf
from typing import TYPE_CHECKING, Final, Mapping

from derivslab.instruments.base import Instrument
from derivslab.instruments.contracts import EquityContract

if TYPE_CHECKING:
    from derivslab.calendars import TradingCalendar

# Registry key for the trivial spot pricer. Registered once in the
# PricingRouter and reused by every EquityInstrument.
EQUITY_PRICING_MODEL_KEY: Final[str] = "equity_spot"


class EquityInstrument(Instrument):
    """A cash equity, used both as an option's underlying and as a hedge.

    Parameters
    ----------
    equity_contract : EquityContract
        Immutable reference data for this equity.
    """

    def __init__(self, equity_contract: EquityContract) -> None:
        self._contract: EquityContract = equity_contract

    # --- Instrument interface ---------------------------------------

    @property
    def contract(self) -> EquityContract:
        """Return the immutable reference data backing this instrument."""
        return self._contract

    @property
    def underlyings(self) -> list[str]:
        """Return the tickers this instrument's value depends on."""
        return [self._contract.ticker]

    @property
    def pricing_model_key(self) -> str:
        """Return the registry key identifying the pricer to use."""
        return EQUITY_PRICING_MODEL_KEY

    def payoff(self, spots_at_expiry: Mapping[str, float]) -> float:
        """Return the equity's own spot price.

        Parameters
        ----------
        spots_at_expiry : Mapping[str, float]
            Must contain an entry for this equity's ticker.
        """
        return spots_at_expiry[self._contract.ticker]

    def time_to_expiry(
        self,
        valuation_date: date,
        calendar: TradingCalendar | None = None,
    ) -> float:
        """Return infinity — equities have no contractual maturity.

        ``calendar`` is accepted for interface parity with ``Instrument``
        but is never used: an equity's day-count convention is never
        consulted because there is no expiry to measure against.
        """
        return inf
