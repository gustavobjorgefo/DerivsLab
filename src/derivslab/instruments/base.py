"""Behavioral interface shared by every priceable instrument.

``Instrument`` wraps an immutable ``InstrumentContract`` (see
``contracts.py``) and adds the three pieces of behavior the rest of the
system depends on: what it pays off, how far it is from expiry, and which
pricing engine should be used to value it. Nothing else belongs here —
market data (spot, vol, rates) lives in ``MarketState``, holiday
calendars live in ``derivslab.calendars``, and the pricing formulas
themselves live in the ``pricing`` package, keyed off
``pricing_model_key``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from derivslab.calendars import TradingCalendar
    from derivslab.instruments.contracts import InstrumentContract


class Instrument(ABC):
    """Minimal contract every priceable instrument must satisfy.

    A ``PricingRouter`` never inspects an instrument's concrete type — it
    only reads ``pricing_model_key`` to look up the registered pricer, and
    passes the instrument itself so the pricer can read ``contract`` and
    ``underlyings``. Adding a new product to the system means writing one
    class that implements this interface and registering a matching
    pricer; nothing upstream (book, simulation engine, strategy) changes.
    """

    @property
    @abstractmethod
    def contract(self) -> InstrumentContract:
        """Return the immutable reference data backing this instrument."""
        ...

    @property
    @abstractmethod
    def underlyings(self) -> list[str]:
        """Return the tickers this instrument's value depends on."""
        ...

    @property
    @abstractmethod
    def pricing_model_key(self) -> str:
        """Return the registry key identifying the pricer to use."""
        ...

    @abstractmethod
    def payoff(self, spots_at_expiry: Mapping[str, float]) -> float:
        """Compute the payoff given underlying spot prices at expiry.

        Parameters
        ----------
        spots_at_expiry : Mapping[str, float]
            Spot price per underlying ticker, keyed the same way as
            ``underlyings``. A mapping is used rather than a scalar so
            that single-underlying and basket instruments share the same
            signature.

        Returns
        -------
        float
            The instrument's payoff in its settlement currency.
        """
        ...

    @abstractmethod
    def time_to_expiry(
        self,
        valuation_date: date,
        calendar: TradingCalendar | None = None,
    ) -> float:
        """Compute the time to expiry, in years, from a valuation date.

        Parameters
        ----------
        valuation_date : date
            The date from which time to expiry is measured.
        calendar : TradingCalendar | None
            Business-day calendar to use when the contract's day-count
            convention requires one. Ignored by instruments whose
            convention does not need business-day counting.

        Returns
        -------
        float
            Time to expiry in years. May be ``float('inf')`` for
            instruments with no contractual maturity.

        Raises
        ------
        ValueError
            If ``valuation_date`` is after the instrument's expiry, or if
            the contract's day-count convention requires a calendar and
            none was provided.
        """
        ...

    # --- debugging -----------------------------------------------------
 
    def __repr__(self) -> str:
        """Return a debug-friendly representation delegating to the contract.
 
        Every concrete ``Instrument`` already carries a self-describing,
        dataclass-generated ``contract`` — reusing it here avoids every
        subclass having to redefine ``__repr__`` on its own.
        """
        return f"{type(self).__name__}({self.contract!r})"