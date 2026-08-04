
"""DI1 futures instrument (B3 one-day interbank rate future).

Unlike an option, a DI1 future has no path-dependent terminal payoff: its
settlement value converges deterministically to
``DI_FUTURE_SETTLEMENT_VALUE`` at expiry, the same way a zero-coupon bond
converges to par. All of the interest-rate information a DI1 future
carries lives in its marked-to-market PU *before* expiry, not in its
payoff, which is why ``payoff`` here is a constant. Building a yield
curve from a set of DI1 futures is a pricing/curve concern, consuming
each instrument's quoted rate at a valuation date — never its payoff.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Final, Mapping

from derivslab.instruments.base import Instrument
from derivslab.instruments.contracts import DI_FUTURE_SETTLEMENT_VALUE, DIFutureContract

if TYPE_CHECKING:
    from derivslab.calendars import TradingCalendar

# Every DI1 future references the same index — the accumulated Taxa DI —
# so it has no per-instance underlying, unlike a single-name option.
DI_RATE_UNDERLYING: Final[str] = "DI"

# Registered once in the PricingRouter against the discount-factor pricer
# that converts a quoted rate and BUS_252 time-to-expiry into a PU.
DI_FUTURE_PRICING_MODEL_KEY: Final[str] = "di_future_discount"

_BUSINESS_DAYS_PER_YEAR: Final[float] = 252.0


class DIFuture(Instrument):
    """A B3 DI1 futures contract.

    Parameters
    ----------
    future_contract : DIFutureContract
        Immutable reference data for this future.
    """

    def __init__(self, future_contract: DIFutureContract) -> None:
        self._contract: DIFutureContract = future_contract

    # --- Instrument interface ---------------------------------------

    @property
    def contract(self) -> DIFutureContract:
        """Return the immutable reference data backing this instrument."""
        return self._contract

    @property
    def underlyings(self) -> list[str]:
        """Return the tickers this instrument's value depends on."""
        return [DI_RATE_UNDERLYING]

    @property
    def pricing_model_key(self) -> str:
        """Return the registry key identifying the pricer to use."""
        return DI_FUTURE_PRICING_MODEL_KEY

    def payoff(self, spots_at_expiry: Mapping[str, float]) -> float:
        """Return the deterministic settlement PU at expiry.

        A DI1 future's PU converges to ``DI_FUTURE_SETTLEMENT_VALUE`` at
        expiry by contract design, regardless of the path the Taxa DI
        took to get there. ``spots_at_expiry`` is accepted for interface
        parity with ``Instrument`` but never read.
        """
        return DI_FUTURE_SETTLEMENT_VALUE

    def time_to_expiry(
        self,
        valuation_date: date,
        calendar: TradingCalendar | None = None,
    ) -> float:
        """Compute time to expiry in years, counting business days over 252.

        DI1 is always quoted on a BUS_252 basis, so unlike
        ``VanillaOption`` there is no other day-count branch to resolve.

        Raises
        ------
        ValueError
            If ``valuation_date`` is after the contract's expiry, or if
            no calendar was provided.
        """
        expiry = self._contract.expiry
        if valuation_date > expiry:
            raise ValueError(
                f"Valuation date {valuation_date} is past expiry {expiry} "
                f"for {self._contract.instrument_id}."
            )
        if valuation_date == expiry:
            return 0.0
        if calendar is None:
            raise ValueError(
                f"{self._contract.instrument_id} uses BUS_252 day-count, "
                f"which requires a TradingCalendar."
            )
        business_days = calendar.business_days_between(valuation_date, expiry)
        return business_days / _BUSINESS_DAYS_PER_YEAR