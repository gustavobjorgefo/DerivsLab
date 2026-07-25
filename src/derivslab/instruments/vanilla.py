"""Vanilla (call/put) option instrument.

European and American vanillas share this single class because their
payoff formula is identical regardless of exercise style or underlying.
What differs — exercise timing and the underlying's own dynamics (spot
vs. forward) — is fully captured by ``VanillaOptionContract.style`` and
``VanillaOptionContract.underlying_asset_class``. This class's only job
tied to that distinction is resolving the correct ``pricing_model_key``
from the pair of them, so the right pricer (closed-form or numerical,
spot-based or forward-based) can be registered without introducing a
parallel class hierarchy.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Final, Mapping

from derivslab.instruments.base import Instrument
from derivslab.instruments.contracts import (
    DayCountConvention,
    ExerciseStyle,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)

if TYPE_CHECKING:
    from derivslab.calendars import TradingCalendar

# Registry keys per (exercise style, underlying asset class). Style alone
# is not enough: an equity option is priced off spot with a carry term,
# while a future option is priced off the forward itself (Black-76), even
# though both share the same payoff and the same ExerciseStyle. Each key
# is registered in the PricingRouter against the pricer capable of
# handling that combination.
_PRICING_MODEL_KEYS: Final[dict[tuple[ExerciseStyle, UnderlyingAssetClass], str]] = {
    (ExerciseStyle.EUROPEAN, UnderlyingAssetClass.EQUITY): "bs_vanilla_european_equity",
    (ExerciseStyle.EUROPEAN, UnderlyingAssetClass.FUTURE): "black76_vanilla_european_future",
    (ExerciseStyle.AMERICAN, UnderlyingAssetClass.EQUITY): "binomial_vanilla_american_equity",
    (ExerciseStyle.AMERICAN, UnderlyingAssetClass.FUTURE): "binomial_vanilla_american_future",
}

# Denominator per calendar-day convention. BUS_252 is handled separately
# because it counts business days through a TradingCalendar instead.
_CALENDAR_DAYS_PER_YEAR: Final[dict[DayCountConvention, float]] = {
    DayCountConvention.ACT_360: 360.0,
    DayCountConvention.ACT_365: 365.0,
}

_BUSINESS_DAYS_PER_YEAR: Final[float] = 252.0


class VanillaOption(Instrument):
    """A vanilla call or put, European or American.

    Parameters
    ----------
    option_contract : VanillaOptionContract
        Immutable reference data for this option.
    """

    def __init__(self, option_contract: VanillaOptionContract) -> None:
        self._contract: VanillaOptionContract = option_contract

    # --- Instrument interface ---------------------------------------

    @property
    def contract(self) -> VanillaOptionContract:
        """Return the immutable reference data backing this instrument."""
        return self._contract

    @property
    def underlyings(self) -> list[str]:
        """Return the tickers this instrument's value depends on."""
        return [self._contract.underlying]

    @property
    def pricing_model_key(self) -> str:
        """Return the registry key identifying the pricer to use.

        Resolved from ``(style, underlying_asset_class)``: the payoff is
        style-agnostic, but exercise timing alone does not determine
        which engine can value this contract correctly — the underlying's
        dynamics (spot vs. forward) do too.
        """
        contract = self._contract
        return _PRICING_MODEL_KEYS[(contract.style, contract.underlying_asset_class)]

    def payoff(self, spots_at_expiry: Mapping[str, float]) -> float:
        """Compute the intrinsic value at the given underlying spot.

        Parameters
        ----------
        spots_at_expiry : Mapping[str, float]
            Must contain an entry for this option's underlying ticker.
        """
        spot = spots_at_expiry[self._contract.underlying]
        strike = self._contract.strike

        if self._contract.option_type is OptionType.CALL:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    def time_to_expiry(
        self,
        valuation_date: date,
        calendar: TradingCalendar | None = None,
    ) -> float:
        """Compute time to expiry in years under the contract's convention.

        BUS_252 counts business days through ``calendar`` and divides by
        252. ACT_360 and ACT_365 count calendar days directly and never
        touch ``calendar``.

        Parameters
        ----------
        valuation_date : date
            The date from which time to expiry is measured.
        calendar : TradingCalendar | None
            Required when the contract's convention is ``BUS_252``.
            Ignored otherwise.

        Raises
        ------
        ValueError
            If ``valuation_date`` is after the contract's expiry, or if
            the convention is ``BUS_252`` and no calendar was provided.
        """
        expiry = self._contract.expiry
        if valuation_date > expiry:
            raise ValueError(
                f"Valuation date {valuation_date} is past expiry {expiry} "
                f"for {self._contract.instrument_id}."
            )
        if valuation_date == expiry:
            return 0.0

        convention = self._contract.day_count_convention
        if convention is DayCountConvention.BUS_252:
            if calendar is None:
                raise ValueError(
                    f"{self._contract.instrument_id} uses BUS_252 day-count, "
                    f"which requires a TradingCalendar."
                )
            business_days = calendar.business_days_between(valuation_date, expiry)
            return business_days / _BUSINESS_DAYS_PER_YEAR

        calendar_days = (expiry - valuation_date).days
        return calendar_days / _CALENDAR_DAYS_PER_YEAR[convention]