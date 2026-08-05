"""Tests for derivslab.instruments.future.DIFuture."""

from __future__ import annotations

from datetime import date

import pytest

from derivslab.calendars import TradingCalendar
from derivslab.instruments.contracts import DI_FUTURE_SETTLEMENT_VALUE
from derivslab.instruments.future import DI_FUTURE_PRICING_MODEL_KEY, DI_RATE_UNDERLYING, DIFuture


class TestIdentity:
    """Tests for the parts of the interface unrelated to payoff or time."""

    def test_contract_returns_the_wrapped_contract(
        self, di_future: DIFuture, di_future_contract
    ) -> None:
        assert di_future.contract is di_future_contract

    def test_underlyings_is_the_di_rate(self, di_future: DIFuture) -> None:
        assert di_future.underlyings == [DI_RATE_UNDERLYING]

    def test_pricing_model_key_is_di_future_discount(self, di_future: DIFuture) -> None:
        assert di_future.pricing_model_key == DI_FUTURE_PRICING_MODEL_KEY


class TestPayoff:
    """Tests for DIFuture.payoff."""

    def test_payoff_is_always_the_settlement_value(self, di_future: DIFuture) -> None:
        assert di_future.payoff({}) == DI_FUTURE_SETTLEMENT_VALUE

    def test_payoff_ignores_spots_at_expiry(self, di_future: DIFuture) -> None:
        # The negotiated/realized rate never enters the payoff — it is
        # absorbed entirely by the daily adjustment, not the settlement PU.
        assert di_future.payoff({"DI": 999.0}) == DI_FUTURE_SETTLEMENT_VALUE


class TestTimeToExpiry:
    """Tests for DIFuture.time_to_expiry."""

    def test_without_calendar_raises(self, di_future: DIFuture, valuation_date: date) -> None:
        with pytest.raises(ValueError, match="requires a TradingCalendar"):
            di_future.time_to_expiry(valuation_date)

    def test_with_calendar_matches_business_days_over_252(
        self,
        di_future: DIFuture,
        valuation_date: date,
        trading_calendar: TradingCalendar,
    ) -> None:
        expected_days = trading_calendar.business_days_between(
            valuation_date, di_future.contract.expiry
        )
        result = di_future.time_to_expiry(valuation_date, calendar=trading_calendar)
        assert result == pytest.approx(expected_days / 252.0)

    def test_valuation_at_expiry_returns_zero_without_calendar(self, di_future: DIFuture) -> None:
        # The same-day short-circuit must run before the calendar-required check.
        assert di_future.time_to_expiry(di_future.contract.expiry) == 0.0

    def test_valuation_after_expiry_raises(self, di_future: DIFuture) -> None:
        past_expiry = date(2028, 1, 1)
        with pytest.raises(ValueError, match="is past expiry"):
            di_future.time_to_expiry(past_expiry)
