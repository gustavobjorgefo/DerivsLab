"""Tests for derivslab.instruments.equity.EquityInstrument."""

from __future__ import annotations

from datetime import date
from math import inf

from derivslab.calendars import TradingCalendar
from derivslab.instruments.equity import EQUITY_PRICING_MODEL_KEY, EquityInstrument


class TestEquityInstrument:
    """Tests for EquityInstrument."""

    def test_contract_returns_the_wrapped_contract(
        self, equity_instrument: EquityInstrument, equity_contract
    ) -> None:
        assert equity_instrument.contract is equity_contract

    def test_underlyings_is_its_own_ticker(self, equity_instrument: EquityInstrument) -> None:
        assert equity_instrument.underlyings == ["PETR4"]

    def test_pricing_model_key_is_equity_spot(self, equity_instrument: EquityInstrument) -> None:
        assert equity_instrument.pricing_model_key == EQUITY_PRICING_MODEL_KEY

    def test_payoff_returns_the_spot_price(self, equity_instrument: EquityInstrument) -> None:
        assert equity_instrument.payoff({"PETR4": 34.5}) == 34.5

    def test_time_to_expiry_is_infinite(
        self, equity_instrument: EquityInstrument, valuation_date: date
    ) -> None:
        assert equity_instrument.time_to_expiry(valuation_date) == inf

    def test_time_to_expiry_ignores_a_calendar_if_given(
        self, equity_instrument: EquityInstrument, valuation_date: date
    ) -> None:
        calendar = TradingCalendar(holidays=frozenset({valuation_date}))
        assert equity_instrument.time_to_expiry(valuation_date, calendar=calendar) == inf