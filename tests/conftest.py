"""Shared fixtures for the DerivsLab test suite."""

from __future__ import annotations

from datetime import date
from typing import Callable

import pytest

from derivslab.calendars import TradingCalendar
from derivslab.instruments.contracts import (
    Currency,
    DayCountConvention,
    DIFutureContract,
    EquityContract,
    ExerciseStyle,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)
from derivslab.instruments.equity import EquityInstrument
from derivslab.instruments.future import DIFuture
from derivslab.instruments.vanilla import VanillaOption


@pytest.fixture
def valuation_date() -> date:
    """A fixed valuation date used across tests for reproducibility."""
    return date(2026, 7, 13)


@pytest.fixture
def equity_contract() -> EquityContract:
    """A minimal, valid equity contract."""
    return EquityContract(instrument_id="PETR4", currency=Currency.BRL, ticker="PETR4")


@pytest.fixture
def equity_instrument(equity_contract: EquityContract) -> EquityInstrument:
    """An EquityInstrument wrapping ``equity_contract``."""
    return EquityInstrument(equity_contract)


@pytest.fixture
def make_option_contract() -> Callable[..., VanillaOptionContract]:
    """Factory for VanillaOptionContract with sensible overridable defaults."""

    def _make(**overrides: object) -> VanillaOptionContract:
        defaults: dict[str, object] = {
            "instrument_id": "PETRA123",
            "currency": Currency.BRL,
            "ticker": "PETRA123",
            "underlying": "PETR4",
            "underlying_asset_class": UnderlyingAssetClass.EQUITY,
            "option_type": OptionType.CALL,
            "style": ExerciseStyle.EUROPEAN,
            "strike": 35.0,
            "expiry": date(2026, 12, 18),
            "day_count_convention": DayCountConvention.ACT_365,
        }
        defaults.update(overrides)
        return VanillaOptionContract(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def european_call(make_option_contract: Callable[..., VanillaOptionContract]) -> VanillaOption:
    """A European call, ACT_365 day-count."""
    return VanillaOption(make_option_contract())


@pytest.fixture
def american_put(make_option_contract: Callable[..., VanillaOptionContract]) -> VanillaOption:
    """An American put, BUS_252 day-count, expiring the week after ``valuation_date``."""
    contract = make_option_contract(
        instrument_id="PETRA999",
        ticker="PETRA999",
        option_type=OptionType.PUT,
        style=ExerciseStyle.AMERICAN,
        expiry=date(2026, 7, 20),
        day_count_convention=DayCountConvention.BUS_252,
    )
    return VanillaOption(contract)


@pytest.fixture
def trading_calendar() -> TradingCalendar:
    """A test calendar with a single holiday (2026-07-16, a Thursday)."""
    return TradingCalendar(holidays=frozenset({date(2026, 7, 16)}), name="test-calendar")


@pytest.fixture
def di_future_contract() -> DIFutureContract:
    """A minimal, valid DI1 futures contract, expiring one year after ``valuation_date``."""
    return DIFutureContract(
        instrument_id="DI1F27",
        currency=Currency.BRL,
        ticker="DI1F27",
        expiry=date(2027, 7, 13),
    )


@pytest.fixture
def di_future(di_future_contract: DIFutureContract) -> DIFuture:
    """A DIFuture wrapping ``di_future_contract``."""
    return DIFuture(di_future_contract)
