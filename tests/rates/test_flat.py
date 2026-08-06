"""
Tests for derivslab.rates.flat.

Coverage
--------
FlatRateCurve
    - valuation_date, calendar, day_count_convention, and rate are stored
      and returned as given.
    - day_count_convention defaults to ACT_365.

FlatRateCurve.discount_factor
    - Equals 1.0 at valuation_date.
    - Matches exp(-rate * year_fraction) under ACT_365.
    - Matches exp(-rate * year_fraction) under BUS_252.
    - maturity before valuation_date raises ValueError.

Integration with the inherited RateCurve behavior
    - zero_rate recovers the flat rate exactly, for any maturity.
    - forward_rate recovers the flat rate exactly, for any interval —
      a flat curve has no term structure to give it shape.
"""

from __future__ import annotations

from datetime import date
from math import exp

import pytest

from derivslab.calendars import TradingCalendar
from derivslab.instruments.contracts import DayCountConvention
from derivslab.rates.flat import FlatRateCurve

RATE: float = 0.135

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calendar() -> TradingCalendar:
    return TradingCalendar(name="test")


@pytest.fixture
def flat_curve(calendar: TradingCalendar, valuation_date: date) -> FlatRateCurve:
    """A flat curve at RATE, ACT_365, anchored at valuation_date."""
    return FlatRateCurve(rate=RATE, valuation_date=valuation_date, calendar=calendar)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for the fields FlatRateCurve is constructed with."""

    def test_rate_is_stored(self, flat_curve: FlatRateCurve) -> None:
        assert flat_curve.rate == RATE

    def test_valuation_date_is_stored(
        self, flat_curve: FlatRateCurve, valuation_date: date
    ) -> None:
        assert flat_curve.valuation_date == valuation_date

    def test_calendar_is_stored(
        self, flat_curve: FlatRateCurve, calendar: TradingCalendar
    ) -> None:
        assert flat_curve.calendar is calendar

    def test_day_count_convention_defaults_to_act_365(self, flat_curve: FlatRateCurve) -> None:
        assert flat_curve.day_count_convention is DayCountConvention.ACT_365

    def test_day_count_convention_can_be_overridden(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = FlatRateCurve(
            rate=RATE,
            valuation_date=valuation_date,
            calendar=calendar,
            day_count_convention=DayCountConvention.BUS_252,
        )
        assert curve.day_count_convention is DayCountConvention.BUS_252


# ---------------------------------------------------------------------------
# discount_factor
# ---------------------------------------------------------------------------


class TestDiscountFactor:
    """Tests for FlatRateCurve.discount_factor."""

    def test_equals_one_at_valuation_date(
        self, flat_curve: FlatRateCurve, valuation_date: date
    ) -> None:
        assert flat_curve.discount_factor(valuation_date) == 1.0

    def test_matches_exp_formula_under_act_365(
        self, flat_curve: FlatRateCurve, valuation_date: date
    ) -> None:
        maturity = date(2027, 7, 13)  # exactly 365 days after valuation_date
        expected = exp(-RATE * (365 / 365.0))
        assert flat_curve.discount_factor(maturity) == pytest.approx(expected)

    def test_matches_exp_formula_under_bus_252(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = FlatRateCurve(
            rate=RATE,
            valuation_date=valuation_date,
            calendar=calendar,
            day_count_convention=DayCountConvention.BUS_252,
        )
        maturity = date(2026, 7, 17)  # Friday, same week as valuation_date (Monday)
        business_days = calendar.business_days_between(valuation_date, maturity)
        expected = exp(-RATE * (business_days / 252.0))
        assert curve.discount_factor(maturity) == pytest.approx(expected)

    def test_maturity_before_valuation_date_raises(
        self, flat_curve: FlatRateCurve, valuation_date: date
    ) -> None:
        earlier = date(2026, 1, 1)
        with pytest.raises(ValueError, match="before valuation_date"):
            flat_curve.discount_factor(earlier)

    def test_decreases_as_maturity_extends(self, flat_curve: FlatRateCurve) -> None:
        near = flat_curve.discount_factor(date(2027, 1, 1))
        far = flat_curve.discount_factor(date(2028, 1, 1))
        assert 0.0 < far < near < 1.0


# ---------------------------------------------------------------------------
# Integration with RateCurve.zero_rate / forward_rate
# ---------------------------------------------------------------------------


class TestInheritedRateCurveBehavior:
    """A flat curve has no term structure — every rate it reports is RATE."""

    def test_zero_rate_recovers_the_flat_rate(self, flat_curve: FlatRateCurve) -> None:
        assert flat_curve.zero_rate(date(2027, 7, 13)) == pytest.approx(RATE)
        assert flat_curve.zero_rate(date(2036, 7, 13)) == pytest.approx(RATE)

    def test_forward_rate_recovers_the_flat_rate(self, flat_curve: FlatRateCurve) -> None:
        assert flat_curve.forward_rate(date(2027, 1, 1), date(2028, 1, 1)) == pytest.approx(RATE)
        assert flat_curve.forward_rate(date(2030, 1, 1), date(2031, 6, 1)) == pytest.approx(RATE)
