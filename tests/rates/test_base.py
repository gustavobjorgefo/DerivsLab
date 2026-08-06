"""
Tests for derivslab.rates.base.

Coverage
--------
RateCurve
    - Cannot be instantiated directly (ABC without a concrete
      ``discount_factor``).

RateCurve.year_fraction
    - BUS_252 counts business days through the calendar and divides by 252.
    - ACT_360 / ACT_365 count calendar days directly, ignoring the calendar.
    - Same start and end returns 0.0.
    - end before start raises ValueError.

RateCurve.zero_rate
    - Recovers the continuously-compounded rate implied by a known
      discount factor.
    - maturity == valuation_date raises ValueError.
    - maturity before valuation_date raises ValueError (propagated from
      year_fraction).

RateCurve.forward_rate
    - Recovers the continuously-compounded forward rate implied by two
      known discount factors.
    - end == start raises ValueError.
    - end before start raises ValueError (propagated from year_fraction).

Uses ``_StubRateCurve``, a minimal concrete double whose discount factors
are supplied directly by the test — decoupled from any particular curve
model (e.g. FlatRateCurve), so these tests exercise the ABC's derived
math in isolation.
"""

from __future__ import annotations

from datetime import date
from math import exp, log

import pytest

from derivslab.calendars import TradingCalendar
from derivslab.instruments.contracts import DayCountConvention
from derivslab.rates.base import RateCurve

# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class _StubRateCurve(RateCurve):
    """Minimal concrete RateCurve backed by a caller-supplied DF mapping."""

    def __init__(
        self,
        discount_factors: dict[date, float],
        valuation_date: date,
        calendar: TradingCalendar,
        day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    ) -> None:
        self._discount_factors: dict[date, float] = discount_factors
        self._valuation_date: date = valuation_date
        self._calendar: TradingCalendar = calendar
        self._day_count_convention: DayCountConvention = day_count_convention

    @property
    def valuation_date(self) -> date:
        return self._valuation_date

    @property
    def calendar(self) -> TradingCalendar:
        return self._calendar

    @property
    def day_count_convention(self) -> DayCountConvention:
        return self._day_count_convention

    def discount_factor(self, maturity: date) -> float:
        return self._discount_factors[maturity]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calendar() -> TradingCalendar:
    return TradingCalendar(name="test")


# ---------------------------------------------------------------------------
# RateCurve
# ---------------------------------------------------------------------------


class TestRateCurveIsAbstract:
    """RateCurve cannot be instantiated without a concrete discount_factor."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            RateCurve()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# year_fraction
# ---------------------------------------------------------------------------


class TestYearFraction:
    """Tests for RateCurve.year_fraction."""

    def test_bus_252_divides_business_days_by_252(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={},
            valuation_date=valuation_date,
            calendar=calendar,
            day_count_convention=DayCountConvention.BUS_252,
        )
        end = date(2026, 7, 17)  # Friday, same week as valuation_date (Monday)
        expected = calendar.business_days_between(valuation_date, end) / 252.0
        assert curve.year_fraction(valuation_date, end) == pytest.approx(expected)

    def test_act_365_divides_calendar_days_by_365(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={},
            valuation_date=valuation_date,
            calendar=calendar,
            day_count_convention=DayCountConvention.ACT_365,
        )
        end = date(2027, 7, 13)
        assert curve.year_fraction(valuation_date, end) == pytest.approx(365 / 365.0)

    def test_act_360_divides_calendar_days_by_360(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={},
            valuation_date=valuation_date,
            calendar=calendar,
            day_count_convention=DayCountConvention.ACT_360,
        )
        end = date(2027, 7, 13)
        assert curve.year_fraction(valuation_date, end) == pytest.approx(365 / 360.0)

    def test_act_365_ignores_the_calendar(self, valuation_date: date) -> None:
        # A calendar with valuation_date itself as a holiday must have no
        # effect on ACT_365 — it only counts calendar days.
        holiday_calendar = TradingCalendar(holidays=frozenset({valuation_date}))
        curve = _StubRateCurve(
            discount_factors={},
            valuation_date=valuation_date,
            calendar=holiday_calendar,
            day_count_convention=DayCountConvention.ACT_365,
        )
        end = date(2026, 8, 12)
        assert curve.year_fraction(valuation_date, end) == pytest.approx(30 / 365.0)

    def test_same_start_and_end_returns_zero(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={}, valuation_date=valuation_date, calendar=calendar
        )
        assert curve.year_fraction(valuation_date, valuation_date) == 0.0

    def test_end_before_start_raises(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={}, valuation_date=valuation_date, calendar=calendar
        )
        earlier = date(2026, 1, 1)
        with pytest.raises(ValueError, match="is before start"):
            curve.year_fraction(valuation_date, earlier)


# ---------------------------------------------------------------------------
# zero_rate
# ---------------------------------------------------------------------------


class TestZeroRate:
    """Tests for RateCurve.zero_rate."""

    def test_recovers_the_rate_implied_by_the_discount_factor(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        maturity = date(2027, 7, 13)  # exactly 365 days after valuation_date
        rate = 0.12
        curve = _StubRateCurve(
            discount_factors={maturity: exp(-rate * 1.0)},
            valuation_date=valuation_date,
            calendar=calendar,
            day_count_convention=DayCountConvention.ACT_365,
        )
        assert curve.zero_rate(maturity) == pytest.approx(rate)

    def test_maturity_equal_valuation_date_raises(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={valuation_date: 1.0},
            valuation_date=valuation_date,
            calendar=calendar,
        )
        with pytest.raises(ValueError, match="strictly after"):
            curve.zero_rate(valuation_date)

    def test_maturity_before_valuation_date_raises(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={}, valuation_date=valuation_date, calendar=calendar
        )
        earlier = date(2026, 1, 1)
        with pytest.raises(ValueError, match="is before start"):
            curve.zero_rate(earlier)


# ---------------------------------------------------------------------------
# forward_rate
# ---------------------------------------------------------------------------


class TestForwardRate:
    """Tests for RateCurve.forward_rate."""

    def test_recovers_the_rate_implied_by_two_discount_factors(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        start = date(2027, 7, 13)
        end = date(2028, 7, 12)  # exactly 365 days after start (2028 is a leap year)
        df_start = 0.90
        df_end = 0.81
        curve = _StubRateCurve(
            discount_factors={start: df_start, end: df_end},
            valuation_date=valuation_date,
            calendar=calendar,
            day_count_convention=DayCountConvention.ACT_365,
        )
        expected_forward = log(df_start / df_end) / (365 / 365.0)
        assert curve.forward_rate(start, end) == pytest.approx(expected_forward)

    def test_end_equal_start_raises(self, calendar: TradingCalendar, valuation_date: date) -> None:
        curve = _StubRateCurve(
            discount_factors={}, valuation_date=valuation_date, calendar=calendar
        )
        with pytest.raises(ValueError, match="strictly after"):
            curve.forward_rate(valuation_date, valuation_date)

    def test_end_before_start_raises(
        self, calendar: TradingCalendar, valuation_date: date
    ) -> None:
        curve = _StubRateCurve(
            discount_factors={}, valuation_date=valuation_date, calendar=calendar
        )
        start = date(2027, 7, 13)
        end = date(2026, 1, 1)
        with pytest.raises(ValueError, match="is before start"):
            curve.forward_rate(start, end)
