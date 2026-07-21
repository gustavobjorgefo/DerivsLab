"""Tests for derivslab.calendars.TradingCalendar."""

from __future__ import annotations

from datetime import date

import pytest

from derivslab.calendars import TradingCalendar


class TestIsBusinessDay:
    """Tests for TradingCalendar.is_business_day."""

    def test_weekday_without_holiday_is_business_day(self) -> None:
        calendar = TradingCalendar()
        assert calendar.is_business_day(date(2026, 7, 13))  # Monday

    def test_saturday_is_not_business_day(self) -> None:
        calendar = TradingCalendar()
        assert not calendar.is_business_day(date(2026, 7, 18))  # Saturday

    def test_sunday_is_not_business_day(self) -> None:
        calendar = TradingCalendar()
        assert not calendar.is_business_day(date(2026, 7, 19))  # Sunday

    def test_holiday_weekday_is_not_business_day(self) -> None:
        holiday = date(2026, 7, 16)  # Thursday
        calendar = TradingCalendar(holidays=frozenset({holiday}))
        assert not calendar.is_business_day(holiday)


class TestBusinessDaysBetween:
    """Tests for TradingCalendar.business_days_between."""

    def test_same_business_day_counts_as_one(self) -> None:
        calendar = TradingCalendar()
        assert calendar.business_days_between(date(2026, 7, 13), date(2026, 7, 13)) == 1

    def test_full_business_week_counts_five(self) -> None:
        calendar = TradingCalendar()
        # Monday 2026-07-13 through Friday 2026-07-17.
        assert calendar.business_days_between(date(2026, 7, 13), date(2026, 7, 17)) == 5

    def test_range_spanning_weekend_excludes_weekend_days(self) -> None:
        calendar = TradingCalendar()
        # Monday 2026-07-13 through Monday 2026-07-20: 6 weekdays, weekend excluded.
        assert calendar.business_days_between(date(2026, 7, 13), date(2026, 7, 20)) == 6

    def test_holiday_reduces_count_by_one(self, trading_calendar: TradingCalendar) -> None:
        # trading_calendar has a holiday on Thursday 2026-07-16.
        without_holiday = TradingCalendar().business_days_between(
            date(2026, 7, 13), date(2026, 7, 17)
        )
        with_holiday = trading_calendar.business_days_between(date(2026, 7, 13), date(2026, 7, 17))
        assert with_holiday == without_holiday - 1

    def test_end_before_start_raises_value_error(self) -> None:
        calendar = TradingCalendar()
        with pytest.raises(ValueError, match="is before start date"):
            calendar.business_days_between(date(2026, 7, 20), date(2026, 7, 13))
