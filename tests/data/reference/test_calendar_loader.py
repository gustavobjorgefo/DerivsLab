"""Tests for derivslab.data.reference.calendar_loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from derivslab.data.reference.calendar_loader import load_trading_calendar, save_holidays


@pytest.fixture
def canonical_holidays_df() -> pd.DataFrame:
    """A minimal DataFrame already in the canonical (date, description, source) schema."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-04-21", "2026-12-25"]),
            "description": ["Confraternização Universal", "Tiradentes", "Natal"],
            "source": ["ANBIMA", "ANBIMA", "ANBIMA"],
        }
    )


class TestSaveHolidays:
    """Tests for save_holidays."""

    def test_writes_a_parquet_file(
        self, canonical_holidays_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "calendars" / "anbima_holidays.parquet"
        save_holidays(canonical_holidays_df, path)
        assert path.exists()

    def test_missing_canonical_column_raises(self, tmp_path: Path) -> None:
        incomplete_df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"])})
        with pytest.raises(ValueError, match="missing expected columns"):
            save_holidays(incomplete_df, tmp_path / "bad.parquet")


class TestLoadTradingCalendar:
    """Tests for load_trading_calendar."""

    def test_holidays_round_trip_as_dates(
        self, canonical_holidays_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "anbima_holidays.parquet"
        save_holidays(canonical_holidays_df, path)

        calendar = load_trading_calendar(path, name="ANBIMA")

        assert calendar.holidays == frozenset(
            {date(2026, 1, 1), date(2026, 4, 21), date(2026, 12, 25)}
        )

    def test_calendar_is_tagged_with_given_name(
        self, canonical_holidays_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "anbima_holidays.parquet"
        save_holidays(canonical_holidays_df, path)

        calendar = load_trading_calendar(path, name="ANBIMA")

        assert calendar.name == "ANBIMA"

    def test_loaded_calendar_recognizes_a_known_holiday(
        self, canonical_holidays_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "anbima_holidays.parquet"
        save_holidays(canonical_holidays_df, path)

        calendar = load_trading_calendar(path, name="ANBIMA")

        assert not calendar.is_business_day(date(2026, 12, 25))
