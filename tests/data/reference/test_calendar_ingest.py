"""Tests for derivslab.data.reference.calendar_ingest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from derivslab.data.reference.calendar_ingest import ANBIMA_SOURCE_NAME, parse_anbima_holidays


class TestParseAnbimaHolidays:
    """Tests for parse_anbima_holidays."""

    def test_drops_footnote_rows(self, sample_anbima_xlsx: Path) -> None:
        result = parse_anbima_holidays(sample_anbima_xlsx)
        assert len(result) == 3

    def test_returns_canonical_columns(self, sample_anbima_xlsx: Path) -> None:
        result = parse_anbima_holidays(sample_anbima_xlsx)
        assert list(result.columns) == ["date", "description", "source"]

    def test_tags_every_row_with_anbima_source(self, sample_anbima_xlsx: Path) -> None:
        result = parse_anbima_holidays(sample_anbima_xlsx)
        assert (result["source"] == ANBIMA_SOURCE_NAME).all()

    def test_preserves_holiday_descriptions(self, sample_anbima_xlsx: Path) -> None:
        result = parse_anbima_holidays(sample_anbima_xlsx)
        assert "Tiradentes" in result["description"].values

    def test_result_is_sorted_by_date(self, sample_anbima_xlsx: Path) -> None:
        result = parse_anbima_holidays(sample_anbima_xlsx)
        assert list(result["date"]) == sorted(result["date"])

    def test_missing_expected_columns_raises(self, tmp_path: Path) -> None:
        malformed_path = tmp_path / "malformed.xlsx"
        pd.DataFrame({"Coluna Errada": [1, 2, 3]}).to_excel(malformed_path, index=False)

        with pytest.raises(ValueError, match="missing expected columns"):
            parse_anbima_holidays(malformed_path)
