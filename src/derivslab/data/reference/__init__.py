"""Slow-changing reference data: holiday calendars and instrument registries.

Exposes the ingestion functions (one per raw source format) and the
loader (reads the canonical Parquet schema into a ``TradingCalendar``).
"""

from __future__ import annotations

from derivslab.data.reference.calendar_ingest import ANBIMA_SOURCE_NAME, parse_anbima_holidays
from derivslab.data.reference.calendar_loader import (
    CALENDARS_DIR,
    load_trading_calendar,
    save_holidays,
)

__all__ = [
    "ANBIMA_SOURCE_NAME",
    "CALENDARS_DIR",
    "load_trading_calendar",
    "parse_anbima_holidays",
    "save_holidays",
]
