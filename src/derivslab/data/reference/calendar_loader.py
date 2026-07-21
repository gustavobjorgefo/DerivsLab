"""Routine loading of holiday-calendar reference data.

This module only ever reads the canonical schema (``date``,
``description``, ``source``) that ``calendar_ingest.py`` produces — it
has no knowledge of any provider's raw file format. Splitting ingestion
(rare: run once per new source, or once a year to refresh) from loading
(routine: run on every pricing or simulation job) keeps one-off parsing
logic out of the hot path and out of anything that depends on
``TradingCalendar``.

Storage starts as local Parquet under ``CALENDARS_DIR``; migrating to S3
later only means swapping that path for an ``s3://`` URI, since pandas
and pyarrow both read Parquet transparently from either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from derivslab.calendars import TradingCalendar

# Local default; swap for an s3:// URI once reference data moves to S3.
CALENDARS_DIR: Final[Path] = Path("data/reference/calendars")

_CANONICAL_COLUMNS: Final[list[str]] = ["date", "description", "source"]


def save_holidays(holidays: pd.DataFrame, path: Path) -> None:
    """Persist a canonical-schema holiday DataFrame to Parquet.

    Parameters
    ----------
    holidays : pd.DataFrame
        Must contain ``date``, ``description``, and ``source`` columns.
    path : Path
        Destination file. Parent directories are created if missing.

    Raises
    ------
    ValueError
        If any canonical column is missing.
    """
    missing_columns = set(_CANONICAL_COLUMNS) - set(holidays.columns)
    if missing_columns:
        raise ValueError(f"holidays is missing expected columns: {missing_columns}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    holidays.loc[:, _CANONICAL_COLUMNS].to_parquet(path, index=False)


def load_trading_calendar(path: Path, name: str) -> TradingCalendar:
    """Build a TradingCalendar from a canonical-schema Parquet file.

    Only the ``date`` column is read into the calendar itself —
    ``description`` and ``source`` stay in the Parquet file for audit
    purposes and are not carried into the in-memory ``TradingCalendar``.

    Parameters
    ----------
    path : Path
        Path to a Parquet file written by ``save_holidays``.
    name : str
        Identifier attached to the resulting calendar (e.g. "ANBIMA").

    Returns
    -------
    TradingCalendar
        Calendar whose ``holidays`` is the full set of dates in the file.
    """
    holidays = pd.read_parquet(path, columns=["date"])
    holiday_dates = frozenset(holidays["date"].dt.date)
    return TradingCalendar(holidays=holiday_dates, name=name)
