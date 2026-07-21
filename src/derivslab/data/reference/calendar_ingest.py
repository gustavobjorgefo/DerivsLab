"""One-off ingestion of raw holiday-calendar source files.

Each data provider (ANBIMA, B3, BACEN, ...) publishes its holiday list in
its own raw format. This module holds one parsing function per source,
converting that raw format into the canonical reference-data schema —
``date``, ``description``, ``source`` — used by every downstream Parquet
file under ``data/reference/calendars/``.

Adding a new provider means adding one function here; it never touches
``calendar_loader.py``, which only ever reads the canonical schema and
has no notion of where the data originally came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import pandas as pd

ANBIMA_SOURCE_NAME: Final[str] = "ANBIMA"

_RAW_DATE_COLUMN: Final[str] = "Data"
_RAW_DESCRIPTION_COLUMN: Final[str] = "Feriado"
_CANONICAL_COLUMNS: Final[list[str]] = ["date", "description", "source"]


def parse_anbima_holidays(xlsx_path: Path) -> pd.DataFrame:
    """Parse ANBIMA's national holiday spreadsheet into the canonical schema.

    ANBIMA's published file has a header row, one row per holiday, and
    trailing footnote rows with no parseable date (a "Fonte: ANBIMA" line
    plus numbered notes) — rows whose date fails to parse are dropped
    rather than enumerated, since the footnote text itself is free-form
    and not meant to be parsed.

    Parameters
    ----------
    xlsx_path : Path
        Path to the raw .xlsx file as published by ANBIMA.

    Returns
    -------
    pd.DataFrame
        Columns ``date`` (datetime64[ns]), ``description`` (str), and
        ``source`` (str, always ``ANBIMA_SOURCE_NAME``), one row per
        holiday, sorted by date.

    Raises
    ------
    ValueError
        If the expected raw columns are not present in the file.
    """
    raw = pd.read_excel(xlsx_path)

    missing_columns = {_RAW_DATE_COLUMN, _RAW_DESCRIPTION_COLUMN} - set(raw.columns)
    if missing_columns:
        raise ValueError(
            f"ANBIMA file at {xlsx_path} is missing expected columns: {missing_columns}."
        )

    parsed_dates = pd.to_datetime(raw[_RAW_DATE_COLUMN], errors="coerce")
    valid_rows = raw.loc[parsed_dates.notna()].copy()
    valid_rows["date"] = parsed_dates.loc[parsed_dates.notna()]
    valid_rows["description"] = valid_rows[_RAW_DESCRIPTION_COLUMN]
    valid_rows["source"] = ANBIMA_SOURCE_NAME

    result: pd.DataFrame = valid_rows.loc[:, _CANONICAL_COLUMNS].sort_values("date")
    return cast(pd.DataFrame, result.reset_index(drop=True))
