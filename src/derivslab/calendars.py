"""Trading calendars used to compute business-day counts.

Only the BUS_252 day-count convention needs a calendar — a holiday set
layered on top of weekends — to count business days. ACT_360 and ACT_365
are pure calendar-day counts and never touch this module; see
``VanillaOption.time_to_expiry`` for how the two paths are chosen based on
``InstrumentContract.day_count_convention``.

Populating a calendar with real exchange holidays (B3, ANBIMA, ...) is a
data-loading concern — fetched from an exchange feed or a reference
table — and deliberately does not live in this module: hardcoding a
holiday list in source code goes stale every year without anyone
noticing. This module only implements the counting logic over whatever
holiday set it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np


@dataclass(frozen=True)
class TradingCalendar:
    """A business-day calendar: weekends plus an explicit holiday set.

    Parameters
    ----------
    holidays : frozenset[date]
        Non-business days beyond weekends (e.g. national and exchange
        holidays). Empty by default, meaning weekends-only.
    name : str
        Human-readable identifier (e.g. "B3"), used only in error
        messages.
    """

    holidays: frozenset[date] = field(default_factory=frozenset)
    name: str = "default"

    def is_business_day(self, day: date) -> bool:
        """Return whether ``day`` is a business day under this calendar."""
        return day.weekday() < 5 and day not in self.holidays

    def business_days_between(self, start: date, end: date) -> int:
        """Count business days in the closed interval ``[start, end]``.

        Parameters
        ----------
        start : date
            First date in the count.
        end : date
            Last date in the count.

        Returns
        -------
        int
            Number of business days from ``start`` to ``end``, inclusive
            of both endpoints.

        Raises
        ------
        ValueError
            If ``end`` is before ``start``.
        """
        if end < start:
            raise ValueError(
                f"End date {end} is before start date {start} " f"in calendar '{self.name}'."
            )

        holidays_np = np.array(sorted(self.holidays), dtype="datetime64[D]")
        # busday_count treats its end argument as exclusive, so we push it
        # forward one day to make the caller's end date inclusive.
        return int(np.busday_count(start, end + timedelta(days=1), holidays=holidays_np))
