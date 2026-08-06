"""Interest-rate curve abstractions for the DerivsLab rates package.

Responsibilities
----------------
- Define ``RateCurve``, the behavioral interface every interest-rate
  curve model must implement: a flat curve, an interpolated curve
  bootstrapped from DI futures, or a single node produced by a
  stochastic rate simulation.
- Derive ``zero_rate`` and ``forward_rate`` once, in terms of the sole
  abstract primitive, ``discount_factor``, so every concrete curve gets
  them for free and none can disagree on the compounding convention
  used to report them.
- Standardise ``RateCurve``'s public output on continuous compounding
  (``e^(-rT)``) regardless of the market convention a concrete curve was
  built from — e.g. BUS_252 discrete, as quoted on B3's DI futures.
  Converting between market conventions is the concern of whatever
  builds the curve (see ``derivslab.rates.conventions``, once it
  exists), never of ``RateCurve`` itself or of its consumers.

What does NOT belong here
--------------------------
- Concrete curve implementations (``FlatRateCurve``, an interpolated
  curve bootstrapped from DI futures) — those live in their own
  modules.
- Calibration or bootstrap logic — a ``RateCurve`` is an immutable
  result, never the process that produced it.
- Stochastic simulation of the curve over time — that is a generator
  of ``RateCurve`` instances, not a variant of this interface.
- Holiday calendars — those live in ``derivslab.calendars``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from math import log
from typing import TYPE_CHECKING, Final

from derivslab.instruments.contracts import DayCountConvention

if TYPE_CHECKING:
    from derivslab.calendars import TradingCalendar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Denominator per calendar-day convention. BUS_252 is handled separately
# because it counts business days through a TradingCalendar instead.
_CALENDAR_DAYS_PER_YEAR: Final[dict[DayCountConvention, float]] = {
    DayCountConvention.ACT_360: 360.0,
    DayCountConvention.ACT_365: 365.0,
}

_BUSINESS_DAYS_PER_YEAR: Final[float] = 252.0


# ---------------------------------------------------------------------------
# RateCurve
# ---------------------------------------------------------------------------


class RateCurve(ABC):
    """Behavioral interface shared by every interest-rate curve model.

    A ``RateCurve`` is an immutable snapshot: it answers questions about
    discounting and rates as of its own ``valuation_date`` and never
    mutates afterwards. Refitting a curve — from a new set of DI futures
    quotes, or from the next step of a stochastic simulation — always
    produces a new ``RateCurve`` instance rather than updating an
    existing one, the same way a new ``MarketSnapshot`` replaces the
    previous one on every refresh cycle.

    ``discount_factor`` is the only abstract primitive. ``zero_rate``
    and ``forward_rate`` are derived from it once, here, so a concrete
    curve — flat, interpolated, or a node of a simulated path — never
    implements them itself and can never report a rate under a
    different compounding convention than the rest of the system
    expects.

    See Also
    --------
    derivslab.rates.flat.FlatRateCurve : Simplest concrete curve — a
        single constant rate applied to every maturity.
    """

    # --- abstract interface -------------------------------------------------

    @property
    @abstractmethod
    def valuation_date(self) -> date:
        """Return the date this curve is anchored to."""
        ...

    @property
    @abstractmethod
    def calendar(self) -> TradingCalendar:
        """Return the trading calendar used to count business days."""
        ...

    @property
    @abstractmethod
    def day_count_convention(self) -> DayCountConvention:
        """Return the day-count convention used to compute year fractions."""
        ...

    @abstractmethod
    def discount_factor(self, maturity: date) -> float:
        """Return the present value of 1 unit of currency paid at ``maturity``.

        Parameters
        ----------
        maturity : date
            Date the unit of currency is paid. May equal
            ``valuation_date``.

        Returns
        -------
        float
            Discount factor in ``(0, 1]``. Equal to ``1.0`` when
            ``maturity == valuation_date``.
        """
        ...

    # --- derived, concrete ---------------------------------------------------

    def zero_rate(self, maturity: date) -> float:
        """Compute the continuously-compounded zero rate to ``maturity``.

        Derived from ``discount_factor`` as
        ``-ln(discount_factor(maturity)) / year_fraction(valuation_date, maturity)``,
        independently of whichever market convention the concrete curve
        was originally built from.

        Parameters
        ----------
        maturity : date
            Date the zero rate is measured to. Must be strictly after
            ``valuation_date``.

        Returns
        -------
        float
            Continuously-compounded annualized rate.

        Raises
        ------
        ValueError
            If ``maturity`` is not strictly after ``valuation_date``.
        """
        year_fraction = self.year_fraction(self.valuation_date, maturity)
        if year_fraction <= 0.0:
            raise ValueError(
                f"zero_rate requires maturity ({maturity}) strictly after "
                f"valuation_date ({self.valuation_date})."
            )
        return -log(self.discount_factor(maturity)) / year_fraction

    def forward_rate(self, start: date, end: date) -> float:
        """Compute the continuously-compounded forward rate over ``[start, end]``.

        The rate implied by today's curve for a future interval — not a
        prediction of the rate that will actually prevail then, but the
        rate consistent with today's discount factors at both endpoints
        under no-arbitrage.

        Parameters
        ----------
        start : date
            Start of the interval.
        end : date
            End of the interval. Must be strictly after ``start``.

        Returns
        -------
        float
            Continuously-compounded annualized forward rate.

        Raises
        ------
        ValueError
            If ``end`` is not strictly after ``start``.
        """
        year_fraction = self.year_fraction(start, end)
        if year_fraction <= 0.0:
            raise ValueError(f"forward_rate requires end ({end}) strictly after start ({start}).")
        discount_factor_start = self.discount_factor(start)
        discount_factor_end = self.discount_factor(end)
        return log(discount_factor_start / discount_factor_end) / year_fraction

    def year_fraction(self, start: date, end: date) -> float:
        """Convert a date interval into a year fraction under this curve's convention.

        BUS_252 counts business days through ``calendar`` and divides by
        252. ACT_360 and ACT_365 count calendar days directly and never
        touch ``calendar``. Mirrors
        ``derivslab.instruments.vanilla.VanillaOption.time_to_expiry``, so
        curve and instrument agree on year fractions under the same
        convention.

        Parameters
        ----------
        start : date
            Start of the interval.
        end : date
            End of the interval.

        Returns
        -------
        float
            Year fraction, ``0.0`` when ``start == end``.

        Raises
        ------
        ValueError
            If ``end`` is before ``start``.
        """
        if end < start:
            raise ValueError(f"end ({end}) is before start ({start}).")
        if end == start:
            return 0.0

        if self.day_count_convention is DayCountConvention.BUS_252:
            business_days = self.calendar.business_days_between(start, end)
            return business_days / _BUSINESS_DAYS_PER_YEAR

        return (end - start).days / _CALENDAR_DAYS_PER_YEAR[self.day_count_convention]
