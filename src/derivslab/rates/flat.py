"""Flat interest-rate curve: a single constant rate for every maturity.

Responsibilities
----------------
- Provide the simplest concrete ``RateCurve``: one continuously-
  compounded rate applied uniformly to any maturity, with no term
  structure.
- Serve as the reference implementation of the ``RateCurve`` contract —
  every other concrete curve (bootstrapped, stochastic) must reproduce
  the same ``zero_rate``/``forward_rate``/``discount_factor``
  relationships that this one demonstrates trivially, since all three
  are derived once in ``RateCurve`` itself.

What does NOT belong here
--------------------------
- Interpolation or bootstrap from market instruments (e.g. DI futures)
  — that belongs to a future interpolated-curve module.
- Rate-convention conversion (e.g. BUS_252 discrete, as quoted on B3,
  to continuous) — that is the caller's job before constructing this
  curve; see ``derivslab.rates.conventions`` once it exists.
"""

from __future__ import annotations

from datetime import date
from math import exp
from typing import TYPE_CHECKING

from derivslab.instruments.contracts import DayCountConvention
from derivslab.rates.base import RateCurve

if TYPE_CHECKING:
    from derivslab.calendars import TradingCalendar


class FlatRateCurve(RateCurve):
    """A rate curve with a single continuously-compounded rate for every maturity.

    Parameters
    ----------
    rate : float
        Continuously-compounded annualized rate, applied uniformly to
        every maturity. Must already be expressed under continuous
        compounding — convert a market-quoted rate (e.g. BUS_252
        discrete) before constructing this curve.
    valuation_date : date
        The date this curve is anchored to.
    calendar : TradingCalendar
        Trading calendar used to count business days when
        ``day_count_convention`` is ``BUS_252``. Required even though a
        flat curve has no term structure of its own, so every
        ``RateCurve`` implementation shares the same construction
        contract.
    day_count_convention : DayCountConvention
        Convention used to convert date intervals into year fractions.
        Defaults to ``ACT_365``, the common international convention;
        pass ``BUS_252`` explicitly to match B3-quoted rates.
    """

    def __init__(
        self,
        rate: float,
        valuation_date: date,
        calendar: TradingCalendar,
        day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    ) -> None:
        self._rate: float = rate
        self._valuation_date: date = valuation_date
        self._calendar: TradingCalendar = calendar
        self._day_count_convention: DayCountConvention = day_count_convention

    # --- RateCurve interface -------------------------------------------------

    @property
    def valuation_date(self) -> date:
        """Return the date this curve is anchored to."""
        return self._valuation_date

    @property
    def calendar(self) -> TradingCalendar:
        """Return the trading calendar used to count business days."""
        return self._calendar

    @property
    def day_count_convention(self) -> DayCountConvention:
        """Return the day-count convention used to compute year fractions."""
        return self._day_count_convention

    def discount_factor(self, maturity: date) -> float:
        """Return the present value of 1 unit of currency paid at ``maturity``.

        Computed as ``exp(-rate * year_fraction(valuation_date, maturity))``.
        Since the curve is flat, this is the entire pricing logic — no
        interpolation between vertices is involved.

        Parameters
        ----------
        maturity : date
            Date the unit of currency is paid. May equal
            ``valuation_date``.

        Returns
        -------
        float
            Discount factor. Equal to ``1.0`` when
            ``maturity == valuation_date``.

        Raises
        ------
        ValueError
            If ``maturity`` is before ``valuation_date``.
        """
        if maturity < self._valuation_date:
            raise ValueError(
                f"Cannot discount maturity {maturity} before valuation_date "
                f"{self._valuation_date}."
            )
        year_fraction = self.year_fraction(self._valuation_date, maturity)
        return exp(-self._rate * year_fraction)

    # --- flat-curve-specific ---------------------------------------------------

    @property
    def rate(self) -> float:
        """Return the flat continuously-compounded rate applied to every maturity."""
        return self._rate
