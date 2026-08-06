"""Manual smoke test: instantiate a rate curve and print it.

Run directly with:
    python scripts/smoke_rates.py

Not a pytest test — a quick, throwaway way to eyeball that a RateCurve
builds correctly and reads back sensibly.
"""

from __future__ import annotations

from datetime import date

from derivslab.calendars import TradingCalendar
from derivslab.rates import FlatRateCurve

VALUATION_DATE = date(2026, 7, 27)
FLAT_RATE = 0.08  # 8% a.a., continuously compounded


def main() -> None:
    calendar = TradingCalendar(name="smoke-test")

    curve = FlatRateCurve(
        rate=FLAT_RATE,
        valuation_date=VALUATION_DATE,
        calendar=calendar,
    )

    print(f"valuation_date       : {curve.valuation_date}")
    print(f"calendar             : {curve.calendar.name}")
    print(f"day_count_convention : {curve.day_count_convention}")
    print(f"flat rate            : {curve.rate:.4%}")
    print()

    one_year = date(2027, 7, 27)
    five_years = date(2031, 7, 27)

    print(f"discount_factor(1y)  : {curve.discount_factor(one_year):.6f}")
    print(f"discount_factor(5y)  : {curve.discount_factor(five_years):.6f}")
    print()

    print(f"zero_rate(1y)        : {curve.zero_rate(one_year):.4%}")
    print(f"forward_rate(1y, 5y) : {curve.forward_rate(one_year, five_years):.4%}")


if __name__ == "__main__":
    main()