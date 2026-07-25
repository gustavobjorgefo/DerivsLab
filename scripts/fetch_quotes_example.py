"""Manual smoke test — ProfitProRTDProvider.

Runs a real quote collection via ProfitPro RTD and prints the results
to the terminal. This is not an automated test — it requires an active
external environment.

Pre-conditions
--------------
1. ProfitPro open and logged in with RTD active.
2. Excel open with ``rtd_feed.xlsx`` loaded and the ``rtd_feed``
   worksheet present.
3. Virtual environment activated with ``xlwings`` installed.

How to run
----------
With the .venv activated, from the project root::

    python scripts/fetch_quotes_example.py

Adjust the ``EQUITIES`` and ``OPTIONS`` lists below to match the
instruments available in your ProfitPro session.
"""

from __future__ import annotations

from datetime import date

from derivslab.instruments.contracts import (
    Currency,
    EquityContract,
    Exchange,
    ExchangeSegment,
    ExerciseStyle,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)
from derivslab.instruments.registry import InstrumentRegistry
from derivslab.market_data.base import PartialSnapshotError, QuoteUnavailableError
from derivslab.market_data.providers.profitpro_rtd import ProfitProRTDProvider

# ---------------------------------------------------------------------------
# Instruments to quote — adjust as needed
# ---------------------------------------------------------------------------

EQUITIES: list[str] = [
    "PETR4",
]

OPTIONS: list[tuple[str, str, OptionType, float, date]] = [
    # (ticker,    underlying,  type,            strike,  expiry)
    ("PETRA201",  "PETR4",    OptionType.CALL,  20.13,  date(2027, 1, 15)),
    ("PETRM201",  "PETR4",    OptionType.PUT,   20.13,  date(2027, 1, 15)),
]


# ---------------------------------------------------------------------------
# Build registry
# ---------------------------------------------------------------------------

def build_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()

    for ticker in EQUITIES:
        registry.add(EquityContract(
            instrument_id=ticker,
            currency=Currency.BRL,
            ticker=ticker,
            exchange=Exchange.B3,
            exchange_segment=ExchangeSegment.BOVESPA,
        ))

    for ticker, underlying, opt_type, strike, expiry in OPTIONS:
        registry.add(VanillaOptionContract(
            instrument_id=ticker,
            currency=Currency.BRL,
            ticker=ticker,
            underlying=underlying,
            underlying_asset_class=UnderlyingAssetClass.EQUITY,
            option_type=opt_type,
            style=ExerciseStyle.AMERICAN,
            strike=strike,
            expiry=expiry,
            exchange=Exchange.B3,
            exchange_segment=ExchangeSegment.BOVESPA,
        ))

    return registry


# ---------------------------------------------------------------------------
# Formatted quote table
# ---------------------------------------------------------------------------

def _format_last(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "  —"


def _print_table_header() -> None:
    print(f"\n{'SYMBOL':<12} {'BID':>10} {'ASK':>10} {'LAST':>10} {'MID':>10}  TIMESTAMP")
    print("-" * 72)


def _print_quote_row(q: object) -> None:  # type: ignore[type-arg]
    from derivslab.market_data.base import Quote
    assert isinstance(q, Quote)
    print(
        f"{q.symbol:<12} "
        f"{q.bid:>10.4f} "
        f"{q.ask:>10.4f} "
        f"{_format_last(q.last):>10} "
        f"{q.mid:>10.4f}  "
        f"{q.timestamp.strftime('%H:%M:%S %Z')}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    all_symbols = EQUITIES + [ticker for ticker, *_ in OPTIONS]

    print(f"Registered instruments : {len(all_symbols)}")
    print(f"Symbols                : {', '.join(all_symbols)}")

    registry = build_registry()

    print("\nConnecting to ProfitPro RTD via Excel...")
    try:
        provider = ProfitProRTDProvider(registry=registry, settle_time=2.0)
    except Exception as exc:
        print(f"\n✗ Connection failed: {exc}")
        print("  Make sure Excel is open with rtd_feed.xlsx and ProfitPro RTD is active.")
        return

    print("Connected. Fetching quotes...\n")

    try:
        snapshot = provider.get_quotes(all_symbols)
        print(f"Snapshot collected at : {snapshot.collected_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Quotes received       : {len(snapshot)} / {len(all_symbols)}")

        _print_table_header()
        for symbol in all_symbols:
            q = snapshot.get(symbol)
            if q is None:
                print(f"{symbol:<12}  ✗  not found in snapshot")
                continue
            _print_quote_row(q)

    except PartialSnapshotError as exc:
        print(
            f"⚠  Partial snapshot — {len(exc.snapshot)} quote(s) received, "
            f"{len(exc.unavailable)} failure(s)."
        )

        if exc.snapshot.quotes:
            _print_table_header()
            for q in exc.snapshot.quotes.values():
                _print_quote_row(q)

        if exc.unavailable:
            print("\nFailures:")
            for symbol, reason in exc.unavailable.items():
                print(f"  ✗  {symbol}: {reason}")


if __name__ == "__main__":
    main()