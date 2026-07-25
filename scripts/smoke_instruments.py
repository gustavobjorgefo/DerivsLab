"""Manual smoke test: instantiate instruments and print them.

Run directly with:
    python scripts/smoke_instruments.py

Not a pytest test — a quick, throwaway way to eyeball that an instrument
builds correctly and reads back sensibly.
"""

from __future__ import annotations

from datetime import date

from derivslab.instruments.contracts import (
    Currency,
    EquityContract,
    ExerciseStyle,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)
from derivslab.instruments.equity import EquityInstrument
from derivslab.instruments.vanilla import VanillaOption


def main() -> None:
    equity_contract = EquityContract(
        instrument_id="PETR4",
        currency=Currency.BRL,
        ticker="PETR4",
    )
    equity = EquityInstrument(equity_contract)
    print(equity)

    option_contract = VanillaOptionContract(
        instrument_id="PETRA123",
        currency=Currency.BRL,
        ticker="PETRA123",
        underlying="PETR4",
        underlying_asset_class=UnderlyingAssetClass.EQUITY,
        option_type=OptionType.CALL,
        style=ExerciseStyle.EUROPEAN,
        strike=35.0,
        expiry=date(2026, 12, 18),
    )
    call = VanillaOption(option_contract)
    print(call)


if __name__ == "__main__":
    main()