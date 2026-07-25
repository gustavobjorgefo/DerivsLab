"""
Tests for derivslab.instruments.registry.

Coverage
--------
InstrumentRegistry
    - add() stores a contract retrievable by get().
    - get() returns None for unknown symbols.
    - Duplicate add() silently overwrites the existing entry.
    - load() registers a list of contracts in a single call.
    - __contains__ works with the `in` operator.
    - __len__ reflects the number of registered contracts.
    - __iter__ yields all instrument_ids.
    - all_contracts() returns a snapshot of all registered contracts.
    - __repr__ includes the count.
"""

from __future__ import annotations

from datetime import date

import pytest

from derivslab.instruments.contracts import (
    Currency,
    DayCountConvention,
    EquityContract,
    Exchange,
    ExchangeSegment,
    ExerciseStyle,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)
from derivslab.instruments.registry import InstrumentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_equity(ticker: str = "PETR4") -> EquityContract:
    return EquityContract(
        instrument_id=ticker,
        currency=Currency.BRL,
        ticker=ticker,
        exchange=Exchange.B3,
        exchange_segment=ExchangeSegment.BOVESPA,
    )


def make_option(
    ticker: str = "PETRA201",
    underlying: str = "PETR4",
    strike: float = 20.13,
) -> VanillaOptionContract:
    return VanillaOptionContract(
        instrument_id=ticker,
        currency=Currency.BRL,
        ticker=ticker,
        underlying=underlying,
        underlying_asset_class=UnderlyingAssetClass.EQUITY,
        option_type=OptionType.CALL,
        style=ExerciseStyle.AMERICAN,
        strike=strike,
        expiry=date(2027, 1, 15),
        exchange=Exchange.B3,
        exchange_segment=ExchangeSegment.BOVESPA,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInstrumentRegistryAdd:

    def test_add_and_get_returns_same_contract(self) -> None:
        registry = InstrumentRegistry()
        contract = make_equity()
        registry.add(contract)
        assert registry.get("PETR4") is contract

    def test_get_returns_none_for_unknown_symbol(self) -> None:
        registry = InstrumentRegistry()
        assert registry.get("UNKNOWN") is None

    def test_duplicate_add_overwrites_existing(self) -> None:
        registry = InstrumentRegistry()
        first = make_equity("PETR4")
        second = make_equity("PETR4")
        registry.add(first)
        registry.add(second)
        assert registry.get("PETR4") is second

    def test_add_multiple_different_contracts(self) -> None:
        registry = InstrumentRegistry()
        equity = make_equity("PETR4")
        option = make_option("PETRA201")
        registry.add(equity)
        registry.add(option)
        assert registry.get("PETR4") is equity
        assert registry.get("PETRA201") is option


class TestInstrumentRegistryLoad:

    def test_load_registers_all_contracts(self) -> None:
        registry = InstrumentRegistry()
        contracts = [make_equity("PETR4"), make_option("PETRA201"), make_option("PETRM201", strike=20.13)]
        registry.load(contracts)
        assert registry.get("PETR4") is not None
        assert registry.get("PETRA201") is not None
        assert registry.get("PETRM201") is not None

    def test_load_empty_list_is_noop(self) -> None:
        registry = InstrumentRegistry()
        registry.load([])
        assert len(registry) == 0

    def test_load_duplicate_last_wins(self) -> None:
        registry = InstrumentRegistry()
        first = make_equity("PETR4")
        second = make_equity("PETR4")
        registry.load([first, second])
        assert registry.get("PETR4") is second


class TestInstrumentRegistryDunder:

    def test_contains_returns_true_for_registered_symbol(self) -> None:
        registry = InstrumentRegistry()
        registry.add(make_equity("PETR4"))
        assert "PETR4" in registry

    def test_contains_returns_false_for_unknown_symbol(self) -> None:
        registry = InstrumentRegistry()
        assert "PETR4" not in registry

    def test_len_reflects_count(self) -> None:
        registry = InstrumentRegistry()
        registry.add(make_equity("PETR4"))
        registry.add(make_option("PETRA201"))
        assert len(registry) == 2

    def test_len_zero_for_empty_registry(self) -> None:
        assert len(InstrumentRegistry()) == 0

    def test_iter_yields_all_instrument_ids(self) -> None:
        registry = InstrumentRegistry()
        registry.add(make_equity("PETR4"))
        registry.add(make_option("PETRA201"))
        assert set(registry) == {"PETR4", "PETRA201"}

    def test_repr_contains_count(self) -> None:
        registry = InstrumentRegistry()
        registry.add(make_equity("PETR4"))
        assert "1" in repr(registry)


class TestInstrumentRegistryAllContracts:

    def test_all_contracts_returns_all(self) -> None:
        registry = InstrumentRegistry()
        equity = make_equity("PETR4")
        option = make_option("PETRA201")
        registry.add(equity)
        registry.add(option)
        all_c = registry.all_contracts()
        assert len(all_c) == 2
        assert equity in all_c
        assert option in all_c

    def test_all_contracts_returns_empty_list_when_empty(self) -> None:
        registry = InstrumentRegistry()
        assert registry.all_contracts() == []

    def test_all_contracts_is_snapshot_not_reference(self) -> None:
        registry = InstrumentRegistry()
        registry.add(make_equity("PETR4"))
        snapshot = registry.all_contracts()
        registry.add(make_option("PETRA201"))
        # snapshot should not have grown
        assert len(snapshot) == 1