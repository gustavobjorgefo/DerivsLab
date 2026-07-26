"""
Tests for derivslab.market_data.base.

Coverage
--------
Quote
    - Required fields are stored correctly.
    - Optional fields default to None.
    - mid property returns (bid + ask) / 2.
    - frozen=True prevents mutation after construction.
    - timestamp is preserved as-is (timezone-aware).

MarketSnapshot
    - Construction from a dict of quotes.
    - get() returns correct Quote or None for unknown symbol.
    - symbols property lists all keys.
    - is_complete() returns True only when all expected symbols are present.
    - __len__ reflects the number of quotes.
    - __repr__ includes symbol list and collected_at.

QuoteUnavailableError
    - Carries symbol and reason as attributes.
    - Default reason falls back to UNAVAILABLE_REASON_UNKNOWN.
    - str() includes symbol and reason.

PartialSnapshotError
    - Carries partial snapshot and unavailable mapping as attributes.
    - str() reports counts and failed symbols.

MarketDataProvider (via FakeProvider stub)
    - ABC cannot be instantiated without implementing abstract methods.
    - get_mid() delegates to get_quote() and returns quote.mid.
    - Concrete stub that always fails raises QuoteUnavailableError from get_mid.
"""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

from derivslab.market_data.base import (
    MarketDataProvider,
    MarketSnapshot,
    MarketDataError,
    PartialSnapshotError,
    Quote,
    QuoteUnavailableError,
    UNAVAILABLE_REASON_UNKNOWN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_timestamp() -> datetime:
    return datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


def make_quote(
    symbol: str = "PETR4",
    bid: float = 36.10,
    ask: float = 36.20,
    *,
    last: float | None = 36.15,
    bid_size: float | None = None,
    ask_size: float | None = None,
    volume: float | None = None,
    timestamp: datetime | None = None,
) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        bid_size=bid_size,
        ask_size=ask_size,
        volume=volume,
        timestamp=timestamp or make_timestamp(),
    )


def make_snapshot(symbols: list[str] | None = None) -> MarketSnapshot:
    symbols = symbols or ["PETR4", "PETRA201", "PETRM201"]
    quotes = {s: make_quote(symbol=s) for s in symbols}
    return MarketSnapshot(quotes=quotes, collected_at=make_timestamp())


# ---------------------------------------------------------------------------
# Stub providers — used to test ABC and get_mid
# ---------------------------------------------------------------------------


class AlwaysSuccessProvider(MarketDataProvider):
    """Returns a hardcoded quote for any symbol."""

    def get_quote(self, symbol: str) -> Quote:
        return make_quote(symbol=symbol, bid=10.00, ask=10.20)

    def get_quotes(self, symbols: list[str]) -> MarketSnapshot:
        quotes = {s: self.get_quote(s) for s in symbols}
        return MarketSnapshot(quotes=quotes, collected_at=make_timestamp())


class AlwaysFailProvider(MarketDataProvider):
    """Always raises QuoteUnavailableError."""

    def get_quote(self, symbol: str) -> Quote:
        raise QuoteUnavailableError(symbol, "provider offline")

    def get_quotes(self, symbols: list[str]) -> MarketSnapshot:
        raise MarketDataError("provider offline")


# ---------------------------------------------------------------------------
# Quote tests
# ---------------------------------------------------------------------------


class TestQuote:

    def test_required_fields_stored(self) -> None:
        ts = make_timestamp()
        quote = Quote(symbol="PETR4", bid=36.10, ask=36.20, timestamp=ts)
        assert quote.symbol == "PETR4"
        assert quote.bid == 36.10
        assert quote.ask == 36.20
        assert quote.timestamp is ts

    def test_optional_fields_default_to_none(self) -> None:
        quote = Quote(symbol="PETR4", bid=36.10, ask=36.20, timestamp=make_timestamp())
        assert quote.last is None
        assert quote.bid_size is None
        assert quote.ask_size is None
        assert quote.volume is None

    def test_optional_fields_stored_when_provided(self) -> None:
        quote = make_quote(last=36.15, bid_size=1000.0, ask_size=500.0, volume=250_000.0)
        assert quote.last == 36.15
        assert quote.bid_size == 1000.0
        assert quote.ask_size == 500.0
        assert quote.volume == 250_000.0

    def test_last_can_be_none_for_illiquid_options(self) -> None:
        quote = make_quote(symbol="PETRA201", last=None)
        assert quote.last is None

    def test_mid_is_average_of_bid_and_ask(self) -> None:
        quote = make_quote(bid=10.00, ask=10.20)
        assert quote.mid == pytest.approx(10.10)

    def test_mid_exact_when_bid_equals_ask(self) -> None:
        quote = make_quote(bid=20.00, ask=20.00)
        assert quote.mid == pytest.approx(20.00)

    def test_frozen_prevents_mutation(self) -> None:
        quote = make_quote()
        with pytest.raises(FrozenInstanceError):
            quote.bid = 99.99  # type: ignore[misc]

    def test_frozen_prevents_mutation_of_optional_field(self) -> None:
        quote = make_quote()
        with pytest.raises(FrozenInstanceError):
            quote.last = 0.0  # type: ignore[misc]

    def test_timestamp_preserved_with_timezone(self) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        quote = Quote(symbol="PETR4", bid=1.0, ask=1.1, timestamp=ts)
        assert quote.timestamp.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# MarketSnapshot tests
# ---------------------------------------------------------------------------


class TestMarketSnapshot:

    def test_construction_with_quotes(self) -> None:
        snapshot = make_snapshot(["PETR4", "PETRA201"])
        assert len(snapshot) == 2

    def test_get_returns_quote_for_known_symbol(self) -> None:
        snapshot = make_snapshot(["PETR4"])
        result = snapshot.get("PETR4")
        assert result is not None
        assert result.symbol == "PETR4"

    def test_get_returns_none_for_unknown_symbol(self) -> None:
        snapshot = make_snapshot(["PETR4"])
        assert snapshot.get("VALE3") is None

    def test_symbols_lists_all_keys(self) -> None:
        symbols = ["PETR4", "PETRA201", "PETRM201"]
        snapshot = make_snapshot(symbols)
        assert sorted(snapshot.symbols) == sorted(symbols)

    def test_is_complete_true_when_all_present(self) -> None:
        snapshot = make_snapshot(["PETR4", "PETRA201"])
        assert snapshot.is_complete(["PETR4", "PETRA201"]) is True

    def test_is_complete_true_for_subset(self) -> None:
        snapshot = make_snapshot(["PETR4", "PETRA201", "PETRM201"])
        assert snapshot.is_complete(["PETR4"]) is True

    def test_is_complete_false_when_symbol_missing(self) -> None:
        snapshot = make_snapshot(["PETR4"])
        assert snapshot.is_complete(["PETR4", "PETRA201"]) is False

    def test_is_complete_false_for_empty_snapshot(self) -> None:
        snapshot = MarketSnapshot(quotes={}, collected_at=make_timestamp())
        assert snapshot.is_complete(["PETR4"]) is False

    def test_is_complete_true_for_empty_expected(self) -> None:
        snapshot = make_snapshot(["PETR4"])
        assert snapshot.is_complete([]) is True

    def test_len_reflects_quote_count(self) -> None:
        snapshot = make_snapshot(["PETR4", "PETRA201", "PETRM201"])
        assert len(snapshot) == 3

    def test_len_zero_for_empty_snapshot(self) -> None:
        snapshot = MarketSnapshot(quotes={}, collected_at=make_timestamp())
        assert len(snapshot) == 0

    def test_repr_contains_symbol_list(self) -> None:
        snapshot = make_snapshot(["PETR4"])
        assert "PETR4" in repr(snapshot)

    def test_repr_contains_collected_at(self) -> None:
        snapshot = make_snapshot(["PETR4"])
        assert snapshot.collected_at.isoformat() in repr(snapshot)

    def test_collected_at_is_stored(self) -> None:
        ts = make_timestamp()
        snapshot = MarketSnapshot(quotes={}, collected_at=ts)
        assert snapshot.collected_at == ts


# ---------------------------------------------------------------------------
# QuoteUnavailableError tests
# ---------------------------------------------------------------------------


class TestQuoteUnavailableError:

    def test_symbol_attribute(self) -> None:
        error = QuoteUnavailableError("PETRA201")
        assert error.symbol == "PETRA201"

    def test_default_reason(self) -> None:
        error = QuoteUnavailableError("PETRA201")
        assert error.reason == UNAVAILABLE_REASON_UNKNOWN

    def test_custom_reason(self) -> None:
        error = QuoteUnavailableError("PETRA201", "no market data returned")
        assert error.reason == "no market data returned"

    def test_str_includes_symbol(self) -> None:
        error = QuoteUnavailableError("PETRA201", "timeout")
        assert "PETRA201" in str(error)

    def test_str_includes_reason(self) -> None:
        error = QuoteUnavailableError("PETRA201", "timeout")
        assert "timeout" in str(error)

    def test_is_subclass_of_market_data_error(self) -> None:
        error = QuoteUnavailableError("PETR4")
        assert isinstance(error, MarketDataError)

    def test_is_catchable_as_exception(self) -> None:
        with pytest.raises(QuoteUnavailableError):
            raise QuoteUnavailableError("PETR4", "test")


# ---------------------------------------------------------------------------
# PartialSnapshotError tests
# ---------------------------------------------------------------------------


class TestPartialSnapshotError:

    def _make_error(self) -> PartialSnapshotError:
        snapshot = make_snapshot(["PETR4"])
        unavailable = {"PETRA201": "no data", "PETRM201": "timeout"}
        return PartialSnapshotError(snapshot=snapshot, unavailable=unavailable)

    def test_snapshot_attribute(self) -> None:
        error = self._make_error()
        assert isinstance(error.snapshot, MarketSnapshot)
        assert "PETR4" in error.snapshot.symbols

    def test_unavailable_attribute(self) -> None:
        error = self._make_error()
        assert "PETRA201" in error.unavailable
        assert "PETRM201" in error.unavailable

    def test_unavailable_carries_reasons(self) -> None:
        error = self._make_error()
        assert error.unavailable["PETRA201"] == "no data"
        assert error.unavailable["PETRM201"] == "timeout"

    def test_str_includes_failed_symbols(self) -> None:
        error = self._make_error()
        assert "PETRA201" in str(error)
        assert "PETRM201" in str(error)

    def test_str_includes_success_count(self) -> None:
        error = self._make_error()
        assert "1" in str(error)

    def test_is_subclass_of_market_data_error(self) -> None:
        error = self._make_error()
        assert isinstance(error, MarketDataError)

    def test_partial_snapshot_is_accessible_from_except_block(self) -> None:
        snapshot = make_snapshot(["PETR4"])
        unavailable = {"PETRA201": "timeout"}

        with pytest.raises(PartialSnapshotError) as exc_info:
            raise PartialSnapshotError(snapshot=snapshot, unavailable=unavailable)

        assert exc_info.value.snapshot.get("PETR4") is not None
        assert exc_info.value.unavailable["PETRA201"] == "timeout"


# ---------------------------------------------------------------------------
# MarketDataProvider (ABC + concrete helpers) tests
# ---------------------------------------------------------------------------


class TestMarketDataProviderABC:

    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            MarketDataProvider()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
        class IncompleteProvider(MarketDataProvider):
            def get_quote(self, symbol: str) -> Quote:
                return make_quote(symbol=symbol)

            # get_quotes not implemented

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self) -> None:
        provider = AlwaysSuccessProvider()
        assert isinstance(provider, MarketDataProvider)


class TestGetMid:

    def test_get_mid_delegates_to_get_quote(self) -> None:
        provider = AlwaysSuccessProvider()
        mid = provider.get_mid("PETR4")
        assert mid == pytest.approx(10.10)

    def test_get_mid_propagates_quote_unavailable_error(self) -> None:
        provider = AlwaysFailProvider()
        with pytest.raises(QuoteUnavailableError) as exc_info:
            provider.get_mid("PETRA201")
        assert exc_info.value.symbol == "PETRA201"


# ---------------------------------------------------------------------------
# Consumer pattern — snapshot partial handling
# ---------------------------------------------------------------------------


class TestConsumerPartialSnapshotPattern:
    """
    Verify the pattern consumers are expected to use when handling partial
    snapshots from get_quotes.
    """

    def test_consumer_can_proceed_with_partial_snapshot(self) -> None:
        partial_snapshot = make_snapshot(["PETR4"])
        unavailable = {"PETRA201": "no data"}

        received_snapshot: MarketSnapshot | None = None

        try:
            raise PartialSnapshotError(
                snapshot=partial_snapshot,
                unavailable=unavailable,
            )
        except PartialSnapshotError as exc:
            received_snapshot = exc.snapshot

        assert received_snapshot is not None
        assert received_snapshot.get("PETR4") is not None

    def test_consumer_detects_incomplete_snapshot_for_required_symbols(self) -> None:
        required = ["PETR4", "PETRA201"]
        partial_snapshot = make_snapshot(["PETR4"])
        unavailable = {"PETRA201": "timeout"}

        try:
            raise PartialSnapshotError(
                snapshot=partial_snapshot,
                unavailable=unavailable,
            )
        except PartialSnapshotError as exc:
            is_ready = exc.snapshot.is_complete(required)

        assert is_ready is False
