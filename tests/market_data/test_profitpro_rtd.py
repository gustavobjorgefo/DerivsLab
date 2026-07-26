"""
Tests for derivslab.market_data.providers.profitpro_rtd.

Strategy
--------
``ProfitProRTDProvider`` depends on xlwings to communicate with a running
Excel instance, so no test here opens Excel or connects to a real RTD server.
Instead:

- ``xw`` is patched at the module level using ``unittest.mock.patch`` so
  the provider never tries to reach Excel.
- A ``FakeSheet`` stub captures every write (symbol labels, RTD formulas,
  clear calls) and returns configurable read values, letting tests verify
  the full write → wait → read → clear → translate pipeline without I/O.
- ``settle_time=0.0`` on every provider fixture eliminates ``time.sleep``
  overhead without additional patching.

Coverage
--------
FakeSheet / helpers
    - _is_rtd_error correctly classifies None, Excel error strings, and floats.
    - _to_optional_float returns None on errors and float otherwise.
    - _rtd_formula produces the expected English-comma syntax.

_resolve_rtd_name
    - Symbol not in registry → QuoteUnavailableError with descriptive reason.
    - Contract with no exchange_segment → QuoteUnavailableError.
    - Contract with unsupported segment → QuoteUnavailableError.
    - BOVESPA segment → ``_B_0`` suffix.
    - BMF segment → ``_F_0`` suffix.

_row_to_quote
    - Valid row → Quote with correct field mapping and timestamp.
    - Bid RTD error → QuoteUnavailableError.
    - Ask RTD error → QuoteUnavailableError.
    - Last RTD error → last=None (optional field, not fatal).
    - Optional size / volume RTD error → None fields.

ProfitProRTDProvider.__init__
    - Connects to the configured workbook/sheet on construction.
    - Writes the static header row to A1.
    - Raises MarketDataError when Excel is not reachable.

get_quote
    - Delegates to get_quotes and returns the single Quote.
    - Propagates QuoteUnavailableError when symbol is unavailable.

get_quotes
    - All symbols succeed → complete MarketSnapshot, no exception.
    - Registry miss for one symbol → PartialSnapshotError with that symbol
      in unavailable; successful symbols still appear in the snapshot.
    - RTD error on bid/ask → symbol in unavailable, others succeed.
    - Excel communication error during batch → all symbols in batch go to
      unavailable; sheet is still cleared (finally block).
    - Symbols processed in batches: 3 symbols with batch_size=2 produces
      2 separate write→read→clear cycles.
    - Sheet is always cleared after a successful read.
    - Sheet is always cleared even when read raises.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
from derivslab.market_data.base import (
    MarketDataError,
    MarketSnapshot,
    PartialSnapshotError,
    Quote,
    QuoteUnavailableError,
)
from derivslab.market_data.providers.profitpro_rtd import (
    RTD_SERVER,
    ProfitProRTDProvider,
    _RTD_ATTRIBUTES,
)


# ---------------------------------------------------------------------------
# Fake xlwings sheet
# ---------------------------------------------------------------------------


class _FakeRange:
    """Stub for xlwings Range — captures writes, returns configured reads."""

    def __init__(self, sheet: "FakeSheet", address: str) -> None:
        self._sheet = sheet
        self._address = address

    # value -------------------------------------------------------------------

    @property
    def value(self) -> Any:
        if self._sheet._read_raises is not None:
            raise self._sheet._read_raises
        if self._sheet._read_responses:
            return self._sheet._read_responses.popleft()
        return None

    @value.setter
    def value(self, v: Any) -> None:
        if v is None:
            self._sheet._clear_count += 1
        else:
            self._sheet._value_writes.append((self._address, v))

    # formula -----------------------------------------------------------------

    @property
    def formula(self) -> Any:
        return None

    @formula.setter
    def formula(self, v: Any) -> None:
        self._sheet._formula_writes.append((self._address, v))


class FakeSheet:
    """Minimal xlwings Sheet stub.

    Configure responses with ``push_read_response`` before the provider
    reads the sheet.  Each push is consumed by one ``.value`` read, in
    FIFO order — one per batch.
    """

    def __init__(self) -> None:
        self._read_responses: deque[Any] = deque()
        self._read_raises: Exception | None = None
        self._value_writes: list[tuple[str, Any]] = []
        self._formula_writes: list[tuple[str, Any]] = []
        self._clear_count: int = 0

    def push_read_response(self, value: Any) -> None:
        """Queue a value to be returned by the next .value read."""
        self._read_responses.append(value)

    def set_read_raises(self, exc: Exception) -> None:
        """Make every subsequent .value read raise *exc*."""
        self._read_raises = exc

    def range(self, address: str) -> _FakeRange:
        return _FakeRange(self, address)

    # --- inspection helpers --------------------------------------------------

    @property
    def formula_write_count(self) -> int:
        return len(self._formula_writes)

    @property
    def clear_count(self) -> int:
        return self._clear_count

    @property
    def header_written(self) -> bool:
        """True if A1 was written with a list value (the header row)."""
        return any(addr == "A1" for addr, _ in self._value_writes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_equity(ticker: str = "PETR4") -> EquityContract:
    return EquityContract(
        instrument_id=ticker,
        currency=Currency.BRL,
        ticker=ticker,
        exchange=Exchange.B3,
        exchange_segment=ExchangeSegment.BOVESPA,
    )


def _make_option(
    ticker: str = "PETRA201",
    underlying: str = "PETR4",
    segment: ExchangeSegment = ExchangeSegment.BOVESPA,
) -> VanillaOptionContract:
    return VanillaOptionContract(
        instrument_id=ticker,
        currency=Currency.BRL,
        ticker=ticker,
        underlying=underlying,
        underlying_asset_class=UnderlyingAssetClass.EQUITY,
        option_type=OptionType.CALL,
        style=ExerciseStyle.AMERICAN,
        strike=20.13,
        expiry=date(2027, 1, 15),
        exchange=Exchange.B3,
        exchange_segment=segment,
    )


def _make_registry(*contracts: Any) -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for c in contracts:
        registry.add(c)
    return registry


def _raw_row(
    last: float | str | None = 36.15,
    bid: float | str | None = 36.10,
    ask: float | str | None = 36.20,
    bid_size: float | str | None = 1000.0,
    ask_size: float | str | None = 500.0,
    volume: float | str | None = 250_000.0,
) -> list[Any]:
    return [last, bid, ask, bid_size, ask_size, volume]


@pytest.fixture()
def fake_sheet() -> FakeSheet:
    return FakeSheet()


@pytest.fixture()
def patch_xlwings(fake_sheet: FakeSheet):
    """Patch the xw module used by profitpro_rtd so no Excel is opened."""
    with patch("derivslab.market_data.providers.profitpro_rtd.xw") as mock_xw:
        (mock_xw.apps.active.books.__getitem__.return_value.sheets.__getitem__.return_value) = (
            fake_sheet
        )
        yield mock_xw


@pytest.fixture()
def registry() -> InstrumentRegistry:
    return _make_registry(
        _make_equity("PETR4"),
        _make_option("PETRA201"),
        _make_option("PETRM201"),
    )


@pytest.fixture()
def provider(
    fake_sheet: FakeSheet, registry: InstrumentRegistry, patch_xlwings: Any
) -> ProfitProRTDProvider:
    return ProfitProRTDProvider(registry=registry, settle_time=0.0)


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


class TestIsRtdError:

    def test_none_is_error(self) -> None:
        assert ProfitProRTDProvider._is_rtd_error(None) is True

    def test_hash_na_is_error(self) -> None:
        assert ProfitProRTDProvider._is_rtd_error("#N/A") is True

    def test_hash_value_is_error(self) -> None:
        assert ProfitProRTDProvider._is_rtd_error("#VALUE!") is True

    def test_any_hash_prefix_is_error(self) -> None:
        assert ProfitProRTDProvider._is_rtd_error("#REF!") is True

    def test_float_is_not_error(self) -> None:
        assert ProfitProRTDProvider._is_rtd_error(36.15) is False

    def test_zero_is_not_error(self) -> None:
        assert ProfitProRTDProvider._is_rtd_error(0.0) is False

    def test_integer_is_not_error(self) -> None:
        assert ProfitProRTDProvider._is_rtd_error(100) is False


class TestToOptionalFloat:

    def test_returns_none_for_none(self) -> None:
        assert ProfitProRTDProvider._to_optional_float(None) is None

    def test_returns_none_for_excel_error_string(self) -> None:
        assert ProfitProRTDProvider._to_optional_float("#N/A") is None

    def test_converts_float(self) -> None:
        assert ProfitProRTDProvider._to_optional_float(36.15) == pytest.approx(36.15)

    def test_converts_int_to_float(self) -> None:
        result = ProfitProRTDProvider._to_optional_float(100)
        assert isinstance(result, float)
        assert result == pytest.approx(100.0)


class TestRtdFormula:

    def test_formula_starts_with_equals_rtd(self) -> None:
        formula = ProfitProRTDProvider._rtd_formula("PETRA201_B_0", "ULT")
        assert formula.startswith("=RTD(")

    def test_formula_contains_server_name(self) -> None:
        formula = ProfitProRTDProvider._rtd_formula("PETRA201_B_0", "ULT")
        assert RTD_SERVER in formula

    def test_formula_contains_rtd_name(self) -> None:
        formula = ProfitProRTDProvider._rtd_formula("PETRA201_B_0", "ULT")
        assert "PETRA201_B_0" in formula

    def test_formula_contains_attribute(self) -> None:
        formula = ProfitProRTDProvider._rtd_formula("PETRA201_B_0", "OCP")
        assert "OCP" in formula

    def test_formula_uses_comma_separator(self) -> None:
        # English syntax — locale translation handled by xlwings .formula
        formula = ProfitProRTDProvider._rtd_formula("PETRA201_B_0", "ULT")
        assert "," in formula


# ---------------------------------------------------------------------------
# _resolve_rtd_name
# ---------------------------------------------------------------------------


class TestResolveRtdName:

    def test_bovespa_gets_b0_suffix(self, provider: ProfitProRTDProvider) -> None:
        assert provider._resolve_rtd_name("PETRA201") == "PETRA201_B_0"

    def test_bmf_gets_f0_suffix(self, fake_sheet: FakeSheet, patch_xlwings: Any) -> None:
        registry = _make_registry(_make_option("DI1F26", segment=ExchangeSegment.BMF))
        p = ProfitProRTDProvider(registry=registry, settle_time=0.0)
        assert p._resolve_rtd_name("DI1F26") == "DI1F26_F_0"

    def test_symbol_not_in_registry_raises(self, provider: ProfitProRTDProvider) -> None:
        with pytest.raises(QuoteUnavailableError) as exc_info:
            provider._resolve_rtd_name("UNKNOWN")
        assert "not found in InstrumentRegistry" in exc_info.value.reason

    def test_no_exchange_segment_raises(self, fake_sheet: FakeSheet, patch_xlwings: Any) -> None:
        # Contract with no exchange_segment
        contract = EquityContract(
            instrument_id="PETR4",
            currency=Currency.BRL,
            ticker="PETR4",
        )
        registry = _make_registry(contract)
        p = ProfitProRTDProvider(registry=registry, settle_time=0.0)
        with pytest.raises(QuoteUnavailableError) as exc_info:
            p._resolve_rtd_name("PETR4")
        assert "exchange_segment not set" in exc_info.value.reason

    def test_symbol_attribute_set_on_error(self, provider: ProfitProRTDProvider) -> None:
        with pytest.raises(QuoteUnavailableError) as exc_info:
            provider._resolve_rtd_name("MISSING")
        assert exc_info.value.symbol == "MISSING"


# ---------------------------------------------------------------------------
# _row_to_quote
# ---------------------------------------------------------------------------


class TestRowToQuote:

    def _ts(self) -> datetime:
        return datetime(2025, 7, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_valid_row_maps_all_fields(self, provider: ProfitProRTDProvider) -> None:
        ts = self._ts()
        quote = provider._row_to_quote("PETRA201", _raw_row(), ts)
        assert quote.symbol == "PETRA201"
        assert quote.bid == pytest.approx(36.10)
        assert quote.ask == pytest.approx(36.20)
        assert quote.last == pytest.approx(36.15)
        assert quote.bid_size == pytest.approx(1000.0)
        assert quote.ask_size == pytest.approx(500.0)
        assert quote.volume == pytest.approx(250_000.0)
        assert quote.timestamp == ts

    def test_bid_error_raises_quote_unavailable(self, provider: ProfitProRTDProvider) -> None:
        with pytest.raises(QuoteUnavailableError) as exc_info:
            provider._row_to_quote("PETRA201", _raw_row(bid="#N/A"), self._ts())
        assert exc_info.value.symbol == "PETRA201"
        assert "bid" in exc_info.value.reason

    def test_ask_error_raises_quote_unavailable(self, provider: ProfitProRTDProvider) -> None:
        with pytest.raises(QuoteUnavailableError) as exc_info:
            provider._row_to_quote("PETRA201", _raw_row(ask=None), self._ts())
        assert exc_info.value.symbol == "PETRA201"
        assert "ask" in exc_info.value.reason

    def test_last_error_sets_none(self, provider: ProfitProRTDProvider) -> None:
        quote = provider._row_to_quote("PETRA201", _raw_row(last="#N/A"), self._ts())
        assert quote.last is None

    def test_bid_size_error_sets_none(self, provider: ProfitProRTDProvider) -> None:
        quote = provider._row_to_quote("PETRA201", _raw_row(bid_size=None), self._ts())
        assert quote.bid_size is None

    def test_ask_size_error_sets_none(self, provider: ProfitProRTDProvider) -> None:
        quote = provider._row_to_quote("PETRA201", _raw_row(ask_size="#VALUE!"), self._ts())
        assert quote.ask_size is None

    def test_volume_error_sets_none(self, provider: ProfitProRTDProvider) -> None:
        quote = provider._row_to_quote("PETRA201", _raw_row(volume=None), self._ts())
        assert quote.volume is None


# ---------------------------------------------------------------------------
# __init__ / connection
# ---------------------------------------------------------------------------


class TestProviderInit:

    def test_writes_header_on_init(self, fake_sheet: FakeSheet, patch_xlwings: Any) -> None:
        registry = _make_registry(_make_equity())
        ProfitProRTDProvider(registry=registry, settle_time=0.0)
        assert fake_sheet.header_written

    def test_raises_market_data_error_when_excel_unavailable(self) -> None:
        with patch("derivslab.market_data.providers.profitpro_rtd.xw") as mock_xw:
            mock_xw.apps.active.books.__getitem__.side_effect = Exception("workbook not found")
            with pytest.raises(MarketDataError) as exc_info:
                ProfitProRTDProvider(registry=InstrumentRegistry(), settle_time=0.0)
            assert "workbook not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------


class TestGetQuote:

    def test_returns_quote_for_valid_symbol(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.push_read_response(_raw_row())  # single row → flat list
        quote = provider.get_quote("PETRA201")
        assert isinstance(quote, Quote)
        assert quote.symbol == "PETRA201"

    def test_raises_quote_unavailable_for_missing_symbol(
        self, provider: ProfitProRTDProvider
    ) -> None:
        with pytest.raises(QuoteUnavailableError) as exc_info:
            provider.get_quote("MISSING")
        assert exc_info.value.symbol == "MISSING"

    def test_raises_quote_unavailable_when_rtd_returns_error(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.push_read_response(_raw_row(bid="#N/A"))
        with pytest.raises(QuoteUnavailableError) as exc_info:
            provider.get_quote("PETRA201")
        assert exc_info.value.symbol == "PETRA201"


# ---------------------------------------------------------------------------
# get_quotes
# ---------------------------------------------------------------------------


class TestGetQuotes:

    def test_all_succeed_returns_complete_snapshot(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.push_read_response([_raw_row(bid=10.0, ask=10.2), _raw_row(bid=9.0, ask=9.2)])
        snapshot = provider.get_quotes(["PETRA201", "PETRM201"])
        assert isinstance(snapshot, MarketSnapshot)
        assert snapshot.is_complete(["PETRA201", "PETRM201"])

    def test_registry_miss_goes_to_unavailable(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.push_read_response(_raw_row())
        with pytest.raises(PartialSnapshotError) as exc_info:
            provider.get_quotes(["PETRA201", "MISSING"])
        assert "MISSING" in exc_info.value.unavailable
        assert exc_info.value.snapshot.get("PETRA201") is not None

    def test_rtd_bid_error_goes_to_unavailable(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.push_read_response(
            [
                _raw_row(bid=10.0, ask=10.2),
                _raw_row(bid="#N/A", ask=9.2),
            ]
        )
        with pytest.raises(PartialSnapshotError) as exc_info:
            provider.get_quotes(["PETRA201", "PETRM201"])
        assert "PETRM201" in exc_info.value.unavailable
        assert exc_info.value.snapshot.get("PETRA201") is not None

    def test_excel_error_marks_whole_batch_unavailable(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.set_read_raises(OSError("Excel COM error"))
        with pytest.raises(PartialSnapshotError) as exc_info:
            provider.get_quotes(["PETRA201", "PETRM201"])
        assert "PETRA201" in exc_info.value.unavailable
        assert "PETRM201" in exc_info.value.unavailable

    def test_clear_runs_after_successful_read(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.push_read_response(_raw_row())
        provider.get_quotes(["PETRA201"])
        assert fake_sheet.clear_count >= 1

    def test_clear_runs_even_after_failed_read(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.set_read_raises(OSError("COM error"))
        with pytest.raises(PartialSnapshotError):
            provider.get_quotes(["PETRA201"])
        assert fake_sheet.clear_count >= 1

    def test_batching_produces_multiple_write_cycles(
        self, fake_sheet: FakeSheet, patch_xlwings: Any
    ) -> None:
        # 3 symbols with batch_size=2 → 2 batches → 2 formula write cycles
        registry = _make_registry(
            _make_option("OPT1"),
            _make_option("OPT2"),
            _make_option("OPT3"),
        )
        p = ProfitProRTDProvider(registry=registry, settle_time=0.0, batch_size=2)

        # Push one response per batch
        fake_sheet.push_read_response([_raw_row(bid=1.0, ask=1.1), _raw_row(bid=2.0, ask=2.1)])
        fake_sheet.push_read_response(_raw_row(bid=3.0, ask=3.1))

        snapshot = p.get_quotes(["OPT1", "OPT2", "OPT3"])

        # 2 batches means 2 formula write events (B2:G... range per batch)
        assert fake_sheet.formula_write_count == 2
        assert snapshot.is_complete(["OPT1", "OPT2", "OPT3"])

    def test_partial_snapshot_accessible_from_exception(
        self, provider: ProfitProRTDProvider, fake_sheet: FakeSheet
    ) -> None:
        fake_sheet.push_read_response(_raw_row())
        try:
            provider.get_quotes(["PETRA201", "MISSING"])
        except PartialSnapshotError as exc:
            assert exc.snapshot.get("PETRA201") is not None
            assert exc.unavailable["MISSING"] != ""
