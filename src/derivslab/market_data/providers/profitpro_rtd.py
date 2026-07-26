"""ProfitPro RTD market data provider.

Collects market data by writing RTD formulas into a pre-configured Excel
worksheet via xlwings, waiting for the ProfitPro RTD server to populate
the values, reading the results as a block, and clearing the range
afterwards.

Pre-conditions (caller's responsibility — not validated by this provider)
-------------------------------------------------------------------------
- ProfitPro must be running and the RTD server must be active before
  instantiating this provider.
- Excel must be running with ``WORKBOOK_NAME`` (``"rtd_feed.xlsx"``)
  open and ``WORKSHEET_NAME`` (``"rtd_feed"``) present inside it.
- The ``InstrumentRegistry`` passed to the constructor must already be
  populated with contracts for all symbols that will be requested.

RTD formula syntax
------------------
Formulas are written using the English comma separator via xlwings'
``.formula`` property, which handles locale translation automatically::

    =RTD("RTDTrading.RTDServer",,"PETRA201_B_0","ULT")

If locale-related issues arise on a Brazilian Portuguese Excel instance,
switch the ``_write_batch`` helper to use ``.formula_local`` with
semicolons as separators.

RTD ticker naming
-----------------
The RTD ticker is composed as ``{symbol}{suffix}``, where the suffix
depends on the exchange segment of the instrument's contract::

    BOVESPA  →  _B_0   (equities, equity options, ETFs, FIIs)
    BMF      →  _F_0   (futures, FX, rates, commodities)

Worksheet layout (``rtd_feed`` sheet)
--------------------------------------
Row 1 is a static header written once at startup.  Data occupies rows 2
onwards for the duration of each batch, and is cleared immediately after
reading::

    A         B      C      D      E         F         G
    symbol    last   bid    ask    bid_size  ask_size  volume
    PETRA201  =RTD…  =RTD…  =RTD…  =RTD…    =RTD…    =RTD…
    PETRM201  =RTD…  =RTD…  =RTD…  =RTD…    =RTD…    =RTD…

Batch processing
----------------
Large symbol lists are split into batches of ``batch_size`` (default
``DEFAULT_BATCH_SIZE``). Each batch is written, read, and cleared before
the next one starts. The caller receives a single coherent
``MarketSnapshot`` that aggregates all batches; batching is an internal
implementation detail.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Final

import xlwings as xw

from derivslab.instruments.contracts import ExchangeSegment
from derivslab.instruments.registry import InstrumentRegistry
from derivslab.market_data.base import (
    MarketDataError,
    MarketDataProvider,
    MarketSnapshot,
    PartialSnapshotError,
    Quote,
    QuoteUnavailableError,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

WORKBOOK_NAME: Final[str] = "rtd_feed.xlsx"
WORKSHEET_NAME: Final[str] = "rtd_feed"
RTD_SERVER: Final[str] = "RTDTrading.RTDServer"
DEFAULT_SETTLE_TIME: Final[float] = 2.0
DEFAULT_BATCH_SIZE: Final[int] = 50

_DATA_START_ROW: Final[int] = 2
_HEADER: Final[list[str]] = [
    "symbol",
    "last",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "volume",
]

# RTD attribute names in the order they occupy columns B → G.
_RTD_ATTRIBUTES: Final[list[str]] = ["ULT", "OCP", "OVD", "VOC", "VOV", "VOL"]

# Mapping from ExchangeSegment to ProfitPro RTD ticker suffix.
# Segments for international exchanges are not yet used — add the
# corresponding ExchangeSegment members and uncomment when needed.
_RTD_SUFFIX: Final[dict[ExchangeSegment, str]] = {
    ExchangeSegment.BOVESPA: "_B_0",
    ExchangeSegment.BMF: "_F_0",
    # ExchangeSegment.DJ:     "_X_0",  # Dow Jones — implement when needed
    # ExchangeSegment.NYSE:   "_Y_0",  # NYSE       — implement when needed
    # ExchangeSegment.NASDAQ: "_N_0",  # NASDAQ     — implement when needed
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ProfitProRTDProvider(MarketDataProvider):
    """Market data provider backed by ProfitPro's RTD server via Excel.

    Implements ``MarketDataProvider`` (pull-only) by writing RTD formulas
    into a dedicated Excel worksheet, waiting for the values to populate,
    reading them back, and clearing the range. Large symbol lists are
    processed in batches to avoid overloading the RTD server.

    Parameters
    ----------
    registry : InstrumentRegistry
        Populated registry used to resolve the ``ExchangeSegment`` of each
        requested symbol — required to build the correct RTD ticker suffix.
        Symbols absent from the registry result in ``QuoteUnavailableError``
        (or ``PartialSnapshotError`` entries) rather than silent failures.
    settle_time : float
        Seconds to wait after writing RTD formulas before reading back the
        values. Increase if quotes are frequently returning stale or error
        values. Defaults to ``DEFAULT_SETTLE_TIME`` (2.0 s).
    batch_size : int
        Maximum number of symbols processed per Excel write-read-clear
        cycle. Defaults to ``DEFAULT_BATCH_SIZE`` (50).

    Raises
    ------
    MarketDataError
        At construction time, if the provider cannot connect to
        ``WORKBOOK_NAME`` / ``WORKSHEET_NAME`` in the active Excel instance.

    Notes
    -----
    The xlwings ``Sheet`` object is acquired once at construction and
    reused across calls. If Excel is closed and reopened during the
    lifetime of this provider, reinstantiate the provider.
    """

    def __init__(
        self,
        registry: InstrumentRegistry,
        settle_time: float = DEFAULT_SETTLE_TIME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._registry: InstrumentRegistry = registry
        self._settle_time: float = settle_time
        self._batch_size: int = batch_size
        self._sheet: xw.Sheet = self._connect()
        self._write_header()

    # --- MarketDataProvider interface ---------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        """Return the current quote for a single instrument.

        Delegates to ``get_quotes`` and extracts the single result.

        Parameters
        ----------
        symbol : str
            Instrument identifier in DerivsLab's internal vocabulary.

        Returns
        -------
        Quote

        Raises
        ------
        QuoteUnavailableError
            If the symbol is not in the registry, has no exchange segment,
            or if the RTD server returns an error value for bid or ask.
        """
        try:
            snapshot = self.get_quotes([symbol])
        except PartialSnapshotError as exc:
            reason = exc.unavailable.get(symbol, "unknown error in get_quotes")
            raise QuoteUnavailableError(symbol, reason) from exc

        quote = snapshot.get(symbol)
        if quote is None:
            # Defensive: should not happen if get_quotes succeeded without error.
            raise QuoteUnavailableError(symbol, "quote missing from snapshot")
        return quote

    def get_quotes(self, symbols: list[str]) -> MarketSnapshot:
        """Return a coherent snapshot of quotes for a list of instruments.

        Processes symbols in batches of ``batch_size``. Each batch is
        written to the Excel sheet, allowed to settle, read back, and
        cleared before the next batch starts. All results are merged into
        a single ``MarketSnapshot``.

        Parameters
        ----------
        symbols : list[str]
            Instrument identifiers in DerivsLab's internal vocabulary.

        Returns
        -------
        MarketSnapshot
            Snapshot containing one ``Quote`` per symbol when all quotes
            succeed.

        Raises
        ------
        PartialSnapshotError
            If one or more symbols could not be quoted — e.g. symbol not
            in registry, RTD error on bid/ask, or Excel communication
            failure for a batch. The exception carries the partial snapshot
            (successful quotes) and a per-symbol failure map.
        MarketDataError
            If a systemic failure (e.g. Excel becomes unresponsive) prevents
            any data from being collected.
        """
        quotes: dict[str, Quote] = {}
        unavailable: dict[str, str] = {}
        collected_at: datetime = datetime.now(tz=timezone.utc)

        # --- resolve RTD names upfront; isolate per-symbol registry failures
        valid_symbols: list[str] = []
        rtd_names: list[str] = []
        for symbol in symbols:
            try:
                rtd_names.append(self._resolve_rtd_name(symbol))
                valid_symbols.append(symbol)
            except QuoteUnavailableError as exc:
                unavailable[symbol] = exc.reason

        # --- process in batches
        for offset in range(0, len(valid_symbols), self._batch_size):
            batch_symbols = valid_symbols[offset : offset + self._batch_size]
            batch_rtd_names = rtd_names[offset : offset + self._batch_size]
            n = len(batch_symbols)

            raw_rows: list[list[object]] | None = None
            try:
                self._write_batch(batch_symbols, batch_rtd_names)
                time.sleep(self._settle_time)
                raw_rows = self._read_batch(n)
            except Exception as exc:
                for sym in batch_symbols:
                    unavailable[sym] = f"Excel communication error: {exc}"
            finally:
                # Always attempt to clear, even when the read failed, so the
                # next batch starts from a clean sheet.
                try:
                    self._clear_batch(n)
                except Exception:
                    pass

            if raw_rows is None:
                continue

            for symbol, row in zip(batch_symbols, raw_rows):
                try:
                    quotes[symbol] = self._row_to_quote(symbol, row, collected_at)
                except QuoteUnavailableError as exc:
                    unavailable[symbol] = exc.reason

        snapshot = MarketSnapshot(quotes=quotes, collected_at=collected_at)
        if unavailable:
            raise PartialSnapshotError(snapshot=snapshot, unavailable=unavailable)
        return snapshot

    # --- connection helpers -------------------------------------------------

    def _connect(self) -> xw.Sheet:
        """Connect to the open Excel workbook and return the RTD sheet.

        Raises
        ------
        MarketDataError
            If no Excel instance is running, ``WORKBOOK_NAME`` is not open,
            or ``WORKSHEET_NAME`` does not exist in the workbook.
        """
        app = xw.apps.active
        if app is None:
            raise MarketDataError(
                "No active Excel application found. Ensure Excel is running "
                "before instantiating ProfitProRTDProvider."
            )
        try:
            book: xw.Book = app.books[WORKBOOK_NAME]
            return book.sheets[WORKSHEET_NAME]
        except Exception as exc:
            raise MarketDataError(
                f"Cannot connect to '{WORKBOOK_NAME}' / '{WORKSHEET_NAME}'. "
                f"Ensure Excel is running with the workbook open and "
                f"ProfitPro RTD is active. Detail: {exc}"
            ) from exc

    def _write_header(self) -> None:
        """Write the static header row to row 1 of the RTD sheet."""
        self._sheet.range("A1").value = _HEADER

    # --- RTD resolution -----------------------------------------------------

    def _resolve_rtd_name(self, symbol: str) -> str:
        """Return the RTD ticker string for *symbol*.

        Parameters
        ----------
        symbol : str
            DerivsLab instrument identifier.

        Returns
        -------
        str
            RTD ticker, e.g. ``"PETRA201_B_0"``.

        Raises
        ------
        QuoteUnavailableError
            If the symbol is not in the registry, has no exchange segment
            set, or its segment has no configured RTD suffix.
        """
        contract = self._registry.get(symbol)
        if contract is None:
            raise QuoteUnavailableError(symbol, "symbol not found in InstrumentRegistry")

        segment = contract.exchange_segment
        if segment is None:
            raise QuoteUnavailableError(
                symbol,
                "exchange_segment not set on contract — cannot determine RTD suffix",
            )

        suffix = _RTD_SUFFIX.get(segment)
        if suffix is None:
            raise QuoteUnavailableError(
                symbol,
                f"no RTD suffix configured for segment '{segment.value}'",
            )

        return f"{symbol}{suffix}"

    @staticmethod
    def _rtd_formula(rtd_name: str, attribute: str) -> str:
        """Build a single RTD formula string using English comma syntax.

        Parameters
        ----------
        rtd_name : str
            Full RTD ticker, e.g. ``"PETRA201_B_0"``.
        attribute : str
            RTD attribute code, e.g. ``"ULT"``, ``"OCP"``.

        Returns
        -------
        str
            Formula string starting with ``=RTD(``.
        """
        return f'=RTD("{RTD_SERVER}",,"{rtd_name}","{attribute}")'

    # --- sheet I/O ----------------------------------------------------------

    def _write_batch(self, symbols: list[str], rtd_names: list[str]) -> None:
        """Write symbol labels and RTD formulas for one batch.

        Writes column A (symbol labels) and columns B–G (RTD formulas)
        starting at ``_DATA_START_ROW``.

        Parameters
        ----------
        symbols : list[str]
            DerivsLab tickers for this batch.
        rtd_names : list[str]
            Corresponding RTD tickers (same length as *symbols*).
        """
        n = len(symbols)
        end_row = _DATA_START_ROW + n - 1

        # Column A — symbol labels for readability and audit
        self._sheet.range(f"A{_DATA_START_ROW}:A{end_row}").value = [[s] for s in symbols]

        # Columns B–G — one RTD formula per attribute per symbol
        formulas: list[list[str]] = [
            [self._rtd_formula(rtd_name, attr) for attr in _RTD_ATTRIBUTES]
            for rtd_name in rtd_names
        ]
        self._sheet.range(f"B{_DATA_START_ROW}:G{end_row}").formula = formulas

    def _read_batch(self, n: int) -> list[list[object]]:
        """Read the RTD values for *n* rows starting at ``_DATA_START_ROW``.

        xlwings returns a flat list for a single row and a 2-D list for
        multiple rows. This method always returns a 2-D list.

        Parameters
        ----------
        n : int
            Number of rows (symbols) in the current batch.

        Returns
        -------
        list[list[object]]
            2-D list of shape ``(n, 6)``, one inner list per symbol in
            column order ``[last, bid, ask, bid_size, ask_size, volume]``.
        """
        end_row = _DATA_START_ROW + n - 1
        raw = self._sheet.range(f"B{_DATA_START_ROW}:G{end_row}").value
        # Normalise single-row result to 2-D
        if n == 1:
            return [raw]
        return raw  # type: ignore[no-any-return]

    def _clear_batch(self, n: int) -> None:
        """Clear the data rows for *n* symbols starting at ``_DATA_START_ROW``.

        Parameters
        ----------
        n : int
            Number of rows to clear.
        """
        end_row = _DATA_START_ROW + n - 1
        self._sheet.range(f"A{_DATA_START_ROW}:G{end_row}").value = None

    # --- quote construction -------------------------------------------------

    @staticmethod
    def _is_rtd_error(value: object) -> bool:
        """Return ``True`` if *value* represents an Excel/RTD error or missing data.

        Handles the cases xlwings may return for unavailable RTD values:
        ``None`` for empty cells, strings starting with ``#`` for Excel
        error codes (``#N/A``, ``#VALUE!``, etc.).

        Parameters
        ----------
        value : object
            Raw value read from the Excel range.

        Returns
        -------
        bool
        """
        if value is None:
            return True
        if isinstance(value, str) and value.startswith("#"):
            return True
        return False

    @staticmethod
    def _to_optional_float(value: object) -> float | None:
        """Convert *value* to ``float``, or ``None`` if it is an RTD error.

        Parameters
        ----------
        value : object
            Raw value from Excel.

        Returns
        -------
        float or None
        """
        if ProfitProRTDProvider._is_rtd_error(value):
            return None
        return float(value)  # type: ignore[arg-type]

    def _row_to_quote(
        self,
        symbol: str,
        row: list[object],
        collected_at: datetime,
    ) -> Quote:
        """Translate a raw RTD data row into a ``Quote``.

        Bid and ask are required fields — an RTD error on either raises
        ``QuoteUnavailableError`` for this symbol. All other fields are
        optional and become ``None`` when the RTD server returns an error.

        Parameters
        ----------
        symbol : str
            DerivsLab instrument identifier.
        row : list[object]
            Six-element list in column order:
            ``[last, bid, ask, bid_size, ask_size, volume]``.
        collected_at : datetime
            Timestamp of the snapshot collection, shared across all quotes
            in the same batch.

        Returns
        -------
        Quote

        Raises
        ------
        QuoteUnavailableError
            If bid or ask contains an RTD error value.
        """
        last_raw, bid_raw, ask_raw, bid_size_raw, ask_size_raw, volume_raw = row

        if self._is_rtd_error(bid_raw) or self._is_rtd_error(ask_raw):
            raise QuoteUnavailableError(
                symbol,
                f"bid or ask returned an RTD error " f"(bid={bid_raw!r}, ask={ask_raw!r})",
            )

        return Quote(
            symbol=symbol,
            bid=float(bid_raw),  # type: ignore[arg-type]
            ask=float(ask_raw),  # type: ignore[arg-type]
            last=self._to_optional_float(last_raw),
            bid_size=self._to_optional_float(bid_size_raw),
            ask_size=self._to_optional_float(ask_size_raw),
            volume=self._to_optional_float(volume_raw),
            timestamp=collected_at,
        )
