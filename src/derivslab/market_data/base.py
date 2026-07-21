"""
Market data abstractions for the DerivsLab market data layer.

Responsibilities
----------------
- Define the normalised quote schema (``Quote``, ``MarketSnapshot``) that all
  market data consumers depend on, independently of the upstream data provider.
- Define the ``MarketDataProvider`` abstract interface that every concrete
  provider adapter must implement.
- Define the exception hierarchy used to signal partial or total quote failures.

What does NOT belong here
--------------------------
- Concrete provider implementations (MetaTrader 5, ProfitPro RTD, etc.).
- Instrument registry / B3 file parsing logic.
- In-memory or persistent cache management.
- Any retry, reconnection, or transport-level logic — those are implementation
  details of each concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNAVAILABLE_REASON_UNKNOWN: Final[str] = "unavailable"


# ---------------------------------------------------------------------------
# Normalised quote schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quote:
    """
    Atomic market quote for a single instrument at a single point in time.

    ``Quote`` is the canonical unit of market data inside DerivsLab.  Every
    concrete ``MarketDataProvider`` must translate its native data format into
    this schema before returning data to any consumer.

    Parameters
    ----------
    symbol : str
        Instrument identifier in DerivsLab's internal vocabulary (e.g. the
        B3 ticker such as ``"PETRA201"`` or ``"PETR4"``).  Never the
        provider-specific name.
    bid : float
        Best bid price at the time of the quote.
    ask : float
        Best ask price at the time of the quote.
    timestamp : datetime
        Moment the quote was captured.  Must be timezone-aware (UTC strongly
        recommended).
    last : float or None
        Last traded price.  ``None`` when no recent trade exists — common for
        illiquid or deep out-of-the-money options.  Consumers must not assume
        this field is present.
    bid_size : float or None
        Quantity available at the best bid.  ``None`` when the provider does
        not expose depth.
    ask_size : float or None
        Quantity available at the best ask.  ``None`` when the provider does
        not expose depth.
    volume : float or None
        Total traded volume for the session.  ``None`` when unavailable.

    Notes
    -----
    ``mid`` is intentionally not stored as a field — it is a derived value
    computed on demand via a property to avoid redundant state.
    """

    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    last: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None

    @property
    def mid(self) -> float:
        """
        Midpoint between best bid and best ask.

        Returns
        -------
        float
            ``(bid + ask) / 2``
        """
        return (self.bid + self.ask) / 2.0


# ---------------------------------------------------------------------------
# Snapshot — coherent collection of quotes from a single collection call
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """
    A coherent set of quotes collected in a single provider call.

    ``MarketSnapshot`` groups multiple ``Quote`` objects captured at the same
    logical instant.  Individual quotes carry their own provider-level
    timestamps; ``collected_at`` marks when *this snapshot* was assembled,
    which is useful to detect staleness independently of per-quote timestamps.

    Parameters
    ----------
    quotes : dict[str, Quote]
        Mapping from symbol (DerivsLab vocabulary) to the corresponding
        ``Quote``.  Populated by the provider; consumers treat it as
        read-only.
    collected_at : datetime
        Moment the snapshot was assembled.  Must be timezone-aware.

    Notes
    -----
    ``MarketSnapshot`` is a passive container.  It never initiates provider
    calls.  All population logic lives in ``MarketDataProvider.get_quotes``.
    """

    quotes: dict[str, Quote] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=lambda: datetime.now())

    # --- public API ---------------------------------------------------------

    def get(self, symbol: str) -> Quote | None:
        """
        Return the quote for *symbol*, or ``None`` if not present.

        Prefer this over direct ``dict`` access to avoid ``KeyError`` in
        consumer code.

        Parameters
        ----------
        symbol : str
            Instrument identifier in DerivsLab's internal vocabulary.

        Returns
        -------
        Quote or None
        """
        return self.quotes.get(symbol)

    @property
    def symbols(self) -> list[str]:
        """
        List of instrument identifiers present in this snapshot.

        Returns
        -------
        list[str]
        """
        return list(self.quotes.keys())

    def is_complete(self, expected: list[str]) -> bool:
        """
        Return ``True`` when every expected symbol is present in the snapshot.

        Useful as a pre-flight check before passing the snapshot to a consumer
        that requires all symbols (e.g. a vol surface fitter).

        Parameters
        ----------
        expected : list[str]
            Symbols the consumer requires.

        Returns
        -------
        bool
            ``True`` if all symbols in *expected* have a quote; ``False``
            otherwise.
        """
        return all(symbol in self.quotes for symbol in expected)

    def __len__(self) -> int:
        return len(self.quotes)

    def __repr__(self) -> str:
        return (
            f"MarketSnapshot("
            f"symbols={self.symbols}, "
            f"collected_at={self.collected_at.isoformat()})"
        )


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class MarketDataError(Exception):
    """Base exception for all market data errors."""


class QuoteUnavailableError(MarketDataError):
    """
    Raised when a provider cannot return a quote for a requested symbol.

    Parameters
    ----------
    symbol : str
        The symbol for which the quote was unavailable.
    reason : str
        Human-readable description of why the quote could not be obtained.

    Examples
    --------
    >>> raise QuoteUnavailableError("PETRA201", "no market data returned by provider")
    """

    def __init__(self, symbol: str, reason: str = UNAVAILABLE_REASON_UNKNOWN) -> None:
        self.symbol: str = symbol
        self.reason: str = reason
        super().__init__(f"Quote unavailable for '{symbol}': {reason}")


class PartialSnapshotError(MarketDataError):
    """
    Raised by ``get_quotes`` when one or more symbols could not be quoted.

    The exception carries the partial ``MarketSnapshot`` (symbols that
    succeeded) alongside a mapping of failed symbols to their failure reasons.
    Consumers can inspect the partial result and decide whether to proceed or
    abort.

    Parameters
    ----------
    snapshot : MarketSnapshot
        Quotes that were successfully obtained.
    unavailable : dict[str, str]
        Mapping of ``symbol → reason`` for every symbol that failed.

    Examples
    --------
    >>> try:
    ...     snapshot = provider.get_quotes(symbols)
    ... except PartialSnapshotError as exc:
    ...     snapshot = exc.snapshot
    ...     if not snapshot.is_complete(required_symbols):
    ...         raise
    """

    def __init__(
        self,
        snapshot: MarketSnapshot,
        unavailable: dict[str, str],
    ) -> None:
        self.snapshot: MarketSnapshot = snapshot
        self.unavailable: dict[str, str] = unavailable
        failed = ", ".join(
            f"'{s}' ({r})" for s, r in unavailable.items()
        )
        super().__init__(
            f"Partial snapshot: {len(snapshot)} quote(s) obtained; "
            f"failed symbols — {failed}"
        )


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class MarketDataProvider(ABC):
    """
    Abstract interface for any market data source.

    All concrete provider adapters (MetaTrader 5, ProfitPro RTD, etc.) must
    subclass ``MarketDataProvider`` and implement ``get_quote`` and
    ``get_quotes``.  Consumers depend exclusively on this interface and are
    therefore agnostic to the underlying data source.

    The interface is pull-only: callers request data on demand.  Push / event-
    driven delivery is not part of this contract.

    Notes
    -----
    Symbol identifiers used in all method signatures refer to DerivsLab's
    internal vocabulary (typically B3 tickers).  Translation from internal
    symbols to provider-specific names is the responsibility of each concrete
    implementation.

    See Also
    --------
    Quote : Atomic quote schema returned by this provider.
    MarketSnapshot : Coherent multi-symbol snapshot returned by ``get_quotes``.
    """

    # --- abstract methods ---------------------------------------------------

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Return the current quote for a single instrument.

        Parameters
        ----------
        symbol : str
            Instrument identifier in DerivsLab's internal vocabulary.

        Returns
        -------
        Quote
            Current market quote for *symbol*.

        Raises
        ------
        QuoteUnavailableError
            If the provider cannot return a quote for *symbol* — e.g. the
            symbol is unknown, the market is closed, or the connection failed.
        """
        ...

    @abstractmethod
    def get_quotes(self, symbols: list[str]) -> MarketSnapshot:
        """
        Return a coherent snapshot of quotes for a list of instruments.

        Parameters
        ----------
        symbols : list[str]
            Instrument identifiers in DerivsLab's internal vocabulary.

        Returns
        -------
        MarketSnapshot
            Snapshot containing one ``Quote`` per symbol in *symbols*.

        Raises
        ------
        PartialSnapshotError
            If one or more symbols could not be quoted.  The exception carries
            the partial ``MarketSnapshot`` (successful quotes) and a mapping of
            failed symbols to their reasons.  Callers that require all symbols
            should check ``exc.snapshot.is_complete(symbols)`` and re-raise or
            handle accordingly.
        MarketDataError
            If a systemic failure prevents the provider from returning any
            data at all (e.g. connection lost, authentication failure).
        """
        ...

    # --- concrete helpers ---------------------------------------------------

    def get_mid(self, symbol: str) -> float:
        """
        Convenience method: return the mid-price for a single symbol.

        Delegates to ``get_quote`` and returns ``quote.mid``.

        Parameters
        ----------
        symbol : str
            Instrument identifier in DerivsLab's internal vocabulary.

        Returns
        -------
        float
            ``(bid + ask) / 2`` for *symbol*.

        Raises
        ------
        QuoteUnavailableError
            Propagated from ``get_quote`` if the symbol is unavailable.
        """
        return self.get_quote(symbol).mid