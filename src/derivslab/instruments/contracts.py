"""Reference data contracts for tradable instruments.

This module defines the immutable reference data that describes what an
instrument *is*, as registered by an exchange or agreed upon in an OTC
confirmation: identifiers, currency, contract size, day-count convention,
and product-specific terms (strike, expiry, exercise style, and so on).

Contracts carry no pricing behavior and no market data (spot, volatility,
rates). Behavior belongs to the ``Instrument`` hierarchy in ``base.py``,
which wraps a contract and exposes ``payoff``, ``time_to_expiry``, and
``pricing_model_key``. Holiday calendars belong to ``derivslab.calendars``.
Market data belongs to ``MarketState``, defined elsewhere. Keeping these
concerns separate is what allows the same contract to be revalued under a
different pricing model, or against a different calendar, without
mutation.

Fields mirror a deliberately small subset of B3's InstrumentsConsolidatedFile
public glossary — enough to be realistic without registering every
regulatory attribute a listed exchange tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Final

# Sentinel expiry for instruments with no contractual maturity (e.g. equities).
PERPETUAL_EXPIRY: Final[date] = date(9999, 12, 31)


class Currency(str, Enum):
    """Settlement currency of an instrument.

    Attributes
    ----------
    BRL : Brazilian real.
    USD : United States dollar.
    EUR : Euro.
    JPY : Japanese yen.
    """

    BRL = "BRL"
    USD = "USD"
    EUR = "EUR"
    JPY = "JPY"


class OptionType(str, Enum):
    """Direction of an option's payoff.

    Attributes
    ----------
    CALL : Right to buy the underlying at the strike.
    PUT : Right to sell the underlying at the strike.
    """

    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    """Timing constraint on when an option's payoff may be realized.

    Attributes
    ----------
    EUROPEAN : Exercisable only at expiry.
    AMERICAN : Exercisable at any time up to and including expiry.
    """

    EUROPEAN = "european"
    AMERICAN = "american"


class DayCountConvention(str, Enum):
    """Day-count basis used to convert calendar time into year fractions.

    Mirrors B3's ``BaseCd`` field, which records the day-count basis
    (252, 360, or 365) per instrument rather than assuming a single
    global convention.

    Attributes
    ----------
    BUS_252 : Business days divided by 252 — the Brazilian market
        standard for equities and equity options. Requires a
        ``TradingCalendar`` to count business days.
    ACT_360 : Calendar days divided by 360.
    ACT_365 : Calendar days divided by 365.
    """

    BUS_252 = "bus/252"
    ACT_360 = "act/360"
    ACT_365 = "act/365"


class Exchange(str, Enum):
    """Venue where an instrument is listed or cleared.

    Provider-agnostic — this is the exchange itself, not a data vendor's
    internal naming (e.g. ProfitPro's RTD suffixes). ``None`` on a
    contract means the instrument is not tied to a real venue, which is
    expected for research-only or synthetic instruments.

    Attributes
    ----------
    B3 : Brazilian exchange (Brasil, Bolsa, Balcão).
    """

    B3 = "b3"


class ExchangeSegment(str, Enum):
    """Trading segment within an exchange.

    A segment is orthogonal to product type: it reflects venue-level
    conventions such as clearing, margining, and settlement calendars,
    not the instrument's payoff structure. Two contracts of the same
    ``VanillaOptionContract`` class can sit in different segments
    depending on their underlying (e.g. an equity option vs. an option
    on a rate future).

    Attributes
    ----------
    BOVESPA : Equities and equity derivatives segment (``_B_0`` in
        ProfitPro's RTD naming).
    BMF : Futures, FX, rates, and commodities segment (``_F_0`` in
        ProfitPro's RTD naming).
    """

    BOVESPA = "bovespa"
    BMF = "bmf"


class UnderlyingAssetClass(str, Enum):
    """Nature of the asset a derivative contract is written on.

    Determines which pricing model applies to a given exercise style —
    the payoff formula is identical across asset classes, but the
    dynamics of the underlying (spot vs. forward) are not: an equity
    option is priced off spot with a carry term, a future option is
    priced off the forward itself.

    Attributes
    ----------
    EQUITY : Underlying trades at spot.
    FUTURE : Underlying is a futures contract, priced by cost of carry.
    """

    EQUITY = "equity"
    FUTURE = "future"


@dataclass(frozen=True, kw_only=True)
class InstrumentContract:
    """Common reference data shared by every tradable instrument.

    Parameters
    ----------
    instrument_id : str
        Unique identifier assigned by the exchange or the desk's own
        registry (e.g. a B3 ticker for listed instruments, or an internal
        id for OTC deals).
    currency : Currency
        Settlement currency of the instrument.
    contract_size : int
        Number of underlying units represented by one contract.
    tick_size : float
        Minimum price increment allowed for quoting this instrument.
    day_count_convention : DayCountConvention
        Basis used to convert calendar time into year fractions when
        computing time to expiry. Defaults to the Brazilian market
        standard.
    isin : str | None
        International Securities Identification Number, when the
        instrument has one. Derivatives typically do not.
    cfi_code : str | None
        Classification of Financial Instruments code, when registered.
    exchange : Exchange | None
        Venue where the instrument is listed or cleared. ``None`` for
        research-only or synthetic instruments with no real venue.
    exchange_segment : ExchangeSegment | None
        Trading segment within ``exchange``. ``None`` when ``exchange``
        is ``None``; requires ``exchange`` to be set otherwise.

    Raises
    ------
    ValueError
        If ``contract_size`` or ``tick_size`` is not strictly positive,
        or if ``exchange_segment`` is set without ``exchange``.
    """

    instrument_id: str
    currency: Currency
    contract_size: int = 100
    tick_size: float = 0.01
    day_count_convention: DayCountConvention = DayCountConvention.BUS_252
    isin: str | None = None
    cfi_code: str | None = None
    exchange: Exchange | None = None
    exchange_segment: ExchangeSegment | None = None

    def __post_init__(self) -> None:
        if self.contract_size <= 0:
            raise ValueError(
                f"contract_size must be strictly positive, got {self.contract_size} "
                f"for {self.instrument_id}."
            )
        if self.tick_size <= 0:
            raise ValueError(
                f"tick_size must be strictly positive, got {self.tick_size} "
                f"for {self.instrument_id}."
            )
        if self.exchange_segment is not None and self.exchange is None:
            raise ValueError(
                f"exchange_segment set without exchange for {self.instrument_id}."
            )


@dataclass(frozen=True, kw_only=True)
class EquityContract(InstrumentContract):
    """Reference data for a cash equity.

    An equity has no contractual expiry, so ``expiry`` defaults to
    ``PERPETUAL_EXPIRY`` rather than being made optional — this keeps the
    field present and comparable across every instrument that inherits
    from ``InstrumentContract``, without special-casing equities elsewhere.

    Parameters
    ----------
    ticker : str
        Exchange ticker of the equity (e.g. "PETR4").
    expiry : date
        Contractual expiry. Always ``PERPETUAL_EXPIRY`` for equities.
    """

    ticker: str
    expiry: date = field(default=PERPETUAL_EXPIRY)


@dataclass(frozen=True, kw_only=True)
class VanillaOptionContract(InstrumentContract):
    """Reference data for a vanilla (call/put) option.

    Exercise style is a contractual attribute, not a distinct product: a
    European and an American vanilla share the same payoff formula and
    differ only in when that payoff may be realized. This is what allows
    a single contract class to serve both, with ``style`` alone steering
    the ``Instrument`` to the correct pricing engine.

    ``underlying`` is a single ticker (a foreign key, not a nested
    contract) because a vanilla is single-name by definition and because
    reference data is meant to stay flat and serializable — this mirrors
    B3's own file, which records ``UndrlygTckrSymb`` as plain text.
    Multi-name payoffs (basket options) belong to a separate contract
    class with ``underlyings: list[str]``, not to this one.

    ``underlying_asset_class`` is kept alongside ``underlying`` for the
    same reason: resolving it would otherwise require materializing the
    underlying's own ``Instrument``, which reintroduces the nested-object
    problem ``underlying`` was designed to avoid. It is required, not
    optional, because the payoff is style-agnostic but the pricing
    dynamics are not — an equity option is priced off spot, a future
    option off the forward — so ``VanillaOption.pricing_model_key``
    resolves from ``(style, underlying_asset_class)`` together.

    Parameters
    ----------
    ticker : str
        Exchange ticker of the option contract (e.g. "PETRA123").
    underlying : str
        Ticker of the underlying instrument.
    underlying_asset_class : UnderlyingAssetClass
        Nature of the underlying (equity, future, ...), determining
        which pricing model applies alongside ``style``.
    option_type : OptionType
        Call or put.
    style : ExerciseStyle
        European or American.
    strike : float
        Strike price.
    expiry : date
        Contractual expiry date.

    Raises
    ------
    ValueError
        If ``strike`` is not strictly positive.
    """

    ticker: str
    underlying: str
    underlying_asset_class: UnderlyingAssetClass
    option_type: OptionType
    style: ExerciseStyle
    strike: float
    expiry: date

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strike <= 0:
            raise ValueError(
                f"strike must be strictly positive, got {self.strike} "
                f"for {self.instrument_id}."
            )