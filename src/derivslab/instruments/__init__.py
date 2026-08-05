"""Instrument reference data and pricing-behavior contracts.

Exposes the public surface of the instruments package: the ``Instrument``
behavioral interface, the immutable ``*Contract`` reference-data classes,
and the concrete instruments (``EquityInstrument``, ``VanillaOption``).
Internal module layout may change; import from here rather than from the
submodules directly.
"""

from __future__ import annotations

from derivslab.instruments.base import Instrument
from derivslab.instruments.contracts import (
    DI_FUTURE_SETTLEMENT_VALUE,
    PERPETUAL_EXPIRY,
    Currency,
    DayCountConvention,
    DIFutureContract,
    EquityContract,
    Exchange,
    ExchangeSegment,
    ExerciseStyle,
    InstrumentContract,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)
from derivslab.instruments.equity import EquityInstrument
from derivslab.instruments.future import DI_RATE_UNDERLYING, DIFuture
from derivslab.instruments.registry import InstrumentRegistry
from derivslab.instruments.vanilla import VanillaOption

__all__ = [
    "Currency",
    "DayCountConvention",
    "DIFuture",
    "DIFutureContract",
    "DI_FUTURE_SETTLEMENT_VALUE",
    "DI_RATE_UNDERLYING",
    "EquityContract",
    "EquityInstrument",
    "Exchange",
    "ExchangeSegment",
    "ExerciseStyle",
    "Instrument",
    "InstrumentContract",
    "InstrumentRegistry",
    "OptionType",
    "PERPETUAL_EXPIRY",
    "UnderlyingAssetClass",
    "VanillaOption",
    "VanillaOptionContract",
]
