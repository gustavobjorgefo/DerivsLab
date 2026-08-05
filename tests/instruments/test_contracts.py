"""Tests for derivslab.instruments.contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from typing import Callable

import pytest

from derivslab.instruments.contracts import (
    PERPETUAL_EXPIRY,
    Currency,
    DayCountConvention,
    DIFutureContract,
    EquityContract,
    Exchange,
    ExchangeSegment,
    ExerciseStyle,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)


class TestInstrumentContractDefaults:
    """Tests for the base InstrumentContract's default values."""

    def test_defaults_are_brazilian_market_standard(self, equity_contract: EquityContract) -> None:
        assert equity_contract.contract_size == 100
        assert equity_contract.tick_size == 0.01
        assert equity_contract.day_count_convention is DayCountConvention.BUS_252

    def test_isin_and_cfi_code_default_to_none(self, equity_contract: EquityContract) -> None:
        assert equity_contract.isin is None
        assert equity_contract.cfi_code is None

    def test_isin_and_cfi_code_can_be_set(self) -> None:
        contract = EquityContract(
            instrument_id="PETR4",
            currency=Currency.BRL,
            ticker="PETR4",
            isin="BRPETRACNOR9",
            cfi_code="ESVUFR",
        )
        assert contract.isin == "BRPETRACNOR9"
        assert contract.cfi_code == "ESVUFR"

    def test_exchange_and_exchange_segment_default_to_none(
        self, equity_contract: EquityContract
    ) -> None:
        assert equity_contract.exchange is None
        assert equity_contract.exchange_segment is None

    def test_exchange_and_exchange_segment_can_be_set_together(self) -> None:
        contract = EquityContract(
            instrument_id="PETR4",
            currency=Currency.BRL,
            ticker="PETR4",
            exchange=Exchange.B3,
            exchange_segment=ExchangeSegment.BOVESPA,
        )
        assert contract.exchange is Exchange.B3
        assert contract.exchange_segment is ExchangeSegment.BOVESPA

    def test_exchange_can_be_set_without_a_segment(self) -> None:
        contract = EquityContract(
            instrument_id="PETR4",
            currency=Currency.BRL,
            ticker="PETR4",
            exchange=Exchange.B3,
        )
        assert contract.exchange is Exchange.B3
        assert contract.exchange_segment is None

    def test_exchange_segment_without_exchange_raises(self) -> None:
        with pytest.raises(ValueError, match="exchange_segment set without exchange"):
            EquityContract(
                instrument_id="PETR4",
                currency=Currency.BRL,
                ticker="PETR4",
                exchange_segment=ExchangeSegment.BOVESPA,
            )

    @pytest.mark.parametrize("contract_size", [0, -1, -100])
    def test_non_positive_contract_size_raises(self, contract_size: int) -> None:
        with pytest.raises(ValueError, match="contract_size must be strictly positive"):
            EquityContract(
                instrument_id="PETR4",
                currency=Currency.BRL,
                ticker="PETR4",
                contract_size=contract_size,
            )

    @pytest.mark.parametrize("tick_size", [0, -0.01])
    def test_non_positive_tick_size_raises(self, tick_size: float) -> None:
        with pytest.raises(ValueError, match="tick_size must be strictly positive"):
            EquityContract(
                instrument_id="PETR4",
                currency=Currency.BRL,
                ticker="PETR4",
                tick_size=tick_size,
            )

    def test_contracts_are_frozen(self, equity_contract: EquityContract) -> None:
        with pytest.raises(FrozenInstanceError):
            equity_contract.ticker = "VALE3"  # type: ignore[misc]


class TestEquityContract:
    """Tests specific to EquityContract."""

    def test_expiry_defaults_to_perpetual(self, equity_contract: EquityContract) -> None:
        assert equity_contract.expiry == PERPETUAL_EXPIRY

    def test_perpetual_expiry_is_far_future(self) -> None:
        assert PERPETUAL_EXPIRY.year == 9999


class TestVanillaOptionContract:
    """Tests specific to VanillaOptionContract."""

    def test_valid_contract_holds_given_fields(
        self, make_option_contract: Callable[..., VanillaOptionContract]
    ) -> None:
        contract = make_option_contract(strike=42.0)
        assert contract.strike == 42.0
        assert contract.underlying == "PETR4"

    @pytest.mark.parametrize("strike", [0.0, -1.0, -100.0])
    def test_non_positive_strike_raises(
        self, make_option_contract: Callable[..., VanillaOptionContract], strike: float
    ) -> None:
        with pytest.raises(ValueError, match="strike must be strictly positive"):
            make_option_contract(strike=strike)

    def test_inherits_base_contract_size_validation(
        self, make_option_contract: Callable[..., VanillaOptionContract]
    ) -> None:
        with pytest.raises(ValueError, match="contract_size must be strictly positive"):
            make_option_contract(contract_size=0)

    def test_underlying_is_a_plain_ticker_string(
        self, make_option_contract: Callable[..., VanillaOptionContract]
    ) -> None:
        contract = make_option_contract()
        assert isinstance(contract.underlying, str)

    def test_underlying_asset_class_is_required(self) -> None:
        with pytest.raises(TypeError):
            VanillaOptionContract(
                instrument_id="PETRA123",
                currency=Currency.BRL,
                ticker="PETRA123",
                underlying="PETR4",
                option_type=OptionType.CALL,
                style=ExerciseStyle.EUROPEAN,
                strike=35.0,
                expiry=date(2026, 12, 18),
            )  # type: ignore[call-arg]

    def test_underlying_asset_class_can_be_future(
        self, make_option_contract: Callable[..., VanillaOptionContract]
    ) -> None:
        contract = make_option_contract(
            underlying="WINQ25",
            underlying_asset_class=UnderlyingAssetClass.FUTURE,
        )
        assert contract.underlying_asset_class is UnderlyingAssetClass.FUTURE


class TestDIFutureContract:
    """Tests specific to DIFutureContract."""

    def test_valid_contract_holds_given_fields(self, di_future_contract: DIFutureContract) -> None:
        assert di_future_contract.ticker == "DI1F27"
        assert di_future_contract.expiry == date(2027, 7, 13)

    def test_contract_size_defaults_to_one(self, di_future_contract: DIFutureContract) -> None:
        assert di_future_contract.contract_size == 1

    def test_tick_size_defaults_to_b3_finer_tier(
        self, di_future_contract: DIFutureContract
    ) -> None:
        assert di_future_contract.tick_size == 0.001

    def test_point_value_defaults_to_one(self, di_future_contract: DIFutureContract) -> None:
        assert di_future_contract.point_value == 1.0

    def test_day_count_convention_defaults_to_bus_252(
        self, di_future_contract: DIFutureContract
    ) -> None:
        assert di_future_contract.day_count_convention is DayCountConvention.BUS_252

    def test_point_value_can_be_overridden(self) -> None:
        contract = DIFutureContract(
            instrument_id="DI1F27",
            currency=Currency.BRL,
            ticker="DI1F27",
            expiry=date(2027, 7, 13),
            point_value=0.5,
        )
        assert contract.point_value == 0.5

    @pytest.mark.parametrize("point_value", [0.0, -1.0, -100.0])
    def test_non_positive_point_value_raises(self, point_value: float) -> None:
        with pytest.raises(ValueError, match="point_value must be strictly positive"):
            DIFutureContract(
                instrument_id="DI1F27",
                currency=Currency.BRL,
                ticker="DI1F27",
                expiry=date(2027, 7, 13),
                point_value=point_value,
            )

    def test_inherits_base_contract_size_validation(self) -> None:
        with pytest.raises(ValueError, match="contract_size must be strictly positive"):
            DIFutureContract(
                instrument_id="DI1F27",
                currency=Currency.BRL,
                ticker="DI1F27",
                expiry=date(2027, 7, 13),
                contract_size=0,
            )

    def test_contract_is_frozen(self, di_future_contract: DIFutureContract) -> None:
        with pytest.raises(FrozenInstanceError):
            di_future_contract.ticker = "DI1F28"  # type: ignore[misc]
