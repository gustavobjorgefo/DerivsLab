"""Tests for derivslab.instruments.vanilla.VanillaOption."""

from __future__ import annotations

from datetime import date
from typing import Callable

import pytest

from derivslab.calendars import TradingCalendar
from derivslab.instruments.contracts import (
    DayCountConvention,
    ExerciseStyle,
    OptionType,
    UnderlyingAssetClass,
    VanillaOptionContract,
)
from derivslab.instruments.vanilla import VanillaOption


class TestIdentity:
    """Tests for the parts of the interface unrelated to payoff or time."""

    def test_underlyings_is_the_single_underlying_ticker(
        self, european_call: VanillaOption
    ) -> None:
        assert european_call.underlyings == ["PETR4"]

    def test_contract_returns_the_wrapped_contract(
        self, european_call: VanillaOption, make_option_contract
    ) -> None:
        assert european_call.contract.ticker == "PETRA123"

    def test_european_style_routes_to_closed_form_pricer(
        self, european_call: VanillaOption
    ) -> None:
        assert european_call.pricing_model_key == "bs_vanilla_european_equity"

    def test_american_style_routes_to_numerical_pricer(self, american_put: VanillaOption) -> None:
        assert american_put.pricing_model_key == "binomial_vanilla_american_equity"


class TestPricingModelKeyResolution:
    """Tests for pricing_model_key resolving from (style, underlying_asset_class).

    Style alone is not enough to pick a pricer: an equity option is priced
    off spot with a carry term, while a future option is priced off the
    forward itself, even under the same exercise style.
    """

    @pytest.mark.parametrize(
        ("style", "underlying_asset_class", "expected_key"),
        [
            (ExerciseStyle.EUROPEAN, UnderlyingAssetClass.EQUITY, "bs_vanilla_european_equity"),
            (
                ExerciseStyle.EUROPEAN,
                UnderlyingAssetClass.FUTURE,
                "black76_vanilla_european_future",
            ),
            (
                ExerciseStyle.AMERICAN,
                UnderlyingAssetClass.EQUITY,
                "binomial_vanilla_american_equity",
            ),
            (
                ExerciseStyle.AMERICAN,
                UnderlyingAssetClass.FUTURE,
                "binomial_vanilla_american_future",
            ),
        ],
    )
    def test_pricing_model_key_resolves_from_style_and_asset_class(
        self,
        make_option_contract: Callable[..., VanillaOptionContract],
        style: ExerciseStyle,
        underlying_asset_class: UnderlyingAssetClass,
        expected_key: str,
    ) -> None:
        contract = make_option_contract(style=style, underlying_asset_class=underlying_asset_class)
        option = VanillaOption(contract)
        assert option.pricing_model_key == expected_key


class TestPayoff:
    """Tests for VanillaOption.payoff."""

    @pytest.mark.parametrize(
        ("spot", "expected"),
        [
            (40.0, 5.0),  # in the money
            (35.0, 0.0),  # at the money
            (30.0, 0.0),  # out of the money
        ],
    )
    def test_call_payoff(
        self,
        make_option_contract: Callable[..., VanillaOptionContract],
        spot: float,
        expected: float,
    ) -> None:
        option = VanillaOption(make_option_contract(option_type=OptionType.CALL, strike=35.0))
        assert option.payoff({"PETR4": spot}) == expected

    @pytest.mark.parametrize(
        ("spot", "expected"),
        [
            (30.0, 5.0),  # in the money
            (35.0, 0.0),  # at the money
            (40.0, 0.0),  # out of the money
        ],
    )
    def test_put_payoff(
        self,
        make_option_contract: Callable[..., VanillaOptionContract],
        spot: float,
        expected: float,
    ) -> None:
        option = VanillaOption(make_option_contract(option_type=OptionType.PUT, strike=35.0))
        assert option.payoff({"PETR4": spot}) == expected


class TestTimeToExpiry:
    """Tests for VanillaOption.time_to_expiry."""

    def test_act_365_matches_calendar_days_over_365(
        self, european_call: VanillaOption, valuation_date: date
    ) -> None:
        expected = (european_call.contract.expiry - valuation_date).days / 365.0
        assert european_call.time_to_expiry(valuation_date) == pytest.approx(expected)

    def test_act_360_matches_calendar_days_over_360(
        self,
        make_option_contract: Callable[..., VanillaOptionContract],
        valuation_date: date,
    ) -> None:
        contract = make_option_contract(day_count_convention=DayCountConvention.ACT_360)
        option = VanillaOption(contract)
        expected = (contract.expiry - valuation_date).days / 360.0
        assert option.time_to_expiry(valuation_date) == pytest.approx(expected)

    def test_bus_252_without_calendar_raises(
        self, american_put: VanillaOption, valuation_date: date
    ) -> None:
        with pytest.raises(ValueError, match="requires a TradingCalendar"):
            american_put.time_to_expiry(valuation_date)

    def test_bus_252_with_calendar_matches_business_days_over_252(
        self,
        american_put: VanillaOption,
        valuation_date: date,
        trading_calendar: TradingCalendar,
    ) -> None:
        expected_days = trading_calendar.business_days_between(
            valuation_date, american_put.contract.expiry
        )
        result = american_put.time_to_expiry(valuation_date, calendar=trading_calendar)
        assert result == pytest.approx(expected_days / 252.0)

    def test_valuation_at_expiry_returns_zero(self, european_call: VanillaOption) -> None:
        assert european_call.time_to_expiry(european_call.contract.expiry) == 0.0

    def test_valuation_after_expiry_raises(self, european_call: VanillaOption) -> None:
        past_expiry = date(2027, 1, 1)
        with pytest.raises(ValueError, match="is past expiry"):
            european_call.time_to_expiry(past_expiry)

    def test_american_at_expiry_returns_zero_even_without_calendar(
        self, american_put: VanillaOption
    ) -> None:
        # The same-day short-circuit must run before the BUS_252/calendar
        # check, since no business-day count is needed when expiry is today.
        assert american_put.time_to_expiry(american_put.contract.expiry) == 0.0
