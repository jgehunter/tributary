"""Unit tests for orderbook-based cost forecasting."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tributary.analytics.cost_forecast import (
    CostForecast,
    estimate_slippage_from_orderbook,
    forecast_execution_cost,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_orderbook():
    """Simple orderbook with 3 levels on each side."""
    return {
        "bid_prices": [0.50, 0.49, 0.48],
        "bid_sizes": [1000.0, 2000.0, 3000.0],
        "ask_prices": [0.52, 0.53, 0.54],
        "ask_sizes": [1000.0, 2000.0, 3000.0],
    }


@pytest.fixture
def deep_orderbook():
    """Deeper orderbook for testing large orders."""
    return {
        "bid_prices": [0.50, 0.49, 0.48, 0.47, 0.46],
        "bid_sizes": [5000.0, 10000.0, 15000.0, 20000.0, 25000.0],
        "ask_prices": [0.52, 0.53, 0.54, 0.55, 0.56],
        "ask_sizes": [5000.0, 10000.0, 15000.0, 20000.0, 25000.0],
    }


@pytest.fixture
def mock_reader():
    """Create a mock QuestDBReader."""
    reader = MagicMock()
    return reader


# =============================================================================
# CostForecast Dataclass Tests
# =============================================================================


class TestCostForecastDataclass:
    """Tests for CostForecast dataclass."""

    def test_create_cost_forecast(self):
        """Should create CostForecast with all fields."""
        forecast = CostForecast(
            mid_price=0.51,
            expected_execution_price=0.52,
            slippage_bps=196.08,
            levels_consumed=2,
            fully_filled=True,
            unfilled_size=0.0,
            total_cost=5200.0,
        )

        assert forecast.mid_price == 0.51
        assert forecast.expected_execution_price == 0.52
        assert forecast.slippage_bps == 196.08
        assert forecast.levels_consumed == 2
        assert forecast.fully_filled is True
        assert forecast.unfilled_size == 0.0
        assert forecast.total_cost == 5200.0


# =============================================================================
# Buy Order Tests
# =============================================================================


class TestBuyOrders:
    """Tests for buy order cost estimation."""

    def test_buy_order_consumes_one_level_exactly(self, simple_orderbook):
        """Buy order that exactly fills first ask level."""
        # Order for 1000 at ask[0] = 0.52
        # Mid = (0.50 + 0.52) / 2 = 0.51
        # Expected price = 0.52
        # Slippage = (0.52 - 0.51) / 0.51 * 10000 = 196.08 bps

        forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="buy",
            **simple_orderbook,
        )

        assert forecast.mid_price == pytest.approx(0.51)
        assert forecast.expected_execution_price == pytest.approx(0.52)
        assert forecast.slippage_bps == pytest.approx(196.078, rel=0.01)
        assert forecast.levels_consumed == 1
        assert forecast.fully_filled is True
        assert forecast.unfilled_size == 0.0
        assert forecast.total_cost == pytest.approx(520.0)

    def test_buy_order_spans_multiple_levels(self, simple_orderbook):
        """Buy order that spans multiple ask levels."""
        # Order for 2500: fills 1000 at 0.52, 1500 at 0.53
        # VWAP = (1000*0.52 + 1500*0.53) / 2500 = (520 + 795) / 2500 = 0.526
        # Mid = 0.51
        # Slippage = (0.526 - 0.51) / 0.51 * 10000 = 313.73 bps

        forecast = estimate_slippage_from_orderbook(
            order_size=2500.0,
            side="buy",
            **simple_orderbook,
        )

        expected_vwap = (1000 * 0.52 + 1500 * 0.53) / 2500
        expected_slippage = (expected_vwap - 0.51) / 0.51 * 10000

        assert forecast.mid_price == pytest.approx(0.51)
        assert forecast.expected_execution_price == pytest.approx(expected_vwap)
        assert forecast.slippage_bps == pytest.approx(expected_slippage, rel=0.01)
        assert forecast.levels_consumed == 2
        assert forecast.fully_filled is True
        assert forecast.unfilled_size == 0.0
        assert forecast.total_cost == pytest.approx(1315.0)

    def test_buy_order_exhausts_all_liquidity(self, simple_orderbook):
        """Buy order larger than total ask liquidity (partial fill)."""
        # Total ask liquidity = 1000 + 2000 + 3000 = 6000
        # Order for 8000 -> fills 6000, unfilled 2000

        forecast = estimate_slippage_from_orderbook(
            order_size=8000.0,
            side="buy",
            **simple_orderbook,
        )

        # VWAP = (1000*0.52 + 2000*0.53 + 3000*0.54) / 6000
        expected_vwap = (1000 * 0.52 + 2000 * 0.53 + 3000 * 0.54) / 6000
        expected_slippage = (expected_vwap - 0.51) / 0.51 * 10000

        assert forecast.mid_price == pytest.approx(0.51)
        assert forecast.expected_execution_price == pytest.approx(expected_vwap)
        assert forecast.slippage_bps == pytest.approx(expected_slippage, rel=0.01)
        assert forecast.levels_consumed == 3
        assert forecast.fully_filled is False
        assert forecast.unfilled_size == pytest.approx(2000.0)

    def test_buy_order_equal_to_total_book_depth(self, simple_orderbook):
        """Buy order exactly equal to total ask depth."""
        # Total ask liquidity = 6000
        forecast = estimate_slippage_from_orderbook(
            order_size=6000.0,
            side="buy",
            **simple_orderbook,
        )

        assert forecast.fully_filled is True
        assert forecast.unfilled_size == 0.0
        assert forecast.levels_consumed == 3


# =============================================================================
# Sell Order Tests
# =============================================================================


class TestSellOrders:
    """Tests for sell order cost estimation."""

    def test_sell_order_consumes_one_level_exactly(self, simple_orderbook):
        """Sell order that exactly fills first bid level."""
        # Order for 1000 at bid[0] = 0.50
        # Mid = 0.51
        # For sell: slippage = (mid - exec_price) / mid * 10000
        # Slippage = (0.51 - 0.50) / 0.51 * 10000 = 196.08 bps

        forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="sell",
            **simple_orderbook,
        )

        assert forecast.mid_price == pytest.approx(0.51)
        assert forecast.expected_execution_price == pytest.approx(0.50)
        assert forecast.slippage_bps == pytest.approx(196.078, rel=0.01)
        assert forecast.levels_consumed == 1
        assert forecast.fully_filled is True
        assert forecast.unfilled_size == 0.0
        assert forecast.total_cost == pytest.approx(500.0)

    def test_sell_order_spans_multiple_levels(self, simple_orderbook):
        """Sell order that spans multiple bid levels."""
        # Order for 2500: fills 1000 at 0.50, 1500 at 0.49
        # VWAP = (1000*0.50 + 1500*0.49) / 2500 = 0.494
        # Mid = 0.51
        # Slippage = (0.51 - 0.494) / 0.51 * 10000

        forecast = estimate_slippage_from_orderbook(
            order_size=2500.0,
            side="sell",
            **simple_orderbook,
        )

        expected_vwap = (1000 * 0.50 + 1500 * 0.49) / 2500
        expected_slippage = (0.51 - expected_vwap) / 0.51 * 10000

        assert forecast.mid_price == pytest.approx(0.51)
        assert forecast.expected_execution_price == pytest.approx(expected_vwap)
        assert forecast.slippage_bps == pytest.approx(expected_slippage, rel=0.01)
        assert forecast.levels_consumed == 2
        assert forecast.fully_filled is True

    def test_sell_order_exhausts_all_liquidity(self, simple_orderbook):
        """Sell order larger than total bid liquidity (partial fill)."""
        # Total bid liquidity = 1000 + 2000 + 3000 = 6000
        # Order for 8000 -> fills 6000, unfilled 2000

        forecast = estimate_slippage_from_orderbook(
            order_size=8000.0,
            side="sell",
            **simple_orderbook,
        )

        assert forecast.fully_filled is False
        assert forecast.unfilled_size == pytest.approx(2000.0)
        assert forecast.levels_consumed == 3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_orderbook_returns_nan(self):
        """Empty orderbook should return NaN values."""
        forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="buy",
            bid_prices=[],
            bid_sizes=[],
            ask_prices=[],
            ask_sizes=[],
        )

        assert np.isnan(forecast.mid_price)
        assert np.isnan(forecast.expected_execution_price)
        assert np.isnan(forecast.slippage_bps)
        assert forecast.levels_consumed == 0
        assert forecast.fully_filled is False
        assert forecast.unfilled_size == 1000.0

    def test_empty_bid_side_only(self, simple_orderbook):
        """Empty bid side should return NaN."""
        forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="buy",
            bid_prices=[],
            bid_sizes=[],
            ask_prices=simple_orderbook["ask_prices"],
            ask_sizes=simple_orderbook["ask_sizes"],
        )

        assert np.isnan(forecast.mid_price)

    def test_empty_ask_side_only(self, simple_orderbook):
        """Empty ask side should return NaN."""
        forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="sell",
            bid_prices=simple_orderbook["bid_prices"],
            bid_sizes=simple_orderbook["bid_sizes"],
            ask_prices=[],
            ask_sizes=[],
        )

        assert np.isnan(forecast.mid_price)

    def test_zero_order_size(self, simple_orderbook):
        """Zero order size should return NaN slippage but be fully filled."""
        forecast = estimate_slippage_from_orderbook(
            order_size=0.0,
            side="buy",
            **simple_orderbook,
        )

        assert np.isnan(forecast.slippage_bps)
        assert forecast.fully_filled is True
        assert forecast.unfilled_size == 0.0
        assert forecast.total_cost == 0.0
        assert forecast.levels_consumed == 0

    def test_negative_order_size(self, simple_orderbook):
        """Negative order size should be treated as zero."""
        forecast = estimate_slippage_from_orderbook(
            order_size=-100.0,
            side="buy",
            **simple_orderbook,
        )

        assert np.isnan(forecast.slippage_bps)
        assert forecast.fully_filled is True

    def test_invalid_side_raises_error(self, simple_orderbook):
        """Invalid side should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid side"):
            estimate_slippage_from_orderbook(
                order_size=1000.0,
                side="invalid",
                **simple_orderbook,
            )

    def test_side_case_insensitive(self, simple_orderbook):
        """Side parameter should be case-insensitive."""
        forecast_lower = estimate_slippage_from_orderbook(
            order_size=1000.0, side="buy", **simple_orderbook
        )
        forecast_upper = estimate_slippage_from_orderbook(
            order_size=1000.0, side="BUY", **simple_orderbook
        )
        forecast_mixed = estimate_slippage_from_orderbook(
            order_size=1000.0, side="Buy", **simple_orderbook
        )

        assert forecast_lower.slippage_bps == forecast_upper.slippage_bps
        assert forecast_lower.slippage_bps == forecast_mixed.slippage_bps


# =============================================================================
# Slippage Sign Convention Tests
# =============================================================================


class TestSlippageConvention:
    """Tests for slippage sign convention (positive = cost)."""

    def test_buy_slippage_positive_when_paying_above_mid(self, simple_orderbook):
        """Buy order should have positive slippage (cost) when paying above mid."""
        forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="buy",
            **simple_orderbook,
        )

        # Buy at 0.52, mid is 0.51 -> paying more = cost
        assert forecast.slippage_bps > 0

    def test_sell_slippage_positive_when_receiving_below_mid(self, simple_orderbook):
        """Sell order should have positive slippage (cost) when receiving below mid."""
        forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="sell",
            **simple_orderbook,
        )

        # Sell at 0.50, mid is 0.51 -> receiving less = cost
        assert forecast.slippage_bps > 0

    def test_slippage_symmetric_for_small_orders(self, simple_orderbook):
        """Small buy and sell orders should have similar absolute slippage."""
        # With symmetric spread (1 tick each side from mid)
        # Small orders should face similar slippage magnitude

        buy_forecast = estimate_slippage_from_orderbook(
            order_size=100.0,
            side="buy",
            **simple_orderbook,
        )
        sell_forecast = estimate_slippage_from_orderbook(
            order_size=100.0,
            side="sell",
            **simple_orderbook,
        )

        # Both should be positive (cost)
        assert buy_forecast.slippage_bps > 0
        assert sell_forecast.slippage_bps > 0
        # And similar magnitude (within 10% for this symmetric book)
        assert abs(buy_forecast.slippage_bps - sell_forecast.slippage_bps) < 10


# =============================================================================
# Mid Price and Levels Consumed Tests
# =============================================================================


class TestMidPriceAndLevels:
    """Tests for mid price calculation and levels consumed."""

    def test_mid_price_calculation(self, simple_orderbook):
        """Mid price should be average of best bid and best ask."""
        forecast = estimate_slippage_from_orderbook(
            order_size=100.0,
            side="buy",
            **simple_orderbook,
        )

        expected_mid = (0.50 + 0.52) / 2
        assert forecast.mid_price == pytest.approx(expected_mid)

    def test_levels_consumed_count_accurate(self, deep_orderbook):
        """Levels consumed should accurately count levels used."""
        # Fill exactly first level
        forecast1 = estimate_slippage_from_orderbook(
            order_size=5000.0,
            side="buy",
            **deep_orderbook,
        )
        assert forecast1.levels_consumed == 1

        # Fill first two levels
        forecast2 = estimate_slippage_from_orderbook(
            order_size=15000.0,
            side="buy",
            **deep_orderbook,
        )
        assert forecast2.levels_consumed == 2

        # Fill all five levels
        forecast3 = estimate_slippage_from_orderbook(
            order_size=75000.0,
            side="buy",
            **deep_orderbook,
        )
        assert forecast3.levels_consumed == 5

    def test_partial_level_consumption(self, simple_orderbook):
        """Order that partially consumes a level should count it."""
        # Order for 500 partially consumes first level (1000)
        forecast = estimate_slippage_from_orderbook(
            order_size=500.0,
            side="buy",
            **simple_orderbook,
        )

        assert forecast.levels_consumed == 1
        assert forecast.fully_filled is True


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestForecastExecutionCost:
    """Tests for forecast_execution_cost convenience function."""

    def test_forecast_execution_cost_queries_reader(self, mock_reader):
        """Should query reader and return CostForecast."""
        as_of = datetime(2024, 1, 1, 12, 0, 0)

        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame(
            {
                "timestamp": [as_of - timedelta(seconds=5)],
                "bid_prices": [[0.50, 0.49]],
                "bid_sizes": [[1000.0, 2000.0]],
                "ask_prices": [[0.52, 0.53]],
                "ask_sizes": [[1000.0, 2000.0]],
            }
        )

        forecast = forecast_execution_cost(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            order_size=500.0,
            side="buy",
            as_of_time=as_of,
        )

        assert isinstance(forecast, CostForecast)
        assert forecast.mid_price == pytest.approx(0.51)
        mock_reader.query_orderbook_snapshots.assert_called_once()

    def test_forecast_execution_cost_uses_latest_snapshot(self, mock_reader):
        """Should use the most recent snapshot in the window."""
        as_of = datetime(2024, 1, 1, 12, 0, 0)

        # Two snapshots, later one should be used
        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame(
            {
                "timestamp": [
                    as_of - timedelta(seconds=30),
                    as_of - timedelta(seconds=5),
                ],
                "bid_prices": [[0.48], [0.50]],
                "bid_sizes": [[1000.0], [1000.0]],
                "ask_prices": [[0.54], [0.52]],
                "ask_sizes": [[1000.0], [1000.0]],
            }
        )

        forecast = forecast_execution_cost(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            order_size=500.0,
            side="buy",
            as_of_time=as_of,
        )

        # Should use second snapshot with mid = 0.51
        assert forecast.mid_price == pytest.approx(0.51)

    def test_forecast_execution_cost_raises_on_no_data(self, mock_reader):
        """Should raise ValueError when no orderbook data found."""
        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No orderbook data found"):
            forecast_execution_cost(
                reader=mock_reader,
                market_id="market-123",
                token_id="token-yes",
                order_size=1000.0,
                side="buy",
            )

    def test_forecast_execution_cost_respects_lookback(self, mock_reader):
        """Should respect lookback_seconds parameter."""
        as_of = datetime(2024, 1, 1, 12, 0, 0)
        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame()

        try:
            forecast_execution_cost(
                reader=mock_reader,
                market_id="market-123",
                token_id="token-yes",
                order_size=1000.0,
                side="buy",
                as_of_time=as_of,
                lookback_seconds=30,
            )
        except ValueError:
            pass  # Expected

        # Verify the query window is correct
        call_args = mock_reader.query_orderbook_snapshots.call_args
        assert call_args.kwargs["start_time"] == as_of - timedelta(seconds=30)


# =============================================================================
# Large Order Behavior Tests
# =============================================================================


class TestLargeOrders:
    """Tests for large order behavior."""

    def test_large_order_exhausting_all_liquidity(self, deep_orderbook):
        """Large order should report partial fill and all levels consumed."""
        total_liquidity = sum(deep_orderbook["ask_sizes"])  # 75000
        order_size = total_liquidity * 2  # 150000

        forecast = estimate_slippage_from_orderbook(
            order_size=order_size,
            side="buy",
            **deep_orderbook,
        )

        assert forecast.fully_filled is False
        assert forecast.unfilled_size == pytest.approx(total_liquidity)
        assert forecast.levels_consumed == 5

    def test_slippage_increases_with_order_size(self, deep_orderbook):
        """Larger orders should have higher slippage."""
        small_forecast = estimate_slippage_from_orderbook(
            order_size=1000.0,
            side="buy",
            **deep_orderbook,
        )
        large_forecast = estimate_slippage_from_orderbook(
            order_size=50000.0,
            side="buy",
            **deep_orderbook,
        )

        assert large_forecast.slippage_bps > small_forecast.slippage_bps
