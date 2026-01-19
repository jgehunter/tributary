"""Tests for FillModel with liquidity consumption and recovery.

Tests that the fill model correctly:
- Walks the orderbook to fill orders
- Tracks consumed liquidity per level
- Models liquidity recovery over time
- Differentiates aggressive vs patient execution strategies
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from tributary.analytics.simulation import FillEvent, FillModel, MarketEvent, OrderEvent


def make_market(
    mid_price: float = 0.50,
    bid_prices: tuple[float, ...] = (0.49, 0.48, 0.47),
    bid_sizes: tuple[float, ...] = (1000.0, 2000.0, 3000.0),
    ask_prices: tuple[float, ...] = (0.51, 0.52, 0.53),
    ask_sizes: tuple[float, ...] = (1000.0, 2000.0, 3000.0),
    timestamp: datetime | None = None,
) -> MarketEvent:
    """Helper to create a MarketEvent for testing."""
    return MarketEvent(
        timestamp=timestamp or datetime.now(timezone.utc),
        market_id="test",
        token_id="test",
        mid_price=mid_price,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )


def make_order(
    size: float,
    side: str = "buy",
    strategy_name: str = "test",
    slice_index: int = 0,
    timestamp: datetime | None = None,
) -> OrderEvent:
    """Helper to create an OrderEvent for testing."""
    return OrderEvent(
        timestamp=timestamp or datetime.now(timezone.utc),
        strategy_name=strategy_name,
        slice_index=slice_index,
        size=size,
        side=side,
    )


class TestFillModelBasics:
    """Basic fill model functionality tests."""

    def test_basic_buy_fill_single_level(self) -> None:
        """Test basic fill against single orderbook level."""
        model = FillModel()
        market = make_market()
        order = make_order(size=500.0, side="buy")

        fill = model.execute(order, market)

        assert fill.filled_size == 500.0
        assert fill.requested_size == 500.0
        assert fill.avg_price == 0.51  # First ask level
        assert fill.levels_consumed == 1
        assert fill.mid_price_at_fill == 0.50
        # Slippage: (0.51 - 0.50) / 0.50 * 10000 = 200 bps
        assert fill.slippage_bps == pytest.approx(200.0, rel=1e-6)

    def test_basic_sell_fill_single_level(self) -> None:
        """Test basic sell fill against bid side."""
        model = FillModel()
        market = make_market()
        order = make_order(size=500.0, side="sell")

        fill = model.execute(order, market)

        assert fill.filled_size == 500.0
        assert fill.avg_price == 0.49  # First bid level
        assert fill.levels_consumed == 1
        # Slippage: (0.50 - 0.49) / 0.50 * 10000 = 200 bps
        assert fill.slippage_bps == pytest.approx(200.0, rel=1e-6)

    def test_fill_across_multiple_levels(self) -> None:
        """Test fill that consumes multiple orderbook levels."""
        model = FillModel()
        market = make_market()
        # Order larger than first level (1000)
        order = make_order(size=2500.0, side="buy")

        fill = model.execute(order, market)

        assert fill.filled_size == 2500.0
        assert fill.levels_consumed == 2
        # VWAP: (1000*0.51 + 1500*0.52) / 2500 = 1290 / 2500 = 0.516
        expected_avg = (1000 * 0.51 + 1500 * 0.52) / 2500
        assert fill.avg_price == pytest.approx(expected_avg, rel=1e-6)

    def test_returns_fill_event_type(self) -> None:
        """Test that execute returns a FillEvent."""
        model = FillModel()
        market = make_market()
        order = make_order(size=100.0)

        fill = model.execute(order, market)

        assert isinstance(fill, FillEvent)
        assert fill.strategy_name == "test"
        assert fill.slice_index == 0


class TestSlippageScaling:
    """Tests for slippage scaling with order size (market impact)."""

    def test_slippage_increases_with_order_size(self) -> None:
        """Test that larger orders have higher slippage."""
        model = FillModel()
        market = make_market()

        # Small order (single level)
        small_order = make_order(size=500.0)
        small_fill = model.execute(small_order, market)

        model.reset()

        # Large order (multiple levels)
        large_order = make_order(size=4000.0)
        large_fill = model.execute(large_order, market)

        assert large_fill.slippage_bps > small_fill.slippage_bps
        assert large_fill.levels_consumed > small_fill.levels_consumed

    def test_very_large_order_partial_fill(self) -> None:
        """Test partial fill when order exceeds available liquidity."""
        model = FillModel()
        # Total liquidity: 1000 + 2000 + 3000 = 6000
        market = make_market()
        order = make_order(size=10000.0, side="buy")

        fill = model.execute(order, market)

        assert fill.filled_size == 6000.0  # Max available
        assert fill.requested_size == 10000.0
        assert fill.levels_consumed == 3
        assert fill.filled_size < fill.requested_size


class TestLiquidityConsumption:
    """Tests for liquidity consumption tracking."""

    def test_consecutive_orders_consume_liquidity(self) -> None:
        """Test that consecutive orders deplete liquidity."""
        model = FillModel()
        ts = datetime.now(timezone.utc)
        market = make_market(timestamp=ts)

        # First order consumes first level
        order1 = make_order(size=1000.0, timestamp=ts)
        fill1 = model.execute(order1, market)

        assert fill1.filled_size == 1000.0
        assert fill1.levels_consumed == 1
        assert fill1.avg_price == 0.51

        # Second order must start from second level
        # (no time elapsed, no recovery)
        order2 = make_order(size=500.0, timestamp=ts)
        fill2 = model.execute(order2, market)

        assert fill2.filled_size == 500.0
        assert fill2.avg_price == 0.52  # Second level
        assert fill2.slippage_bps > fill1.slippage_bps

    def test_partial_level_consumption(self) -> None:
        """Test partial consumption of a level."""
        model = FillModel()
        ts = datetime.now(timezone.utc)
        market = make_market(timestamp=ts)

        # First order partially consumes first level
        order1 = make_order(size=300.0, timestamp=ts)
        fill1 = model.execute(order1, market)

        assert fill1.filled_size == 300.0
        assert fill1.avg_price == 0.51

        # Second order uses remaining from first level
        order2 = make_order(size=900.0, timestamp=ts)
        fill2 = model.execute(order2, market)

        # 700 remaining at 0.51, 200 from 0.52
        assert fill2.filled_size == 900.0
        expected_avg = (700 * 0.51 + 200 * 0.52) / 900
        assert fill2.avg_price == pytest.approx(expected_avg, rel=1e-6)

    def test_buy_and_sell_track_separately(self) -> None:
        """Test that buy and sell sides track liquidity independently."""
        model = FillModel()
        ts = datetime.now(timezone.utc)
        market = make_market(timestamp=ts)

        # Buy consumes ask side
        buy_order = make_order(size=1000.0, side="buy", timestamp=ts)
        model.execute(buy_order, market)

        # Sell should still have full bid liquidity
        sell_order = make_order(size=500.0, side="sell", timestamp=ts)
        sell_fill = model.execute(sell_order, market)

        assert sell_fill.filled_size == 500.0
        assert sell_fill.avg_price == 0.49  # First bid level (not consumed)


class TestLiquidityRecovery:
    """Tests for liquidity recovery over time."""

    def test_partial_recovery_after_time(self) -> None:
        """Test liquidity partially recovers after time elapses."""
        model = FillModel(recovery_rate=0.5, half_life_ms=1000.0)
        t0 = datetime.now(timezone.utc)
        market = make_market(timestamp=t0)

        # Consume all of first level
        order1 = make_order(size=1000.0, timestamp=t0)
        fill1 = model.execute(order1, market)
        assert fill1.avg_price == 0.51

        # After 1 half-life (1s), 50% of 50% recoverable = 25% recovered
        t1 = t0 + timedelta(milliseconds=1000)
        order2 = make_order(size=300.0, timestamp=t1)
        fill2 = model.execute(order2, market)

        # Some liquidity recovered at first level
        # Should be mix of level 1 and level 2
        assert fill2.filled_size == 300.0
        # Price should be between 0.51 and 0.52
        assert 0.51 <= fill2.avg_price <= 0.52

    def test_no_recovery_with_zero_time(self) -> None:
        """Test no recovery when no time has elapsed."""
        model = FillModel(recovery_rate=1.0, half_life_ms=100.0)
        ts = datetime.now(timezone.utc)
        market = make_market(timestamp=ts)

        # Consume first level
        order1 = make_order(size=1000.0, timestamp=ts)
        fill1 = model.execute(order1, market)

        # No time elapsed, should not recover
        order2 = make_order(size=500.0, timestamp=ts)
        fill2 = model.execute(order2, market)

        # Must use second level
        assert fill2.avg_price == 0.52

    def test_full_recovery_after_long_time(self) -> None:
        """Test liquidity fully recovers after sufficient time."""
        model = FillModel(recovery_rate=1.0, half_life_ms=100.0)
        t0 = datetime.now(timezone.utc)
        market = make_market(timestamp=t0)

        # Consume first level
        order1 = make_order(size=1000.0, timestamp=t0)
        model.execute(order1, market)

        # After many half-lives, should be ~fully recovered
        t1 = t0 + timedelta(seconds=10)  # 100 half-lives
        order2 = make_order(size=500.0, timestamp=t1)
        fill2 = model.execute(order2, market)

        # Should be able to use first level again
        assert fill2.avg_price == pytest.approx(0.51, rel=1e-2)


class TestResetBehavior:
    """Tests for reset() method."""

    def test_reset_clears_consumed_state(self) -> None:
        """Test that reset() clears all consumed liquidity."""
        model = FillModel()
        ts = datetime.now(timezone.utc)
        market = make_market(timestamp=ts)

        # Consume first level
        order1 = make_order(size=1000.0, timestamp=ts)
        model.execute(order1, market)

        # Reset
        model.reset()

        # First level should be available again
        order2 = make_order(size=500.0, timestamp=ts)
        fill2 = model.execute(order2, market)

        assert fill2.avg_price == 0.51  # First level restored


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_orderbook(self) -> None:
        """Test handling of empty orderbook."""
        model = FillModel()
        market = MarketEvent(
            timestamp=datetime.now(timezone.utc),
            market_id="test",
            token_id="test",
            mid_price=float("nan"),
            bid_prices=(),
            bid_sizes=(),
            ask_prices=(),
            ask_sizes=(),
        )
        order = make_order(size=1000.0)

        fill = model.execute(order, market)

        assert fill.filled_size == 0.0
        assert fill.levels_consumed == 0
        assert math.isnan(fill.avg_price)
        assert math.isnan(fill.slippage_bps)

    def test_zero_order_size(self) -> None:
        """Test handling of zero-size order."""
        model = FillModel()
        market = make_market()
        order = make_order(size=0.0)

        fill = model.execute(order, market)

        assert fill.filled_size == 0.0
        assert fill.levels_consumed == 0
        assert math.isnan(fill.slippage_bps)

    def test_invalid_side_raises(self) -> None:
        """Test that invalid side raises ValueError."""
        model = FillModel()
        market = make_market()
        order = OrderEvent(
            timestamp=datetime.now(timezone.utc),
            strategy_name="test",
            slice_index=0,
            size=100.0,
            side="invalid",
        )

        with pytest.raises(ValueError, match="Invalid side"):
            model.execute(order, market)


class TestStrategyDifferentiation:
    """Tests that model differentiates aggressive vs patient strategies."""

    def test_large_order_higher_slippage_than_twap(self) -> None:
        """Test that a single large order has higher slippage than TWAP slices.

        This is the key property: market impact makes aggressive execution costly.
        TWAP-style patient execution (assuming orderbook recovers) should be cheaper.
        """
        # Market with 3 levels of liquidity
        market = make_market(
            ask_prices=(0.51, 0.52, 0.53),
            ask_sizes=(1000.0, 2000.0, 3000.0),
        )

        # Single large order
        model_large = FillModel()
        large_order = make_order(size=5000.0, side="buy")
        large_fill = model_large.execute(large_order, market)

        # TWAP: 5 slices of 1000, with reset between (assumes full recovery)
        model_twap = FillModel()
        total_twap_slippage = 0.0
        total_twap_filled = 0.0

        for i in range(5):
            model_twap.reset()  # Simulate full orderbook recovery
            small_order = make_order(size=1000.0, side="buy", slice_index=i)
            small_fill = model_twap.execute(small_order, market)
            total_twap_slippage += small_fill.slippage_bps * small_fill.filled_size
            total_twap_filled += small_fill.filled_size

        weighted_twap_slippage = total_twap_slippage / total_twap_filled

        # Large order should have higher slippage than TWAP
        assert large_fill.slippage_bps > weighted_twap_slippage

    def test_consecutive_without_recovery_worse_than_with_reset(self) -> None:
        """Test that rapid consecutive orders (no recovery) are worse than spaced."""
        ts = datetime.now(timezone.utc)
        market = make_market(timestamp=ts)

        # Rapid consecutive (no reset, no time gap)
        model_rapid = FillModel()
        total_rapid_cost = 0.0
        for i in range(3):
            order = make_order(size=800.0, side="buy", timestamp=ts, slice_index=i)
            fill = model_rapid.execute(order, market)
            total_rapid_cost += fill.avg_price * fill.filled_size

        # With reset between each (simulating full recovery)
        model_spaced = FillModel()
        total_spaced_cost = 0.0
        for i in range(3):
            model_spaced.reset()
            order = make_order(size=800.0, side="buy", timestamp=ts, slice_index=i)
            fill = model_spaced.execute(order, market)
            total_spaced_cost += fill.avg_price * fill.filled_size

        # Rapid execution should cost more (worse prices)
        assert total_rapid_cost > total_spaced_cost


class TestSellSide:
    """Tests specifically for sell order execution."""

    def test_sell_walks_bid_side(self) -> None:
        """Test that sell orders walk the bid side correctly."""
        model = FillModel()
        market = make_market(
            bid_prices=(0.49, 0.48, 0.47),
            bid_sizes=(500.0, 1000.0, 1500.0),
        )
        # Order that spans two levels
        order = make_order(size=1200.0, side="sell")

        fill = model.execute(order, market)

        assert fill.filled_size == 1200.0
        assert fill.levels_consumed == 2
        # VWAP: (500*0.49 + 700*0.48) / 1200
        expected_avg = (500 * 0.49 + 700 * 0.48) / 1200
        assert fill.avg_price == pytest.approx(expected_avg, rel=1e-6)

    def test_sell_slippage_positive(self) -> None:
        """Test that sell slippage is positive (unfavorable)."""
        model = FillModel()
        market = make_market()
        order = make_order(size=500.0, side="sell")

        fill = model.execute(order, market)

        # For sells: slippage = (mid - avg_price) / mid
        # mid = 0.50, avg_price = 0.49
        # slippage = (0.50 - 0.49) / 0.50 * 10000 = 200 bps
        assert fill.slippage_bps > 0
        assert fill.slippage_bps == pytest.approx(200.0, rel=1e-6)

    def test_sell_consumes_liquidity_on_bid_side(self) -> None:
        """Test that sell orders consume bid side liquidity."""
        model = FillModel()
        ts = datetime.now(timezone.utc)
        market = make_market(timestamp=ts)

        # Consume first bid level
        order1 = make_order(size=1000.0, side="sell", timestamp=ts)
        model.execute(order1, market)

        # Next sell should hit second level
        order2 = make_order(size=500.0, side="sell", timestamp=ts)
        fill2 = model.execute(order2, market)

        assert fill2.avg_price == 0.48  # Second bid level
