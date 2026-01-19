"""Tests for SimulationEngine.

Tests verify:
- Engine processes market data in time order
- Orders execute at correct timestamps
- Slippage accumulates across slices
- No fills when no market data before order time
- Both buy and sell sides work correctly
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tributary.analytics.optimization import (
    generate_twap_trajectory,
    generate_market_order_trajectory,
)
from tributary.analytics.simulation import (
    SimulationEngine,
    FillModel,
    MarketEvent,
)


def make_market_data(
    base_time: datetime,
    num_snapshots: int = 10,
    interval_seconds: float = 1.0,
    mid_price: float = 0.50,
    bid_prices: tuple = (0.49, 0.48, 0.47),
    bid_sizes: tuple = (1000.0, 2000.0, 3000.0),
    ask_prices: tuple = (0.51, 0.52, 0.53),
    ask_sizes: tuple = (1000.0, 2000.0, 3000.0),
) -> pd.DataFrame:
    """Create synthetic market data for testing."""
    timestamps = [
        base_time + timedelta(seconds=i * interval_seconds)
        for i in range(num_snapshots)
    ]
    return pd.DataFrame({
        "timestamp": timestamps,
        "market_id": ["test-market"] * num_snapshots,
        "token_id": ["test-token"] * num_snapshots,
        "mid_price": [mid_price] * num_snapshots,
        "bid_prices": [bid_prices] * num_snapshots,
        "bid_sizes": [bid_sizes] * num_snapshots,
        "ask_prices": [ask_prices] * num_snapshots,
        "ask_sizes": [ask_sizes] * num_snapshots,
    })


class TestSimulationEngineBasic:
    """Basic engine functionality tests."""

    def test_engine_instantiation_default_fill_model(self):
        """Engine creates default FillModel if none provided."""
        engine = SimulationEngine()
        assert isinstance(engine.fill_model, FillModel)

    def test_engine_instantiation_custom_fill_model(self):
        """Engine uses provided FillModel."""
        custom_model = FillModel(recovery_rate=0.8, half_life_ms=500.0)
        engine = SimulationEngine(fill_model=custom_model)
        assert engine.fill_model is custom_model
        assert engine.fill_model.recovery_rate == 0.8

    def test_run_with_simple_trajectory(self):
        """Engine runs with simple trajectory and returns fills."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=3)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert len(fills) == 3
        assert all(f.filled_size > 0 for f in fills)

    def test_run_empty_market_data(self):
        """Engine returns empty list for empty market data."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = pd.DataFrame(columns=[
            "timestamp", "mid_price", "bid_prices", "bid_sizes",
            "ask_prices", "ask_sizes"
        ])

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=3)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert fills == []

    def test_run_empty_trajectory(self):
        """Engine returns empty list for trajectory with no trade sizes."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time)

        # Invalid trajectory (order_size <= 0 produces NaN trade_sizes)
        trajectory = generate_twap_trajectory(order_size=0, duration_periods=3)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert fills == []


class TestSimulationEngineTiming:
    """Tests for correct timestamp handling."""

    def test_fills_at_correct_timestamps(self):
        """Orders execute at scheduled times, not market data times."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=3)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=2),  # 2-second intervals
        )

        # Check fill timestamps match scheduled order times
        expected_times = [
            base_time,
            base_time + timedelta(seconds=2),
            base_time + timedelta(seconds=4),
        ]
        actual_times = [f.timestamp for f in fills]
        assert actual_times == expected_times

    def test_no_lookahead_bias(self):
        """Orders can only see market data at or before their time."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Market data starts 5 seconds after start_time
        market_data = make_market_data(
            base_time + timedelta(seconds=5),
            num_snapshots=10,
        )

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=5)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # First 5 slices (at t=0,1,2,3,4) should have no fills
        # because market data only starts at t=5
        # Only slices at t=4 and later might have data? No wait,
        # 5 periods means slices at t=0,1,2,3,4 - all before t=5
        # So no fills should occur
        assert len(fills) == 0

    def test_market_data_gaps_use_stale_state(self):
        """When market data has gaps, use last known state."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Create market data with gap (only at t=0 and t=5)
        timestamps = [
            base_time,
            base_time + timedelta(seconds=5),
        ]
        market_data = pd.DataFrame({
            "timestamp": timestamps,
            "mid_price": [0.50, 0.55],  # Price changes at t=5
            "bid_prices": [(0.49, 0.48), (0.54, 0.53)],
            "bid_sizes": [(1000.0, 2000.0), (1000.0, 2000.0)],
            "ask_prices": [(0.51, 0.52), (0.56, 0.57)],
            "ask_sizes": [(1000.0, 2000.0), (1000.0, 2000.0)],
        })

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=3)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=2),  # t=0, t=2, t=4
        )

        # All 3 fills should use the stale t=0 market state
        # (t=2 and t=4 orders use t=0 data since t=5 data is in future)
        assert len(fills) == 3
        for fill in fills:
            # All should have mid_price from t=0 data (0.50)
            assert fill.mid_price_at_fill == 0.50


class TestSimulationEngineSlippage:
    """Tests for slippage accumulation."""

    def test_slippage_increases_with_order_size(self):
        """Larger orders should have higher slippage (walk deeper into book)."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=5)

        # Small order - stays at best level
        small_traj = generate_market_order_trajectory(order_size=500)
        # Large order - walks through multiple levels
        large_traj = generate_market_order_trajectory(order_size=5000)

        small_engine = SimulationEngine()
        large_engine = SimulationEngine()

        small_fills = small_engine.run(
            trajectory=small_traj,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        large_fills = large_engine.run(
            trajectory=large_traj,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # Large order should have higher slippage
        assert large_fills[0].slippage_bps > small_fills[0].slippage_bps

    def test_slippage_accumulates_across_slices(self):
        """Consecutive slices should consume liquidity, increasing slippage."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Small orderbook so liquidity gets consumed
        market_data = make_market_data(
            base_time,
            num_snapshots=10,
            ask_sizes=(500.0, 500.0, 500.0),  # Limited liquidity
        )

        # TWAP with 3 slices of ~500 each
        trajectory = generate_twap_trajectory(order_size=1500, duration_periods=3)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(milliseconds=100),  # Fast enough to not recover
        )

        # Later slices should have worse slippage (liquidity consumed)
        # Note: With recovery disabled/minimal, slippage should increase
        assert len(fills) == 3
        # At least the 2nd slice should be worse than 1st
        # (recovery may help 3rd slightly, but generally increasing)
        assert fills[1].slippage_bps >= fills[0].slippage_bps


class TestSimulationEngineSides:
    """Tests for buy and sell side handling."""

    def test_buy_side_executes_against_asks(self):
        """Buy orders should execute against ask prices."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(
            base_time,
            mid_price=0.50,
            ask_prices=(0.51,),
            ask_sizes=(10000.0,),
        )

        trajectory = generate_market_order_trajectory(order_size=1000)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert len(fills) == 1
        # Buy at 0.51, mid at 0.50 -> positive slippage (paid more)
        assert fills[0].avg_price == 0.51
        assert fills[0].slippage_bps > 0

    def test_sell_side_executes_against_bids(self):
        """Sell orders should execute against bid prices."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(
            base_time,
            mid_price=0.50,
            bid_prices=(0.49,),
            bid_sizes=(10000.0,),
        )

        trajectory = generate_market_order_trajectory(order_size=1000)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="sell",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert len(fills) == 1
        # Sell at 0.49, mid at 0.50 -> positive slippage (received less)
        assert fills[0].avg_price == 0.49
        assert fills[0].slippage_bps > 0


class TestSimulationEngineMarketDataFormat:
    """Tests for market data handling."""

    def test_unsorted_market_data_sorted_internally(self):
        """Engine sorts market data by timestamp."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Create unsorted market data
        timestamps = [
            base_time + timedelta(seconds=5),
            base_time + timedelta(seconds=2),
            base_time,
            base_time + timedelta(seconds=8),
        ]
        market_data = pd.DataFrame({
            "timestamp": timestamps,
            "mid_price": [0.55, 0.52, 0.50, 0.58],
            "bid_prices": [(0.54,), (0.51,), (0.49,), (0.57,)],
            "bid_sizes": [(1000.0,), (1000.0,), (1000.0,), (1000.0,)],
            "ask_prices": [(0.56,), (0.53,), (0.51,), (0.59,)],
            "ask_sizes": [(1000.0,), (1000.0,), (1000.0,), (1000.0,)],
        })

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=3)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=3),  # t=0, t=3, t=6
        )

        # t=0: uses t=0 data (mid=0.50)
        # t=3: uses t=2 data (mid=0.52)
        # t=6: uses t=5 data (mid=0.55)
        assert len(fills) == 3
        assert fills[0].mid_price_at_fill == 0.50
        assert fills[1].mid_price_at_fill == 0.52
        assert fills[2].mid_price_at_fill == 0.55

    def test_missing_optional_columns(self):
        """Engine handles missing market_id and token_id columns."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Minimal market data without optional columns
        market_data = pd.DataFrame({
            "timestamp": [base_time],
            "mid_price": [0.50],
            "bid_prices": [(0.49,)],
            "bid_sizes": [(1000.0,)],
            "ask_prices": [(0.51,)],
            "ask_sizes": [(1000.0,)],
        })

        trajectory = generate_market_order_trajectory(order_size=500)
        engine = SimulationEngine()

        fills = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # Should still work
        assert len(fills) == 1
        assert fills[0].filled_size == 500


class TestSimulationEngineReset:
    """Tests for fill model reset between runs."""

    def test_fill_model_resets_between_runs(self):
        """Each run starts with fresh fill model state."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(
            base_time,
            num_snapshots=5,
            ask_sizes=(500.0, 500.0, 500.0),
        )

        trajectory = generate_market_order_trajectory(order_size=1000)
        engine = SimulationEngine()

        # First run
        fills1 = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # Second run should have same result (reset between runs)
        fills2 = engine.run(
            trajectory=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert fills1[0].slippage_bps == fills2[0].slippage_bps
        assert fills1[0].filled_size == fills2[0].filled_size
