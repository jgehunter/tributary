"""Tests for StrategyRunner.

Tests verify:
- Runner executes multiple strategies
- Each strategy gets isolated execution (no cross-impact)
- Results order matches input order
- Different strategy sizes execute correctly
- Empty strategies list returns empty results
- Market order vs TWAP shows different slippage patterns
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import numpy as np

from tributary.analytics.optimization import (
    generate_twap_trajectory,
    generate_market_order_trajectory,
    generate_vwap_trajectory,
)
from tributary.analytics.simulation import (
    StrategyRunner,
    StrategyRun,
    SimulationEngine,
    FillModel,
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


class TestStrategyRunnerBasic:
    """Basic runner functionality tests."""

    def test_runner_instantiation_default_params(self):
        """Runner uses default FillModel parameters."""
        runner = StrategyRunner()
        assert runner.recovery_rate == 0.5
        assert runner.half_life_ms == 1000.0

    def test_runner_instantiation_custom_params(self):
        """Runner accepts custom FillModel parameters."""
        runner = StrategyRunner(recovery_rate=0.8, half_life_ms=500.0)
        assert runner.recovery_rate == 0.8
        assert runner.half_life_ms == 500.0

    def test_run_single_strategy(self):
        """Runner executes single strategy correctly."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        strategy = generate_twap_trajectory(order_size=1000, duration_periods=3)
        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[strategy],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert len(results) == 1
        assert isinstance(results[0], StrategyRun)
        assert results[0].trajectory is strategy
        assert len(results[0].fills) == 3

    def test_run_multiple_strategies(self):
        """Runner executes multiple strategies."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        twap = generate_twap_trajectory(order_size=1000, duration_periods=3)
        market = generate_market_order_trajectory(order_size=1000)
        vwap = generate_vwap_trajectory(order_size=1000, volume_profile=np.array([1, 2, 1]))

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[twap, market, vwap],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert len(results) == 3

    def test_empty_strategies_returns_empty(self):
        """Runner returns empty list for empty strategies."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time)

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert results == []


class TestStrategyRunnerIsolation:
    """Tests for isolated execution (no cross-strategy impact)."""

    def test_strategies_have_isolated_execution(self):
        """Each strategy should see same initial market state."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Limited liquidity orderbook
        market_data = make_market_data(
            base_time,
            num_snapshots=10,
            ask_sizes=(500.0, 500.0, 500.0),
        )

        # Two identical strategies
        strategy1 = generate_market_order_trajectory(order_size=1000)
        strategy2 = generate_market_order_trajectory(order_size=1000)

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[strategy1, strategy2],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # Both should have identical slippage (isolated execution)
        assert results[0].fills[0].slippage_bps == results[1].fills[0].slippage_bps
        assert results[0].fills[0].filled_size == results[1].fills[0].filled_size

    def test_no_cross_strategy_liquidity_consumption(self):
        """Strategy 1's liquidity consumption shouldn't affect strategy 2."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Very limited liquidity
        market_data = make_market_data(
            base_time,
            num_snapshots=10,
            ask_sizes=(100.0,),  # Only 100 at best level
        )

        # First strategy takes all liquidity at best level
        large = generate_market_order_trajectory(order_size=200)
        # Second strategy would fail if liquidity was shared
        small = generate_market_order_trajectory(order_size=50)

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[large, small],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # Small strategy should still fill at best level (isolated)
        assert results[1].fills[0].filled_size == 50
        # Small strategy should have lower slippage (didn't see large's impact)
        assert results[1].fills[0].levels_consumed == 1


class TestStrategyRunnerOrder:
    """Tests for result ordering."""

    def test_results_order_matches_input(self):
        """Results should be in same order as input strategies."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        twap = generate_twap_trajectory(order_size=1000, duration_periods=5)
        market = generate_market_order_trajectory(order_size=1000)
        vwap = generate_vwap_trajectory(order_size=1000, volume_profile=np.array([1, 2, 3, 2, 1]))

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[twap, market, vwap],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert results[0].trajectory.strategy_name == "twap"
        assert results[1].trajectory.strategy_name == "market_order"
        assert results[2].trajectory.strategy_name == "vwap"


class TestStrategyRunnerDifferentSizes:
    """Tests for strategies with different sizes."""

    def test_different_order_sizes(self):
        """Strategies with different sizes execute correctly."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        small = generate_market_order_trajectory(order_size=100)
        medium = generate_market_order_trajectory(order_size=500)
        large = generate_market_order_trajectory(order_size=2000)

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[small, medium, large],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert results[0].total_filled == 100
        assert results[1].total_filled == 500
        assert results[2].total_filled == 2000

    def test_different_duration_periods(self):
        """Strategies with different durations execute correctly."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=20)

        short = generate_twap_trajectory(order_size=1000, duration_periods=2)
        medium = generate_twap_trajectory(order_size=1000, duration_periods=5)
        long = generate_twap_trajectory(order_size=1000, duration_periods=10)

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[short, medium, long],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        assert len(results[0].fills) == 2
        assert len(results[1].fills) == 5
        assert len(results[2].fills) == 10


class TestStrategyRunnerSlippagePatterns:
    """Tests for strategy slippage comparison."""

    def test_market_order_vs_twap_slippage(self):
        """Market order should have higher slippage than TWAP."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Limited liquidity to show impact difference
        market_data = make_market_data(
            base_time,
            num_snapshots=10,
            ask_sizes=(500.0, 1000.0, 1500.0),
        )

        # Market order: execute 1500 immediately (walks through levels)
        market = generate_market_order_trajectory(order_size=1500)

        # TWAP: execute same 1500 over 3 slices (500 each)
        twap = generate_twap_trajectory(order_size=1500, duration_periods=3)

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[market, twap],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=100),  # Long interval for full recovery
        )

        market_slippage = results[0].weighted_slippage_bps
        twap_slippage = results[1].weighted_slippage_bps

        # Market order should have higher slippage (walks deeper into book)
        assert market_slippage > twap_slippage

    def test_aggressive_vs_patient_execution(self):
        """Patient execution (more slices) should have lower slippage."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Limited liquidity
        market_data = make_market_data(
            base_time,
            num_snapshots=20,
            ask_sizes=(200.0, 400.0, 600.0),
        )

        # Aggressive: 2 large slices
        aggressive = generate_twap_trajectory(order_size=600, duration_periods=2)

        # Patient: 6 small slices
        patient = generate_twap_trajectory(order_size=600, duration_periods=6)

        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[aggressive, patient],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=2),  # 2 seconds between slices
        )

        # Patient should have lower weighted slippage
        # (smaller slices = stay at better levels)
        aggressive_slippage = results[0].weighted_slippage_bps
        patient_slippage = results[1].weighted_slippage_bps

        assert patient_slippage <= aggressive_slippage


class TestStrategyRunDataclass:
    """Tests for StrategyRun dataclass properties."""

    def test_total_filled_property(self):
        """total_filled sums all fill sizes."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=5)
        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[trajectory],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        result = results[0]
        assert result.total_filled == sum(f.filled_size for f in result.fills)
        assert result.total_filled == 1000  # All 5 slices of 200

    def test_fill_rate_property(self):
        """fill_rate calculates percentage filled."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        trajectory = generate_market_order_trajectory(order_size=1000)
        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[trajectory],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        result = results[0]
        # Full liquidity available, should be 100%
        assert result.fill_rate == 100.0

    def test_weighted_slippage_property(self):
        """weighted_slippage_bps calculates size-weighted average."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=5)
        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[trajectory],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        result = results[0]
        # Manually calculate weighted slippage
        expected = sum(f.slippage_bps * f.filled_size for f in result.fills) / result.total_filled
        assert result.weighted_slippage_bps == pytest.approx(expected)

    def test_weighted_slippage_empty_fills(self):
        """weighted_slippage_bps returns NaN for empty fills."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Market data starts after all order times (no fills)
        market_data = make_market_data(
            base_time + timedelta(seconds=100),
            num_snapshots=5,
        )

        trajectory = generate_twap_trajectory(order_size=1000, duration_periods=3)
        runner = StrategyRunner()

        results = runner.run_strategies(
            strategies=[trajectory],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        result = results[0]
        assert len(result.fills) == 0
        assert np.isnan(result.weighted_slippage_bps)


class TestStrategyRunnerSingleConvenience:
    """Tests for run_single convenience method."""

    def test_run_single_method(self):
        """run_single returns single StrategyRun directly."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        market_data = make_market_data(base_time, num_snapshots=10)

        trajectory = generate_market_order_trajectory(order_size=500)
        runner = StrategyRunner()

        result = runner.run_single(
            strategy=trajectory,
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # Returns StrategyRun directly, not list
        assert isinstance(result, StrategyRun)
        assert result.trajectory is trajectory
        assert len(result.fills) == 1


class TestStrategyRunnerRecoveryParams:
    """Tests for custom recovery parameters."""

    def test_custom_recovery_affects_fill_model(self):
        """Custom recovery parameters are used in fill models."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Larger liquidity pool with multiple levels
        market_data = make_market_data(
            base_time,
            num_snapshots=10,
            ask_sizes=(300.0, 300.0, 400.0),  # 1000 total across 3 levels
        )

        # Execute 500 in 2 slices (250 each) - will consume from best level
        trajectory = generate_twap_trajectory(order_size=500, duration_periods=2)

        # Slow recovery runner
        slow_recovery = StrategyRunner(recovery_rate=0.1, half_life_ms=10000.0)
        # Fast recovery runner
        fast_recovery = StrategyRunner(recovery_rate=0.99, half_life_ms=100.0)

        slow_results = slow_recovery.run_strategies(
            strategies=[trajectory],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),  # 1 second = 1000ms
        )

        fast_results = fast_recovery.run_strategies(
            strategies=[trajectory],
            market_data=market_data,
            side="buy",
            start_time=base_time,
            interval=timedelta(seconds=1),
        )

        # First slice is same for both (no prior consumption)
        assert (
            slow_results[0].fills[0].slippage_bps ==
            fast_results[0].fills[0].slippage_bps
        )

        # Both should have fills for second slice
        assert len(slow_results[0].fills) == 2
        assert len(fast_results[0].fills) == 2

        # Fast recovery should have better or equal slippage on second slice
        # (more liquidity recovered at best level)
        assert (
            fast_results[0].fills[1].slippage_bps <=
            slow_results[0].fills[1].slippage_bps
        )
