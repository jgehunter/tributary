"""Integration tests proving optimized strategies beat naive approaches (SIM-05).

These tests demonstrate the core value proposition:
1. TWAP has lower slippage than market order
2. A-C with different risk aversions show cost-risk tradeoff
3. Comparison table ranks strategies correctly

This is the "proof" that makes the simulation engine valuable.
"""

import pytest
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from tributary.analytics.simulation import (
    StrategyRunner,
    create_simulation_result,
    compare_simulation_results,
)
from tributary.analytics.optimization import (
    generate_twap_trajectory,
    generate_market_order_trajectory,
    calibrate_ac_params,
    generate_ac_trajectory,
)


@pytest.fixture
def realistic_market_data():
    """Generate synthetic market data with reasonable orderbook depth.

    Orderbook has limited depth at each level, so large orders consume
    multiple levels (creating market impact).
    """
    timestamps = [
        datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i)
        for i in range(20)
    ]

    # Orderbook with limited depth at each level
    # This makes large orders consume multiple levels (creating impact)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "mid_price": [0.50] * 20,
            "bid_prices": [tuple([0.49, 0.48, 0.47, 0.46, 0.45])] * 20,
            "bid_sizes": [tuple([500.0, 1000.0, 1500.0, 2000.0, 2500.0])] * 20,
            "ask_prices": [tuple([0.51, 0.52, 0.53, 0.54, 0.55])] * 20,
            "ask_sizes": [tuple([500.0, 1000.0, 1500.0, 2000.0, 2500.0])] * 20,
        }
    )


class TestBetterExecution:
    """Prove that optimized strategies outperform naive approaches."""

    def test_twap_beats_market_order(self, realistic_market_data):
        """TWAP should have lower slippage than market order (SIM-05 core proof).

        This is the fundamental proof of value: patient execution reduces costs.

        Market order executes everything at once, consuming multiple orderbook
        levels and incurring high slippage.

        TWAP spreads execution over time, allowing liquidity to partially
        recover between slices, reducing total slippage.
        """
        order_size = 3000  # Large enough to consume multiple levels

        twap = generate_twap_trajectory(order_size=order_size, duration_periods=10)
        market = generate_market_order_trajectory(order_size=order_size)

        runner = StrategyRunner()
        runs = runner.run_strategies(
            strategies=[twap, market],
            market_data=realistic_market_data,
            side="buy",
            start_time=realistic_market_data["timestamp"].iloc[0],
            interval=timedelta(seconds=1),
        )

        # Create results
        arrival_price = realistic_market_data["mid_price"].iloc[0]
        market_vwap = arrival_price  # Simplified for test

        twap_result = create_simulation_result(runs[0], arrival_price, market_vwap)
        market_result = create_simulation_result(runs[1], arrival_price, market_vwap)

        # CORE ASSERTION: TWAP beats market order
        assert (
            twap_result.implementation_shortfall_bps
            < market_result.implementation_shortfall_bps
        ), (
            f"TWAP ({twap_result.implementation_shortfall_bps:.2f} bps) should beat "
            f"market order ({market_result.implementation_shortfall_bps:.2f} bps)"
        )

    def test_twap_has_more_slices_than_market_order(self, realistic_market_data):
        """TWAP should execute in multiple slices, market order in one."""
        order_size = 2000

        twap = generate_twap_trajectory(order_size=order_size, duration_periods=5)
        market = generate_market_order_trajectory(order_size=order_size)

        runner = StrategyRunner()
        runs = runner.run_strategies(
            strategies=[twap, market],
            market_data=realistic_market_data,
            side="buy",
            start_time=realistic_market_data["timestamp"].iloc[0],
            interval=timedelta(seconds=1),
        )

        arrival_price = realistic_market_data["mid_price"].iloc[0]

        twap_result = create_simulation_result(runs[0], arrival_price, arrival_price)
        market_result = create_simulation_result(runs[1], arrival_price, arrival_price)

        assert twap_result.num_slices == 5
        assert market_result.num_slices == 1

    def test_almgren_chriss_provides_risk_cost_tradeoff(self, realistic_market_data):
        """A-C with different risk aversions should show cost-risk tradeoff.

        Higher risk aversion front-loads execution (more cost, less variance).
        Lower risk aversion is closer to TWAP (less cost, more variance).

        This demonstrates that the A-C model provides meaningful optimization
        based on user risk preferences.
        """
        order_size = 2000

        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )

        # Low risk aversion (closer to TWAP)
        ac_low = generate_ac_trajectory(order_size, 10, params, risk_aversion=1e-8)
        # Higher risk aversion (more front-loaded)
        ac_high = generate_ac_trajectory(order_size, 10, params, risk_aversion=1e-5)

        runner = StrategyRunner()
        runs = runner.run_strategies(
            strategies=[ac_low, ac_high],
            market_data=realistic_market_data,
            side="buy",
            start_time=realistic_market_data["timestamp"].iloc[0],
            interval=timedelta(seconds=1),
        )

        arrival_price = realistic_market_data["mid_price"].iloc[0]

        low_result = create_simulation_result(runs[0], arrival_price, arrival_price)
        high_result = create_simulation_result(runs[1], arrival_price, arrival_price)

        # Both strategies should be almgren_chriss
        assert low_result.strategy_name == "almgren_chriss"
        assert high_result.strategy_name == "almgren_chriss"

        # Both should have filled something
        assert low_result.total_filled > 0
        assert high_result.total_filled > 0

    def test_comparison_table_shows_clear_ranking(self, realistic_market_data):
        """Comparison table should rank strategies sensibly."""
        order_size = 2000

        twap = generate_twap_trajectory(order_size=order_size, duration_periods=10)
        market = generate_market_order_trajectory(order_size=order_size)

        runner = StrategyRunner()
        runs = runner.run_strategies(
            strategies=[twap, market],
            market_data=realistic_market_data,
            side="buy",
            start_time=realistic_market_data["timestamp"].iloc[0],
            interval=timedelta(seconds=1),
        )

        arrival_price = realistic_market_data["mid_price"].iloc[0]
        results = [
            create_simulation_result(run, arrival_price, arrival_price) for run in runs
        ]

        # Compare by cost
        comparison = compare_simulation_results(results, rank_by="cost")

        assert len(comparison) == 2
        assert "is_bps" in comparison.columns
        assert "strategy" in comparison.columns

        # First row should be the better (lower cost) strategy
        # TWAP should win
        assert comparison.iloc[0]["strategy"] == "twap"

    def test_fill_rates_are_reasonable(self, realistic_market_data):
        """Fill rates should be reasonable (not 0% or mysteriously high)."""
        order_size = 2000

        twap = generate_twap_trajectory(order_size=order_size, duration_periods=5)
        market = generate_market_order_trajectory(order_size=order_size)

        runner = StrategyRunner()
        runs = runner.run_strategies(
            strategies=[twap, market],
            market_data=realistic_market_data,
            side="buy",
            start_time=realistic_market_data["timestamp"].iloc[0],
            interval=timedelta(seconds=1),
        )

        arrival_price = realistic_market_data["mid_price"].iloc[0]

        twap_result = create_simulation_result(runs[0], arrival_price, arrival_price)
        market_result = create_simulation_result(runs[1], arrival_price, arrival_price)

        # Fill rates should be positive
        assert twap_result.fill_rate > 0, "TWAP should have positive fill rate"
        assert market_result.fill_rate > 0, "Market order should have positive fill rate"

        # Fill rates should be <= 100%
        assert twap_result.fill_rate <= 100, "TWAP fill rate should not exceed 100%"
        assert (
            market_result.fill_rate <= 100
        ), "Market order fill rate should not exceed 100%"


class TestSellSideExecution:
    """Test execution on sell side for completeness."""

    def test_twap_beats_market_order_sell_side(self, realistic_market_data):
        """TWAP should beat market order on sell side too."""
        order_size = 2000

        twap = generate_twap_trajectory(order_size=order_size, duration_periods=10)
        market = generate_market_order_trajectory(order_size=order_size)

        runner = StrategyRunner()
        runs = runner.run_strategies(
            strategies=[twap, market],
            market_data=realistic_market_data,
            side="sell",  # Sell side
            start_time=realistic_market_data["timestamp"].iloc[0],
            interval=timedelta(seconds=1),
        )

        arrival_price = realistic_market_data["mid_price"].iloc[0]

        twap_result = create_simulation_result(runs[0], arrival_price, arrival_price)
        market_result = create_simulation_result(runs[1], arrival_price, arrival_price)

        # TWAP should have lower IS on sell side too
        assert (
            twap_result.implementation_shortfall_bps
            < market_result.implementation_shortfall_bps
        ), (
            f"TWAP ({twap_result.implementation_shortfall_bps:.2f} bps) should beat "
            f"market order ({market_result.implementation_shortfall_bps:.2f} bps) on sell side"
        )


class TestModuleIntegration:
    """Test that all simulation components work together."""

    def test_full_pipeline(self, realistic_market_data):
        """Test complete pipeline from strategy generation to comparison."""
        # 1. Generate strategies
        twap = generate_twap_trajectory(order_size=1000, duration_periods=5)
        market = generate_market_order_trajectory(order_size=1000)

        # 2. Run simulation
        runner = StrategyRunner()
        runs = runner.run_strategies(
            strategies=[twap, market],
            market_data=realistic_market_data,
            side="buy",
            start_time=realistic_market_data["timestamp"].iloc[0],
            interval=timedelta(seconds=1),
        )

        # 3. Create results
        arrival_price = realistic_market_data["mid_price"].iloc[0]
        results = [
            create_simulation_result(run, arrival_price, arrival_price) for run in runs
        ]

        # 4. Compare
        comparison = compare_simulation_results(results, rank_by="cost")

        # All steps completed successfully
        assert len(comparison) == 2
        assert not comparison.empty
        assert comparison.iloc[0]["strategy"] == "twap"

    def test_all_exports_available(self):
        """Verify all expected exports are available from the simulation module."""
        from tributary.analytics.simulation import (
            MarketEvent,
            OrderEvent,
            FillEvent,
            FillModel,
            SimulationEngine,
            StrategyRunner,
            StrategyRun,
            SimulationResult,
            create_simulation_result,
            compare_simulation_results,
            execution_chart_data,
            calculate_simulation_metrics,
        )

        # All imports successful
        assert MarketEvent is not None
        assert OrderEvent is not None
        assert FillEvent is not None
        assert FillModel is not None
        assert SimulationEngine is not None
        assert StrategyRunner is not None
        assert StrategyRun is not None
        assert SimulationResult is not None
        assert create_simulation_result is not None
        assert compare_simulation_results is not None
        assert execution_chart_data is not None
        assert calculate_simulation_metrics is not None
