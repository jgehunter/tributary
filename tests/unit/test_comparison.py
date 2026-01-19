"""Unit tests for strategy comparison utilities.

Tests the StrategyComparison dataclass and compare_strategies() function
that enables side-by-side comparison of execution strategies before simulation.
"""

import numpy as np
import pandas as pd
import pytest

from tributary.analytics.optimization.comparison import (
    StrategyComparison,
    compare_strategies,
    execution_profile_chart,
)
from tributary.analytics.optimization.almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    calibrate_ac_params,
    generate_ac_trajectory,
)
from tributary.analytics.optimization.strategies import (
    generate_twap_trajectory,
    generate_vwap_trajectory,
    generate_market_order_trajectory,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_params():
    """Create sample A-C params for tests."""
    return calibrate_ac_params(
        daily_volume=50000,
        daily_spread=0.02,
        daily_volatility=0.05,
        price=0.50,
    )


@pytest.fixture
def twap_trajectory():
    """Create sample TWAP trajectory."""
    return generate_twap_trajectory(order_size=1000, duration_periods=10)


@pytest.fixture
def vwap_trajectory():
    """Create sample VWAP trajectory."""
    volume_profile = np.array([1, 2, 3, 2, 1, 1, 2, 3, 2, 1])
    return generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)


@pytest.fixture
def market_order_trajectory():
    """Create sample market order trajectory."""
    return generate_market_order_trajectory(order_size=1000)


@pytest.fixture
def ac_trajectory(sample_params):
    """Create sample Almgren-Chriss trajectory."""
    return generate_ac_trajectory(
        order_size=1000,
        duration_periods=10,
        params=sample_params,
        risk_aversion=1e-5,
    )


# =============================================================================
# Test StrategyComparison
# =============================================================================


class TestStrategyComparison:
    """Tests for StrategyComparison dataclass."""

    def test_comparison_creation(self, twap_trajectory, vwap_trajectory):
        """Test StrategyComparison can be created."""
        comparison = StrategyComparison(
            strategies=[twap_trajectory, vwap_trajectory],
            baseline_name="twap",
            order_size=1000,
        )

        assert len(comparison.strategies) == 2
        assert comparison.baseline_name == "twap"
        assert comparison.order_size == 1000

    def test_comparison_summary_table_columns(self, twap_trajectory, vwap_trajectory):
        """Test summary_table returns correct columns."""
        comparison = StrategyComparison(
            strategies=[twap_trajectory, vwap_trajectory],
            baseline_name="twap",
        )

        df = comparison.summary_table()

        expected_columns = [
            "strategy",
            "risk_aversion",
            "expected_cost_bps",
            "num_slices",
            "max_slice_pct",
            "min_slice_pct",
            "front_loaded",
        ]
        assert list(df.columns) == expected_columns

    def test_comparison_summary_table_values(self, twap_trajectory, vwap_trajectory):
        """Test summary_table returns correct values."""
        comparison = StrategyComparison(
            strategies=[twap_trajectory, vwap_trajectory],
        )

        df = comparison.summary_table()

        # TWAP should have 10 slices, each 10% (100 shares of 1000)
        twap_row = df[df["strategy"] == "twap"].iloc[0]
        assert twap_row["num_slices"] == 10
        assert abs(twap_row["max_slice_pct"] - 10.0) < 0.01
        assert abs(twap_row["min_slice_pct"] - 10.0) < 0.01
        assert twap_row["front_loaded"] == False

    def test_comparison_with_different_strategies(
        self, twap_trajectory, vwap_trajectory, ac_trajectory, market_order_trajectory
    ):
        """Test comparison with multiple strategy types."""
        comparison = StrategyComparison(
            strategies=[twap_trajectory, vwap_trajectory, ac_trajectory, market_order_trajectory],
        )

        df = comparison.summary_table()

        assert len(df) == 4
        assert set(df["strategy"].tolist()) == {"twap", "vwap", "almgren_chriss", "market_order"}

        # Market order should have 1 slice
        mkt_row = df[df["strategy"] == "market_order"].iloc[0]
        assert mkt_row["num_slices"] == 1
        assert abs(mkt_row["max_slice_pct"] - 100.0) < 0.01

    def test_comparison_front_loaded_detection(self, sample_params):
        """Test front_loaded flag is set correctly."""
        # Create risk-averse trajectory (front-loaded)
        ac_risk_averse = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=sample_params,
            risk_aversion=1e-3,  # High risk aversion = front-loaded
        )

        # Create TWAP (uniform)
        twap = generate_twap_trajectory(order_size=1000, duration_periods=10)

        comparison = StrategyComparison(
            strategies=[twap, ac_risk_averse],
        )

        df = comparison.summary_table()

        twap_row = df[df["strategy"] == "twap"].iloc[0]
        ac_row = df[df["strategy"] == "almgren_chriss"].iloc[0]

        # TWAP should not be front-loaded (equal slices)
        assert twap_row["front_loaded"] == False

        # Risk-averse A-C should be front-loaded
        assert ac_row["front_loaded"] == True


# =============================================================================
# Test compare_strategies
# =============================================================================


class TestCompareStrategies:
    """Tests for compare_strategies function."""

    def test_compare_two_strategies(self, twap_trajectory, vwap_trajectory):
        """Test comparing two strategies."""
        comparison = compare_strategies(twap_trajectory, vwap_trajectory)

        assert len(comparison.strategies) == 2
        assert comparison.baseline_name == "twap"  # Default

    def test_compare_multiple_strategies(
        self, twap_trajectory, vwap_trajectory, ac_trajectory, market_order_trajectory
    ):
        """Test comparing multiple strategies."""
        comparison = compare_strategies(
            twap_trajectory,
            vwap_trajectory,
            ac_trajectory,
            market_order_trajectory,
        )

        assert len(comparison.strategies) == 4

    def test_compare_validates_minimum_strategies(self, twap_trajectory):
        """Test that at least 2 strategies are required."""
        with pytest.raises(ValueError, match="At least 2 strategies"):
            compare_strategies(twap_trajectory)

        with pytest.raises(ValueError, match="At least 2 strategies"):
            compare_strategies()

    def test_compare_sets_order_size(self, twap_trajectory, vwap_trajectory):
        """Test that order_size is extracted from trajectories."""
        comparison = compare_strategies(twap_trajectory, vwap_trajectory)

        assert comparison.order_size == 1000

    def test_compare_warns_mismatched_order_sizes(self, twap_trajectory):
        """Test warning when order sizes don't match."""
        # Create trajectory with different order size
        different_size = generate_twap_trajectory(order_size=500, duration_periods=10)

        with pytest.warns(UserWarning, match="order size"):
            compare_strategies(twap_trajectory, different_size)

    def test_compare_custom_baseline(self, twap_trajectory, vwap_trajectory):
        """Test setting custom baseline name."""
        comparison = compare_strategies(
            twap_trajectory,
            vwap_trajectory,
            baseline="vwap",
        )

        assert comparison.baseline_name == "vwap"


# =============================================================================
# Test execution_profile_chart
# =============================================================================


class TestExecutionProfileChart:
    """Tests for execution_profile_chart function."""

    def test_profile_chart_columns(self, twap_trajectory, vwap_trajectory):
        """Test chart data has correct columns."""
        comparison = compare_strategies(twap_trajectory, vwap_trajectory)
        df = execution_profile_chart(comparison)

        expected_columns = ["period", "strategy", "holdings_pct", "trade_size_pct"]
        assert list(df.columns) == expected_columns

    def test_profile_chart_all_strategies_present(self, twap_trajectory, vwap_trajectory):
        """Test all strategies are included in chart data."""
        comparison = compare_strategies(twap_trajectory, vwap_trajectory)
        df = execution_profile_chart(comparison)

        strategies = df["strategy"].unique()
        assert set(strategies) == {"twap", "vwap"}

    def test_profile_chart_holdings_percentages(self, twap_trajectory, vwap_trajectory):
        """Test holdings are correct percentages."""
        comparison = compare_strategies(twap_trajectory, vwap_trajectory)
        df = execution_profile_chart(comparison)

        # Check TWAP holdings
        twap_data = df[df["strategy"] == "twap"].sort_values("period")

        # First period should have 100% holdings
        assert abs(twap_data.iloc[0]["holdings_pct"] - 100.0) < 0.01

        # Last period should have 0% holdings
        assert abs(twap_data.iloc[-1]["holdings_pct"]) < 0.01

    def test_profile_chart_periods_correct(self, twap_trajectory, vwap_trajectory):
        """Test periods are correct."""
        comparison = compare_strategies(twap_trajectory, vwap_trajectory)
        df = execution_profile_chart(comparison)

        # TWAP with 10 periods should have periods 0-10
        twap_data = df[df["strategy"] == "twap"]
        periods = sorted(twap_data["period"].unique())
        assert periods == list(range(11))  # 0 through 10

    def test_profile_chart_trade_sizes_sum_to_100(self, twap_trajectory, vwap_trajectory):
        """Test trade sizes sum to approximately 100%."""
        comparison = compare_strategies(twap_trajectory, vwap_trajectory)
        df = execution_profile_chart(comparison)

        for strategy in df["strategy"].unique():
            strategy_data = df[df["strategy"] == strategy]
            total_traded = strategy_data["trade_size_pct"].sum()
            # Should be approximately 100% (might have small floating point differences)
            assert abs(total_traded - 100.0) < 1.0


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for comparison module."""

    def test_full_comparison_workflow(self, sample_params):
        """Test complete comparison workflow."""
        # Generate strategies
        twap = generate_twap_trajectory(order_size=5000, duration_periods=10)
        vwap = generate_vwap_trajectory(
            order_size=5000,
            volume_profile=np.array([1, 2, 3, 2, 1, 1, 2, 3, 2, 1]),
        )
        ac = generate_ac_trajectory(
            order_size=5000,
            duration_periods=10,
            params=sample_params,
            risk_aversion=1e-5,
        )
        mkt = generate_market_order_trajectory(order_size=5000)

        # Create comparison
        comparison = compare_strategies(twap, vwap, ac, mkt)

        # Get summary table
        summary = comparison.summary_table()
        assert len(summary) == 4

        # Get chart data
        chart_data = execution_profile_chart(comparison)
        assert len(chart_data) > 0

        # Verify all strategies present
        strategies_in_summary = set(summary["strategy"].tolist())
        strategies_in_chart = set(chart_data["strategy"].unique())
        assert strategies_in_summary == strategies_in_chart

    def test_imports_from_analytics_package(self):
        """Test all exports available from tributary.analytics."""
        from tributary.analytics import (
            compare_strategies,
            StrategyComparison,
            execution_profile_chart,
            optimize_schedule,
            TradeSchedule,
            ScheduleConstraints,
            calculate_optimal_intervals,
        )

        # Verify they are the correct types
        assert callable(compare_strategies)
        assert callable(execution_profile_chart)
        assert callable(optimize_schedule)
        assert callable(calculate_optimal_intervals)

    def test_imports_from_optimization_package(self):
        """Test all exports available from tributary.analytics.optimization."""
        from tributary.analytics.optimization import (
            # Almgren-Chriss
            AlmgrenChrissParams,
            ExecutionTrajectory,
            calibrate_ac_params,
            generate_ac_trajectory,
            # Baseline strategies
            generate_twap_trajectory,
            generate_vwap_trajectory,
            generate_market_order_trajectory,
            get_volume_profile_from_db,
            # Scheduler
            ScheduleConstraints,
            TradeSchedule,
            optimize_schedule,
            calculate_optimal_intervals,
            # Comparison
            StrategyComparison,
            compare_strategies,
            execution_profile_chart,
        )

        # All imports succeeded
        assert True

    def test_smoke_test_from_plan_verification(self, sample_params):
        """Test the smoke test from the plan verification section."""
        from tributary.analytics import (
            calibrate_ac_params,
            generate_ac_trajectory,
            generate_twap_trajectory,
            generate_vwap_trajectory,
            generate_market_order_trajectory,
            compare_strategies,
        )

        # Calibrate
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )

        # Generate strategies
        order_size = 5000
        twap = generate_twap_trajectory(order_size, duration_periods=10)
        vwap = generate_vwap_trajectory(
            order_size, volume_profile=np.array([1, 2, 3, 2, 1, 1, 2, 3, 2, 1])
        )
        ac = generate_ac_trajectory(
            order_size, duration_periods=10, params=params, risk_aversion=1e-5
        )
        mkt = generate_market_order_trajectory(order_size)

        # Compare
        comparison = compare_strategies(twap, vwap, ac, mkt)
        summary = comparison.summary_table()

        # Verify output
        assert len(summary) == 4
        assert all(col in summary.columns for col in [
            "strategy", "risk_aversion", "expected_cost_bps",
            "num_slices", "max_slice_pct", "min_slice_pct", "front_loaded"
        ])
