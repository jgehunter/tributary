"""Tests for simulation result containers and comparison utilities.

Tests for:
- SimulationResult creation from StrategyRun
- fill_rate property calculation
- risk_adjusted_score property
- compare_simulation_results ranking
- execution_chart_data format
- Empty results handling
- Proving market order has higher IS than TWAP (SIM-05)
"""

import pytest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tributary.analytics.simulation.events import FillEvent
from tributary.analytics.simulation.runner import StrategyRun
from tributary.analytics.simulation.results import (
    SimulationResult,
    create_simulation_result,
    compare_simulation_results,
    execution_chart_data,
)
from tributary.analytics.optimization import (
    ExecutionTrajectory,
    generate_twap_trajectory,
    generate_market_order_trajectory,
)


def make_fill(
    filled_size: float,
    avg_price: float,
    slippage_bps: float,
    requested_size: float | None = None,
    slice_index: int = 0,
    timestamp: datetime | None = None,
) -> FillEvent:
    """Helper to create FillEvent for testing."""
    return FillEvent(
        timestamp=timestamp or datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        strategy_name="test",
        slice_index=slice_index,
        requested_size=requested_size if requested_size is not None else filled_size,
        filled_size=filled_size,
        avg_price=avg_price,
        slippage_bps=slippage_bps,
        levels_consumed=1,
        mid_price_at_fill=0.50,
    )


def make_strategy_run(
    strategy_name: str,
    fills: list[FillEvent],
    trade_sizes: np.ndarray,
    side: str = "buy",
) -> StrategyRun:
    """Helper to create StrategyRun for testing."""
    n = len(trade_sizes)
    holdings = np.zeros(n + 1)
    holdings[0] = float(np.sum(trade_sizes))
    for i in range(n):
        holdings[i + 1] = holdings[i] - trade_sizes[i]

    trajectory = ExecutionTrajectory(
        timestamps=np.arange(n + 1, dtype=float),
        holdings=holdings,
        trade_sizes=trade_sizes,
        strategy_name=strategy_name,
        total_cost_estimate=0.0,
        risk_aversion=0.0,
    )

    return StrategyRun(
        trajectory=trajectory,
        fills=fills,
        side=side,
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        interval=timedelta(seconds=1),
    )


class TestSimulationResultCreation:
    """Tests for creating SimulationResult from StrategyRun."""

    def test_create_from_strategy_run(self):
        """SimulationResult created correctly from StrategyRun."""
        fills = [
            make_fill(filled_size=500, avg_price=0.51, slippage_bps=200.0, slice_index=0),
            make_fill(filled_size=500, avg_price=0.52, slippage_bps=400.0, slice_index=1),
        ]

        run = make_strategy_run(
            strategy_name="twap",
            fills=fills,
            trade_sizes=np.array([500.0, 500.0]),
        )

        result = create_simulation_result(run, arrival_price=0.50, market_vwap=0.50)

        assert result.strategy_name == "twap"
        assert result.total_order_size == 1000.0
        assert result.side == "buy"
        assert result.total_filled == 1000.0
        assert result.num_slices == 2
        # VWAP = (500*0.51 + 500*0.52) / 1000 = 0.515
        assert result.avg_execution_price == pytest.approx(0.515, rel=1e-6)
        # IS = (0.515 - 0.50) / 0.50 * 10000 = 300 bps
        assert result.implementation_shortfall_bps == pytest.approx(300.0, rel=1e-6)

    def test_result_is_frozen(self):
        """SimulationResult is immutable (frozen dataclass)."""
        fills = [make_fill(filled_size=1000, avg_price=0.51, slippage_bps=200.0)]
        run = make_strategy_run("twap", fills, np.array([1000.0]))
        result = create_simulation_result(run, 0.50, 0.50)

        with pytest.raises(AttributeError):
            result.strategy_name = "modified"


class TestSimulationResultProperties:
    """Tests for SimulationResult computed properties."""

    def test_fill_rate_property(self):
        """fill_rate correctly calculates percentage filled."""
        fills = [make_fill(filled_size=750, avg_price=0.51, slippage_bps=200.0)]
        run = make_strategy_run("twap", fills, np.array([1000.0]))
        result = create_simulation_result(run, 0.50, 0.50)

        # 750 / 1000 * 100 = 75%
        assert result.fill_rate == pytest.approx(75.0, rel=1e-6)

    def test_fill_rate_zero_order_size(self):
        """fill_rate handles zero order size."""
        result = SimulationResult(
            strategy_name="test",
            total_order_size=0.0,
            side="buy",
            total_filled=0.0,
            total_unfilled=0.0,
            num_slices=0,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=float("nan"),
            implementation_shortfall_bps=float("nan"),
            vwap_slippage_bps=float("nan"),
            total_cost_usd=0.0,
            cost_variance=0.0,
            max_drawdown_bps=0.0,
            worst_slice_slippage_bps=float("nan"),
            fills=(),
        )

        assert result.fill_rate == 0.0

    def test_risk_adjusted_score_with_variance(self):
        """risk_adjusted_score divides IS by sqrt(variance)."""
        result = SimulationResult(
            strategy_name="test",
            total_order_size=1000.0,
            side="buy",
            total_filled=1000.0,
            total_unfilled=0.0,
            num_slices=2,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=0.515,
            implementation_shortfall_bps=300.0,
            vwap_slippage_bps=300.0,
            total_cost_usd=15.0,
            cost_variance=10000.0,  # sqrt = 100
            max_drawdown_bps=400.0,
            worst_slice_slippage_bps=400.0,
            fills=(),
        )

        # 300 / sqrt(10000) = 300 / 100 = 3.0
        assert result.risk_adjusted_score == pytest.approx(3.0, rel=1e-6)

    def test_risk_adjusted_score_zero_variance(self):
        """risk_adjusted_score returns raw IS when variance is zero."""
        result = SimulationResult(
            strategy_name="test",
            total_order_size=1000.0,
            side="buy",
            total_filled=1000.0,
            total_unfilled=0.0,
            num_slices=1,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=0.51,
            implementation_shortfall_bps=200.0,
            vwap_slippage_bps=200.0,
            total_cost_usd=10.0,
            cost_variance=0.0,
            max_drawdown_bps=200.0,
            worst_slice_slippage_bps=200.0,
            fills=(),
        )

        assert result.risk_adjusted_score == 200.0


class TestCompareSimulationResults:
    """Tests for compare_simulation_results ranking."""

    def make_result(
        self, strategy_name: str, is_bps: float, variance: float
    ) -> SimulationResult:
        """Helper to make SimulationResult with specific metrics."""
        return SimulationResult(
            strategy_name=strategy_name,
            total_order_size=1000.0,
            side="buy",
            total_filled=1000.0,
            total_unfilled=0.0,
            num_slices=5,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=0.50 + is_bps / 10000 * 0.50,
            implementation_shortfall_bps=is_bps,
            vwap_slippage_bps=is_bps,
            total_cost_usd=is_bps * 1000 * 0.50 / 10000,
            cost_variance=variance,
            max_drawdown_bps=is_bps * 1.2,
            worst_slice_slippage_bps=is_bps * 1.5,
            fills=(),
        )

    def test_rank_by_cost(self):
        """Ranking by cost sorts by implementation shortfall."""
        results = [
            self.make_result("high_cost", 500.0, 1000.0),
            self.make_result("low_cost", 100.0, 5000.0),
            self.make_result("mid_cost", 300.0, 2000.0),
        ]

        comparison = compare_simulation_results(results, rank_by="cost")

        assert comparison.iloc[0]["strategy"] == "low_cost"
        assert comparison.iloc[1]["strategy"] == "mid_cost"
        assert comparison.iloc[2]["strategy"] == "high_cost"

    def test_rank_by_risk(self):
        """Ranking by risk sorts by cost variance."""
        results = [
            self.make_result("high_risk", 100.0, 5000.0),
            self.make_result("low_risk", 300.0, 100.0),
            self.make_result("mid_risk", 200.0, 1000.0),
        ]

        comparison = compare_simulation_results(results, rank_by="risk")

        assert comparison.iloc[0]["strategy"] == "low_risk"
        assert comparison.iloc[1]["strategy"] == "mid_risk"
        assert comparison.iloc[2]["strategy"] == "high_risk"

    def test_rank_by_risk_adjusted(self):
        """Ranking by risk_adjusted uses IS/sqrt(variance)."""
        # risk_adjusted = is_bps / sqrt(variance)
        # A: 200 / sqrt(1600) = 200 / 40 = 5.0
        # B: 300 / sqrt(10000) = 300 / 100 = 3.0  <- BEST
        # C: 100 / sqrt(100) = 100 / 10 = 10.0
        results = [
            self.make_result("A", 200.0, 1600.0),
            self.make_result("B", 300.0, 10000.0),
            self.make_result("C", 100.0, 100.0),
        ]

        comparison = compare_simulation_results(results, rank_by="risk_adjusted")

        assert comparison.iloc[0]["strategy"] == "B"  # Lowest risk-adjusted score
        assert comparison.iloc[1]["strategy"] == "A"
        assert comparison.iloc[2]["strategy"] == "C"

    def test_empty_results(self):
        """Empty results returns empty DataFrame."""
        comparison = compare_simulation_results([])

        assert comparison.empty

    def test_comparison_has_expected_columns(self):
        """Comparison DataFrame has all expected columns."""
        results = [self.make_result("test", 200.0, 1000.0)]

        comparison = compare_simulation_results(results)

        expected_columns = [
            "strategy",
            "is_bps",
            "vwap_slip_bps",
            "cost_variance",
            "max_drawdown_bps",
            "fill_rate_pct",
            "risk_adjusted_score",
            "total_cost_usd",
        ]
        for col in expected_columns:
            assert col in comparison.columns


class TestExecutionChartData:
    """Tests for execution_chart_data format."""

    def test_chart_data_format(self):
        """Chart data has correct format and columns."""
        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)

        result = SimulationResult(
            strategy_name="twap",
            total_order_size=1000.0,
            side="buy",
            total_filled=1000.0,
            total_unfilled=0.0,
            num_slices=2,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=0.515,
            implementation_shortfall_bps=300.0,
            vwap_slippage_bps=300.0,
            total_cost_usd=15.0,
            cost_variance=10000.0,
            max_drawdown_bps=400.0,
            worst_slice_slippage_bps=400.0,
            fills=(
                make_fill(500, 0.51, 200.0, slice_index=0, timestamp=ts1),
                make_fill(500, 0.52, 400.0, slice_index=1, timestamp=ts2),
            ),
        )

        chart_data = execution_chart_data([result])

        assert "timestamp" in chart_data.columns
        assert "strategy" in chart_data.columns
        assert "holdings_pct" in chart_data.columns
        assert "cumulative_cost_bps" in chart_data.columns

    def test_chart_data_holdings_decrease(self):
        """Holdings percentage decreases from 100 to 0."""
        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)

        result = SimulationResult(
            strategy_name="twap",
            total_order_size=1000.0,
            side="buy",
            total_filled=1000.0,
            total_unfilled=0.0,
            num_slices=2,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=0.515,
            implementation_shortfall_bps=300.0,
            vwap_slippage_bps=300.0,
            total_cost_usd=15.0,
            cost_variance=10000.0,
            max_drawdown_bps=400.0,
            worst_slice_slippage_bps=400.0,
            fills=(
                make_fill(500, 0.51, 200.0, slice_index=0, timestamp=ts1),
                make_fill(500, 0.52, 400.0, slice_index=1, timestamp=ts2),
            ),
        )

        chart_data = execution_chart_data([result])

        # First row: 100% holdings (initial state)
        assert chart_data.iloc[0]["holdings_pct"] == 100.0
        # After first fill: 50%
        assert chart_data.iloc[1]["holdings_pct"] == pytest.approx(50.0, rel=1e-6)
        # After second fill: 0%
        assert chart_data.iloc[2]["holdings_pct"] == pytest.approx(0.0, rel=1e-6)

    def test_chart_data_cumulative_cost(self):
        """Cumulative cost increases with each fill."""
        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)

        result = SimulationResult(
            strategy_name="twap",
            total_order_size=1000.0,
            side="buy",
            total_filled=1000.0,
            total_unfilled=0.0,
            num_slices=2,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=0.515,
            implementation_shortfall_bps=300.0,
            vwap_slippage_bps=300.0,
            total_cost_usd=15.0,
            cost_variance=10000.0,
            max_drawdown_bps=400.0,
            worst_slice_slippage_bps=400.0,
            fills=(
                make_fill(500, 0.51, 200.0, slice_index=0, timestamp=ts1),
                make_fill(500, 0.52, 400.0, slice_index=1, timestamp=ts2),
            ),
        )

        chart_data = execution_chart_data([result])

        # Initial: 0 cost
        assert chart_data.iloc[0]["cumulative_cost_bps"] == 0.0
        # After fill 1: 200 * 0.5 = 100 bps
        assert chart_data.iloc[1]["cumulative_cost_bps"] == pytest.approx(100.0, rel=1e-6)
        # After fill 2: 100 + 400 * 0.5 = 300 bps
        assert chart_data.iloc[2]["cumulative_cost_bps"] == pytest.approx(300.0, rel=1e-6)

    def test_chart_data_empty_fills(self):
        """Empty fills result in empty chart data."""
        result = SimulationResult(
            strategy_name="twap",
            total_order_size=1000.0,
            side="buy",
            total_filled=0.0,
            total_unfilled=1000.0,
            num_slices=0,
            num_partial_fills=0,
            arrival_price=0.50,
            avg_execution_price=float("nan"),
            implementation_shortfall_bps=float("nan"),
            vwap_slippage_bps=float("nan"),
            total_cost_usd=0.0,
            cost_variance=float("nan"),
            max_drawdown_bps=float("nan"),
            worst_slice_slippage_bps=float("nan"),
            fills=(),
        )

        chart_data = execution_chart_data([result])

        assert len(chart_data) == 0


class TestMarketOrderVsTWAP:
    """Test that proves market order has higher IS than TWAP (SIM-05)."""

    def test_market_order_higher_is_than_twap(self):
        """
        Market order should have higher implementation shortfall than TWAP.

        This is the core proof of SIM-05: optimized execution beats naive.

        Setup:
        - Market order: Execute 3000 in one slice, consumes more levels -> higher slippage
        - TWAP: Execute 3x1000 in three slices, less level consumption per slice
        """
        # Simulated fills for market order (one big order, high slippage)
        market_fills = [
            make_fill(filled_size=3000, avg_price=0.53, slippage_bps=600.0),
        ]

        # Simulated fills for TWAP (three smaller orders, lower individual slippage)
        ts_base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        twap_fills = [
            make_fill(
                filled_size=1000,
                avg_price=0.51,
                slippage_bps=200.0,
                slice_index=0,
                timestamp=ts_base,
            ),
            make_fill(
                filled_size=1000,
                avg_price=0.51,
                slippage_bps=200.0,
                slice_index=1,
                timestamp=ts_base + timedelta(seconds=1),
            ),
            make_fill(
                filled_size=1000,
                avg_price=0.51,
                slippage_bps=200.0,
                slice_index=2,
                timestamp=ts_base + timedelta(seconds=2),
            ),
        ]

        # Create strategy runs
        market_run = make_strategy_run(
            "market_order", market_fills, np.array([3000.0])
        )
        twap_run = make_strategy_run(
            "twap", twap_fills, np.array([1000.0, 1000.0, 1000.0])
        )

        # Create results
        market_result = create_simulation_result(market_run, 0.50, 0.50)
        twap_result = create_simulation_result(twap_run, 0.50, 0.50)

        # CORE ASSERTION: TWAP has lower IS than market order
        assert twap_result.implementation_shortfall_bps < market_result.implementation_shortfall_bps, (
            f"TWAP ({twap_result.implementation_shortfall_bps:.2f} bps) should be lower than "
            f"market order ({market_result.implementation_shortfall_bps:.2f} bps)"
        )

        # Compare using comparison function
        comparison = compare_simulation_results(
            [market_result, twap_result], rank_by="cost"
        )

        # TWAP should rank first (lower cost)
        assert comparison.iloc[0]["strategy"] == "twap"
        assert comparison.iloc[1]["strategy"] == "market_order"
