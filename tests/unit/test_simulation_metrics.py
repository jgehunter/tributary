"""Tests for simulation metrics calculation.

Tests calculate_simulation_metrics for:
- Single and multiple fills
- Buy and sell side sign conventions
- Empty fills handling
- Partial fills counting
- Risk metrics (variance, max drawdown, worst slice)
- Total cost USD calculation
"""

import pytest
from datetime import datetime, timezone
import numpy as np

from tributary.analytics.simulation.events import FillEvent
from tributary.analytics.simulation.metrics import calculate_simulation_metrics


def make_fill(
    filled_size: float,
    avg_price: float,
    slippage_bps: float,
    requested_size: float | None = None,
    slice_index: int = 0,
) -> FillEvent:
    """Helper to create FillEvent for testing."""
    return FillEvent(
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        strategy_name="test",
        slice_index=slice_index,
        requested_size=requested_size if requested_size is not None else filled_size,
        filled_size=filled_size,
        avg_price=avg_price,
        slippage_bps=slippage_bps,
        levels_consumed=1,
        mid_price_at_fill=0.50,
    )


class TestMetricsBasic:
    """Basic metrics calculation tests."""

    def test_single_fill_buy_side(self):
        """Single fill on buy side calculates IS correctly."""
        fills = [make_fill(filled_size=1000, avg_price=0.51, slippage_bps=200.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.505,
        )

        # IS = (0.51 - 0.50) / 0.50 * 10000 = 200 bps
        assert metrics["implementation_shortfall_bps"] == pytest.approx(200.0, rel=1e-6)
        assert metrics["total_filled"] == 1000.0
        assert metrics["total_unfilled"] == 0.0
        assert metrics["num_slices"] == 1
        assert metrics["avg_execution_price"] == pytest.approx(0.51, rel=1e-6)

    def test_single_fill_sell_side(self):
        """Single fill on sell side calculates IS correctly (sign convention)."""
        fills = [make_fill(filled_size=1000, avg_price=0.49, slippage_bps=200.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="sell",
            market_vwap=0.50,
        )

        # Sell: IS = (arrival - exec) / arrival = (0.50 - 0.49) / 0.50 * 10000 = 200 bps
        assert metrics["implementation_shortfall_bps"] == pytest.approx(200.0, rel=1e-6)

    def test_multiple_fills_calculates_vwap(self):
        """Multiple fills calculates size-weighted average price."""
        fills = [
            make_fill(filled_size=500, avg_price=0.51, slippage_bps=200.0, slice_index=0),
            make_fill(filled_size=500, avg_price=0.52, slippage_bps=400.0, slice_index=1),
        ]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.50,
        )

        # VWAP = (500 * 0.51 + 500 * 0.52) / 1000 = 0.515
        assert metrics["avg_execution_price"] == pytest.approx(0.515, rel=1e-6)
        # IS = (0.515 - 0.50) / 0.50 * 10000 = 300 bps
        assert metrics["implementation_shortfall_bps"] == pytest.approx(300.0, rel=1e-6)

    def test_variance_with_multiple_fills(self):
        """Cost variance calculated from multiple fills."""
        fills = [
            make_fill(filled_size=500, avg_price=0.51, slippage_bps=200.0, slice_index=0),
            make_fill(filled_size=500, avg_price=0.52, slippage_bps=400.0, slice_index=1),
        ]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.50,
        )

        # Variance of [200, 400] = ((200-300)^2 + (400-300)^2) / 2 = 10000
        assert metrics["cost_variance"] == pytest.approx(10000.0, rel=1e-6)

    def test_single_fill_zero_variance(self):
        """Single fill has zero variance."""
        fills = [make_fill(filled_size=1000, avg_price=0.51, slippage_bps=200.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.50,
        )

        assert metrics["cost_variance"] == 0.0


class TestEmptyAndEdgeCases:
    """Edge cases and empty fill handling."""

    def test_empty_fills_returns_nan(self):
        """Empty fill list returns NaN for most metrics."""
        metrics = calculate_simulation_metrics(
            fills=[],
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.50,
        )

        assert np.isnan(metrics["implementation_shortfall_bps"])
        assert np.isnan(metrics["vwap_slippage_bps"])
        assert np.isnan(metrics["avg_execution_price"])
        assert np.isnan(metrics["cost_variance"])
        assert np.isnan(metrics["max_drawdown_bps"])
        assert np.isnan(metrics["worst_slice_slippage_bps"])
        assert metrics["total_filled"] == 0.0
        assert metrics["total_unfilled"] == 1000.0
        assert metrics["num_slices"] == 0
        assert metrics["total_cost_usd"] == 0.0

    def test_partial_fills_counted(self):
        """Partial fills are counted correctly."""
        fills = [
            make_fill(
                filled_size=800, avg_price=0.51, slippage_bps=200.0,
                requested_size=1000, slice_index=0
            ),
            make_fill(
                filled_size=500, avg_price=0.52, slippage_bps=400.0,
                requested_size=500, slice_index=1
            ),
        ]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1500.0,
            side="buy",
            market_vwap=0.50,
        )

        assert metrics["num_partial_fills"] == 1  # Only first fill was partial
        assert metrics["total_filled"] == 1300.0
        assert metrics["total_unfilled"] == 200.0


class TestRiskMetrics:
    """Tests for risk metrics calculation."""

    def test_max_drawdown_calculation(self):
        """Max drawdown tracks worst cumulative cost."""
        fills = [
            make_fill(filled_size=500, avg_price=0.51, slippage_bps=200.0, slice_index=0),
            make_fill(filled_size=500, avg_price=0.53, slippage_bps=600.0, slice_index=1),
        ]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.50,
        )

        # Weighted slippages: 200 * (500/1000) = 100, 600 * (500/1000) = 300
        # Cumulative: [100, 400]
        # Max drawdown = 400
        assert metrics["max_drawdown_bps"] == pytest.approx(400.0, rel=1e-6)

    def test_worst_slice_slippage(self):
        """Worst slice tracks highest individual slippage."""
        fills = [
            make_fill(filled_size=500, avg_price=0.51, slippage_bps=100.0, slice_index=0),
            make_fill(filled_size=500, avg_price=0.53, slippage_bps=500.0, slice_index=1),
            make_fill(filled_size=500, avg_price=0.52, slippage_bps=300.0, slice_index=2),
        ]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1500.0,
            side="buy",
            market_vwap=0.50,
        )

        assert metrics["worst_slice_slippage_bps"] == 500.0


class TestTotalCostUSD:
    """Tests for total cost in USD calculation."""

    def test_total_cost_usd_buy(self):
        """Total cost USD calculated correctly for buy."""
        fills = [make_fill(filled_size=1000, avg_price=0.51, slippage_bps=200.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.50,
        )

        # Cost = 1000 * (0.51 - 0.50) = 10.0
        assert metrics["total_cost_usd"] == pytest.approx(10.0, rel=1e-6)

    def test_total_cost_usd_sell(self):
        """Total cost USD calculated correctly for sell."""
        fills = [make_fill(filled_size=1000, avg_price=0.49, slippage_bps=200.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="sell",
            market_vwap=0.50,
        )

        # Cost = 1000 * (0.50 - 0.49) = 10.0 (positive = cost)
        assert metrics["total_cost_usd"] == pytest.approx(10.0, rel=1e-6)

    def test_favorable_execution_negative_cost(self):
        """Favorable execution shows negative cost (gain)."""
        # Buy at lower than arrival = favorable
        fills = [make_fill(filled_size=1000, avg_price=0.49, slippage_bps=-200.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.50,
        )

        # IS = (0.49 - 0.50) / 0.50 * 10000 = -200 bps (favorable)
        assert metrics["implementation_shortfall_bps"] == pytest.approx(-200.0, rel=1e-6)
        # Cost = 1000 * (0.49 - 0.50) = -10.0 (gain)
        assert metrics["total_cost_usd"] == pytest.approx(-10.0, rel=1e-6)


class TestVWAPSlippage:
    """Tests for VWAP slippage calculation."""

    def test_vwap_slippage_buy(self):
        """VWAP slippage calculated vs market VWAP for buy."""
        fills = [make_fill(filled_size=1000, avg_price=0.52, slippage_bps=400.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="buy",
            market_vwap=0.51,  # Market VWAP lower than execution
        )

        # VWAP slippage = (0.52 - 0.51) / 0.51 * 10000 = 196.08 bps
        assert metrics["vwap_slippage_bps"] == pytest.approx(196.08, rel=1e-2)

    def test_vwap_slippage_sell(self):
        """VWAP slippage calculated vs market VWAP for sell."""
        fills = [make_fill(filled_size=1000, avg_price=0.48, slippage_bps=400.0)]

        metrics = calculate_simulation_metrics(
            fills=fills,
            arrival_price=0.50,
            total_order_size=1000.0,
            side="sell",
            market_vwap=0.49,  # Market VWAP higher than execution
        )

        # VWAP slippage = (0.49 - 0.48) / 0.49 * 10000 = 204.08 bps
        assert metrics["vwap_slippage_bps"] == pytest.approx(204.08, rel=1e-2)
