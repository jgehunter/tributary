"""Unit tests for slippage calculation."""

import math

import pytest

from tributary.analytics.slippage import calculate_slippage_bps


# =============================================================================
# calculate_slippage_bps Tests
# =============================================================================


class TestCalculateSlippageBps:
    """Tests for calculate_slippage_bps function."""

    def test_buy_slippage_positive_paid_more(self):
        """Buy order: paying more than benchmark is a cost (positive slippage)."""
        # Paid 102 when benchmark was 100 -> 2% cost = 200 bps
        slippage = calculate_slippage_bps(
            execution_price=102,
            benchmark_price=100,
            side="buy",
        )

        assert slippage == 200.0

    def test_buy_slippage_negative_paid_less(self):
        """Buy order: paying less than benchmark is favorable (negative slippage)."""
        # Paid 98 when benchmark was 100 -> 2% gain = -200 bps
        slippage = calculate_slippage_bps(
            execution_price=98,
            benchmark_price=100,
            side="buy",
        )

        assert slippage == -200.0

    def test_sell_slippage_positive_received_less(self):
        """Sell order: receiving less than benchmark is a cost (positive slippage)."""
        # Received 98 when benchmark was 100 -> 2% cost = 200 bps
        slippage = calculate_slippage_bps(
            execution_price=98,
            benchmark_price=100,
            side="sell",
        )

        assert slippage == 200.0

    def test_sell_slippage_negative_received_more(self):
        """Sell order: receiving more than benchmark is favorable (negative slippage)."""
        # Received 102 when benchmark was 100 -> 2% gain = -200 bps
        slippage = calculate_slippage_bps(
            execution_price=102,
            benchmark_price=100,
            side="sell",
        )

        assert slippage == -200.0

    def test_zero_benchmark_price_returns_nan(self):
        """Zero benchmark price should return NaN (undefined slippage)."""
        slippage = calculate_slippage_bps(
            execution_price=100,
            benchmark_price=0,
            side="buy",
        )

        assert math.isnan(slippage)

    def test_equal_prices_returns_zero(self):
        """Equal execution and benchmark prices return 0.0 slippage."""
        slippage_buy = calculate_slippage_bps(
            execution_price=100,
            benchmark_price=100,
            side="buy",
        )
        slippage_sell = calculate_slippage_bps(
            execution_price=100,
            benchmark_price=100,
            side="sell",
        )

        assert slippage_buy == 0.0
        assert slippage_sell == 0.0

    def test_large_slippage_values(self):
        """Large slippage (1000+ bps) should calculate correctly."""
        # Paid 120 when benchmark was 100 -> 20% cost = 2000 bps
        slippage = calculate_slippage_bps(
            execution_price=120,
            benchmark_price=100,
            side="buy",
        )

        assert slippage == 2000.0

    def test_invalid_side_raises_value_error(self):
        """Invalid side parameter should raise ValueError."""
        with pytest.raises(ValueError, match="side must be 'buy' or 'sell'"):
            calculate_slippage_bps(
                execution_price=100,
                benchmark_price=100,
                side="invalid",
            )

    def test_small_slippage_precision(self):
        """Small slippage values should maintain precision."""
        # Paid 100.01 when benchmark was 100 -> 0.01% = 1 bp
        slippage = calculate_slippage_bps(
            execution_price=100.01,
            benchmark_price=100,
            side="buy",
        )

        assert abs(slippage - 1.0) < 0.0001

    def test_fractional_prices(self):
        """Slippage should work with fractional prices (prediction market style)."""
        # Paid 0.52 when benchmark was 0.50 -> 4% cost = 400 bps
        slippage = calculate_slippage_bps(
            execution_price=0.52,
            benchmark_price=0.50,
            side="buy",
        )

        assert abs(slippage - 400.0) < 0.0001

    def test_sell_symmetry_with_buy(self):
        """Sell and buy should be symmetric: same deviation = same cost magnitude."""
        # Buy paid 105 vs 100 -> +5% = 500 bps cost
        buy_slippage = calculate_slippage_bps(105, 100, "buy")

        # Sell received 95 vs 100 -> -5% received = 500 bps cost
        sell_slippage = calculate_slippage_bps(95, 100, "sell")

        assert buy_slippage == sell_slippage == 500.0
