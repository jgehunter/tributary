"""Unit tests for baseline execution strategies module."""

import math
from unittest.mock import Mock, MagicMock
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tributary.analytics.optimization.strategies import (
    generate_twap_trajectory,
    generate_vwap_trajectory,
    generate_market_order_trajectory,
    get_volume_profile_from_db,
)
from tributary.analytics.optimization.almgren_chriss import ExecutionTrajectory


class TestGenerateTwapTrajectory:
    """Tests for generate_twap_trajectory function."""

    def test_twap_basic(self):
        """Test basic TWAP creates equal slices."""
        traj = generate_twap_trajectory(order_size=1000, duration_periods=10)
        assert len(traj.trade_sizes) == 10
        assert np.allclose(traj.trade_sizes, 100)

    def test_twap_holdings_start_at_order_size(self):
        """Test that holdings start at order_size."""
        order_size = 5000
        traj = generate_twap_trajectory(order_size=order_size, duration_periods=10)
        assert traj.holdings[0] == pytest.approx(order_size)

    def test_twap_holdings_end_at_zero(self):
        """Test that holdings end at approximately zero."""
        traj = generate_twap_trajectory(order_size=1000, duration_periods=10)
        assert traj.holdings[-1] == pytest.approx(0, abs=1e-10)

    def test_twap_trade_sizes_sum_to_order_size(self):
        """Test that trade sizes sum to order_size."""
        order_size = 5000
        traj = generate_twap_trajectory(order_size=order_size, duration_periods=10)
        assert np.sum(traj.trade_sizes) == pytest.approx(order_size)

    def test_twap_all_slices_equal(self):
        """Test that all slices are equal when not randomized."""
        traj = generate_twap_trajectory(order_size=1000, duration_periods=5)
        expected_size = 1000 / 5
        assert np.allclose(traj.trade_sizes, expected_size)

    def test_twap_randomized_slices_vary(self):
        """Test that randomized slices have variation."""
        traj = generate_twap_trajectory(
            order_size=1000, duration_periods=10, randomize=True, seed=42
        )
        # Not all slices should be equal
        unique_sizes = len(np.unique(np.round(traj.trade_sizes, 5)))
        assert unique_sizes > 1

    def test_twap_randomized_still_sums_to_total(self):
        """Test that randomized slices still sum to order_size."""
        order_size = 5000
        traj = generate_twap_trajectory(
            order_size=order_size, duration_periods=10, randomize=True, seed=123
        )
        assert np.sum(traj.trade_sizes) == pytest.approx(order_size)

    def test_twap_randomized_with_seed_reproducible(self):
        """Test that randomized slices are reproducible with same seed."""
        traj1 = generate_twap_trajectory(
            order_size=1000, duration_periods=10, randomize=True, seed=42
        )
        traj2 = generate_twap_trajectory(
            order_size=1000, duration_periods=10, randomize=True, seed=42
        )
        assert np.allclose(traj1.trade_sizes, traj2.trade_sizes)

    def test_twap_strategy_name_is_twap(self):
        """Test that strategy_name is 'twap'."""
        traj = generate_twap_trajectory(order_size=1000, duration_periods=10)
        assert traj.strategy_name == "twap"

    def test_twap_risk_aversion_is_zero(self):
        """Test that risk_aversion is 0 (risk-neutral)."""
        traj = generate_twap_trajectory(order_size=1000, duration_periods=10)
        assert traj.risk_aversion == 0.0

    def test_twap_invalid_order_size_zero(self):
        """Test that zero order_size returns NaN trajectory."""
        traj = generate_twap_trajectory(order_size=0, duration_periods=10)
        assert math.isnan(traj.holdings[0])
        assert "error" in traj.params

    def test_twap_invalid_order_size_negative(self):
        """Test that negative order_size returns NaN trajectory."""
        traj = generate_twap_trajectory(order_size=-1000, duration_periods=10)
        assert math.isnan(traj.holdings[0])

    def test_twap_invalid_duration_zero(self):
        """Test that zero duration returns NaN trajectory."""
        traj = generate_twap_trajectory(order_size=1000, duration_periods=0)
        assert math.isnan(traj.holdings[0])


class TestGenerateVwapTrajectory:
    """Tests for generate_vwap_trajectory function."""

    def test_vwap_basic(self):
        """Test basic VWAP creates weighted slices."""
        volume_profile = np.array([100, 200, 300, 200, 100])
        traj = generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)
        assert len(traj.trade_sizes) == 5

    def test_vwap_higher_volume_larger_slice(self):
        """Test that higher volume periods get larger slices."""
        volume_profile = np.array([100, 200, 300, 200, 100])
        traj = generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)
        # Middle slice (index 2) should be largest
        assert traj.trade_sizes[2] > traj.trade_sizes[0]
        assert traj.trade_sizes[2] > traj.trade_sizes[4]

    def test_vwap_holdings_start_at_order_size(self):
        """Test that holdings start at order_size."""
        order_size = 5000
        volume_profile = np.array([1, 2, 3, 2, 1])
        traj = generate_vwap_trajectory(order_size=order_size, volume_profile=volume_profile)
        assert traj.holdings[0] == pytest.approx(order_size)

    def test_vwap_holdings_end_at_zero(self):
        """Test that holdings end at approximately zero."""
        volume_profile = np.array([1, 2, 3, 2, 1])
        traj = generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)
        assert traj.holdings[-1] == pytest.approx(0, abs=1e-10)

    def test_vwap_trade_sizes_sum_to_order_size(self):
        """Test that trade sizes sum to order_size."""
        order_size = 5000
        volume_profile = np.array([100, 200, 150, 250, 300])
        traj = generate_vwap_trajectory(order_size=order_size, volume_profile=volume_profile)
        assert np.sum(traj.trade_sizes) == pytest.approx(order_size)

    def test_vwap_uniform_volume_equals_twap(self):
        """Test that uniform volume profile equals TWAP."""
        volume_profile = np.array([100, 100, 100, 100, 100])
        vwap_traj = generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)
        twap_traj = generate_twap_trajectory(order_size=1000, duration_periods=5)
        assert np.allclose(vwap_traj.trade_sizes, twap_traj.trade_sizes)

    def test_vwap_zero_total_volume_fallback_to_twap(self):
        """Test that zero total volume falls back to TWAP."""
        volume_profile = np.array([0, 0, 0, 0, 0])
        traj = generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)
        # Should produce equal slices (TWAP fallback)
        expected_size = 1000 / 5
        assert np.allclose(traj.trade_sizes, expected_size)
        assert traj.params["fallback_to_twap"] is True

    def test_vwap_strategy_name_is_vwap(self):
        """Test that strategy_name is 'vwap'."""
        volume_profile = np.array([1, 2, 3])
        traj = generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)
        assert traj.strategy_name == "vwap"

    def test_vwap_invalid_order_size(self):
        """Test that invalid order_size returns NaN trajectory."""
        volume_profile = np.array([1, 2, 3])
        traj = generate_vwap_trajectory(order_size=0, volume_profile=volume_profile)
        assert math.isnan(traj.holdings[0])

    def test_vwap_empty_volume_profile(self):
        """Test that empty volume_profile returns NaN trajectory."""
        traj = generate_vwap_trajectory(order_size=1000, volume_profile=np.array([]))
        assert math.isnan(traj.holdings[0])
        assert "error" in traj.params


class TestGenerateMarketOrderTrajectory:
    """Tests for generate_market_order_trajectory function."""

    def test_market_order_single_slice(self):
        """Test that market order has single slice."""
        traj = generate_market_order_trajectory(order_size=1000)
        assert len(traj.trade_sizes) == 1
        assert traj.trade_sizes[0] == 1000

    def test_market_order_holdings_start_at_order_size(self):
        """Test that holdings start at order_size."""
        order_size = 5000
        traj = generate_market_order_trajectory(order_size=order_size)
        assert traj.holdings[0] == pytest.approx(order_size)

    def test_market_order_holdings_end_at_zero(self):
        """Test that holdings end at zero."""
        traj = generate_market_order_trajectory(order_size=1000)
        assert traj.holdings[-1] == pytest.approx(0)

    def test_market_order_strategy_name(self):
        """Test that strategy_name is 'market_order'."""
        traj = generate_market_order_trajectory(order_size=1000)
        assert traj.strategy_name == "market_order"

    def test_market_order_risk_aversion_is_inf(self):
        """Test that risk_aversion is infinity."""
        traj = generate_market_order_trajectory(order_size=1000)
        assert traj.risk_aversion == float("inf")

    def test_market_order_invalid_order_size(self):
        """Test that invalid order_size returns NaN trajectory."""
        traj = generate_market_order_trajectory(order_size=0)
        assert math.isnan(traj.holdings[0])

        traj = generate_market_order_trajectory(order_size=-1000)
        assert math.isnan(traj.holdings[0])


class TestGetVolumeProfileFromDb:
    """Tests for get_volume_profile_from_db function."""

    def test_get_volume_profile_returns_array(self):
        """Test that function returns volume array from mock reader."""
        # Create mock reader
        mock_reader = Mock()
        mock_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="H"),
            "vwap": [0.50, 0.51, 0.52, 0.51, 0.50],
            "volume": [100, 200, 300, 200, 100],
            "trade_count": [10, 20, 30, 20, 10],
        })
        mock_reader.query_vwap_sampled.return_value = mock_df

        profile = get_volume_profile_from_db(
            reader=mock_reader,
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            interval="1h",
        )

        assert isinstance(profile, np.ndarray)
        assert len(profile) == 5
        assert profile[2] == 300  # Middle value

    def test_get_volume_profile_empty_result(self):
        """Test that empty DataFrame returns empty array."""
        mock_reader = Mock()
        mock_reader.query_vwap_sampled.return_value = pd.DataFrame()

        profile = get_volume_profile_from_db(
            reader=mock_reader,
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            interval="1h",
        )

        assert isinstance(profile, np.ndarray)
        assert len(profile) == 0


class TestIntegration:
    """Integration tests for strategies module."""

    def test_all_strategies_return_execution_trajectory(self):
        """Test that all strategies return ExecutionTrajectory."""
        twap = generate_twap_trajectory(order_size=1000, duration_periods=10)
        assert isinstance(twap, ExecutionTrajectory)

        vwap = generate_vwap_trajectory(
            order_size=1000, volume_profile=np.array([1, 2, 3])
        )
        assert isinstance(vwap, ExecutionTrajectory)

        market = generate_market_order_trajectory(order_size=1000)
        assert isinstance(market, ExecutionTrajectory)

    def test_imports_from_analytics_package(self):
        """Test that exports work from tributary.analytics package."""
        from tributary.analytics import (
            generate_twap_trajectory,
            generate_vwap_trajectory,
            generate_market_order_trajectory,
        )

        twap = generate_twap_trajectory(order_size=1000, duration_periods=5)
        assert twap.strategy_name == "twap"

        vwap = generate_vwap_trajectory(
            order_size=1000, volume_profile=np.array([1, 2, 3])
        )
        assert vwap.strategy_name == "vwap"

        market = generate_market_order_trajectory(order_size=1000)
        assert market.strategy_name == "market_order"

    def test_twap_vwap_market_order_comparable_format(self):
        """Test that all strategies have comparable format."""
        twap = generate_twap_trajectory(order_size=1000, duration_periods=10)
        vwap = generate_vwap_trajectory(
            order_size=1000, volume_profile=np.array([1] * 10)
        )
        market = generate_market_order_trajectory(order_size=1000)

        # All should have consistent holdings format
        assert twap.holdings[0] == 1000
        assert vwap.holdings[0] == 1000
        assert market.holdings[0] == 1000

        # All should end at zero
        assert twap.holdings[-1] == pytest.approx(0, abs=1e-10)
        assert vwap.holdings[-1] == pytest.approx(0, abs=1e-10)
        assert market.holdings[-1] == pytest.approx(0, abs=1e-10)

        # Trade sizes should sum to order size
        assert np.sum(twap.trade_sizes) == pytest.approx(1000)
        assert np.sum(vwap.trade_sizes) == pytest.approx(1000)
        assert np.sum(market.trade_sizes) == pytest.approx(1000)
