"""Unit tests for benchmark calculations."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tributary.analytics.benchmarks import (
    calculate_vwap,
    calculate_cumulative_vwap,
    calculate_vwap_for_period,
    calculate_twap,
    calculate_twap_from_orderbooks,
    get_arrival_price,
)
from tributary.core.config import QuestDBConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_trades_df():
    """Create a simple trades DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 30),
            ],
            "price": [100.0, 200.0],
            "size": [10.0, 20.0],
        }
    )


@pytest.fixture
def multi_trades_df():
    """Create a trades DataFrame with multiple trades over time."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 10:00", periods=6, freq="30s"),
            "price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "size": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        }
    )


@pytest.fixture
def orderbook_df():
    """Create an orderbook snapshots DataFrame."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 10:00", periods=5, freq="30s"),
            "mid_price": [0.50, 0.51, 0.52, 0.51, 0.50],
        }
    )


@pytest.fixture
def mock_reader():
    """Create a mock QuestDBReader."""
    config = QuestDBConfig(
        host="localhost",
        pg_port=8812,
        pg_user="admin",
        pg_password="quest",
    )
    reader = MagicMock()
    reader.config = config
    return reader


# =============================================================================
# VWAP Tests
# =============================================================================


class TestCalculateVwap:
    """Tests for calculate_vwap function."""

    def test_calculate_vwap_simple(self, simple_trades_df):
        """VWAP with known values: [100, 200] with sizes [10, 20] -> 166.67."""
        # Expected: (100*10 + 200*20) / (10 + 20) = 5000 / 30 = 166.666...
        vwap = calculate_vwap(simple_trades_df)

        assert abs(vwap - 166.66666666666666) < 0.0001

    def test_calculate_vwap_single_trade(self):
        """Single trade should return the trade price."""
        df = pd.DataFrame({"price": [50.0], "size": [100.0]})

        vwap = calculate_vwap(df)

        assert vwap == 50.0

    def test_calculate_vwap_empty_df(self):
        """Empty DataFrame should return NaN."""
        df = pd.DataFrame({"price": [], "size": []})

        vwap = calculate_vwap(df)

        assert np.isnan(vwap)

    def test_calculate_vwap_zero_volume(self):
        """Zero total volume should return NaN."""
        df = pd.DataFrame({"price": [100.0, 200.0], "size": [0.0, 0.0]})

        vwap = calculate_vwap(df)

        assert np.isnan(vwap)

    def test_calculate_vwap_equal_weights(self):
        """Equal sizes should give simple average."""
        df = pd.DataFrame({"price": [100.0, 200.0, 300.0], "size": [10.0, 10.0, 10.0]})

        vwap = calculate_vwap(df)

        assert vwap == 200.0  # Simple average


class TestCalculateCumulativeVwap:
    """Tests for calculate_cumulative_vwap function."""

    def test_cumulative_vwap_ordering(self, simple_trades_df):
        """Cumulative VWAP should change correctly over time."""
        # Trade 1: price=100, size=10 -> VWAP = 100
        # Trade 2: price=200, size=20 -> VWAP = (100*10 + 200*20)/(10+20) = 166.67

        cumulative = calculate_cumulative_vwap(simple_trades_df)

        assert len(cumulative) == 2
        assert cumulative.iloc[0] == 100.0
        assert abs(cumulative.iloc[1] - 166.66666666666666) < 0.0001

    def test_cumulative_vwap_empty_df(self):
        """Empty DataFrame should return empty Series."""
        df = pd.DataFrame({"timestamp": [], "price": [], "size": []})

        cumulative = calculate_cumulative_vwap(df)

        assert isinstance(cumulative, pd.Series)
        assert len(cumulative) == 0

    def test_cumulative_vwap_sorts_by_timestamp(self):
        """Trades should be sorted by timestamp before calculation."""
        # Out-of-order timestamps
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 1, 0),  # Second trade
                    datetime(2024, 1, 1, 10, 0, 0),  # First trade
                ],
                "price": [200.0, 100.0],
                "size": [20.0, 10.0],
            }
        )

        cumulative = calculate_cumulative_vwap(df)

        # After sorting: first trade at 10:00 (price=100), second at 10:01 (price=200)
        assert cumulative.iloc[0] == 100.0  # First VWAP = 100
        assert abs(cumulative.iloc[1] - 166.66666666666666) < 0.0001


class TestCalculateVwapForPeriod:
    """Tests for calculate_vwap_for_period function."""

    def test_vwap_for_period_queries_reader(self, mock_reader, simple_trades_df):
        """Should query reader and calculate VWAP."""
        mock_reader.query_trades.return_value = simple_trades_df

        vwap = calculate_vwap_for_period(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
        )

        mock_reader.query_trades.assert_called_once_with(
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            token_id="token-yes",
        )
        assert abs(vwap - 166.66666666666666) < 0.0001

    def test_vwap_for_period_empty_result(self, mock_reader):
        """Should return NaN if no trades found."""
        mock_reader.query_trades.return_value = pd.DataFrame(
            {"price": [], "size": []}
        )

        vwap = calculate_vwap_for_period(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
        )

        assert np.isnan(vwap)


# =============================================================================
# TWAP Tests
# =============================================================================


class TestCalculateTwap:
    """Tests for calculate_twap function."""

    def test_calculate_twap_uniform_intervals(self, multi_trades_df):
        """TWAP with uniform intervals should sample correctly."""
        # 6 trades at 30s intervals over 3 minutes
        # With 1min resampling, should get 3 interval prices (last of each minute)
        # Minute 1: last price = 101.0 (10:00:30)
        # Minute 2: last price = 103.0 (10:01:30)
        # Minute 3: last price = 105.0 (10:02:30)
        # TWAP = (101 + 103 + 105) / 3 = 103.0

        twap = calculate_twap(multi_trades_df, interval="1min")

        assert abs(twap - 103.0) < 0.0001

    def test_calculate_twap_sparse_intervals(self):
        """TWAP should handle intervals with no trades."""
        # Only 2 trades, 5 minutes apart
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0, 0),
                    datetime(2024, 1, 1, 10, 5, 0),
                ],
                "price": [100.0, 110.0],
            }
        )

        twap = calculate_twap(df, interval="1min")

        # Only 2 intervals have trades, mean = (100 + 110) / 2 = 105
        assert twap == 105.0

    def test_calculate_twap_empty_df(self):
        """Empty DataFrame should return NaN."""
        df = pd.DataFrame({"timestamp": [], "price": []})

        twap = calculate_twap(df)

        assert np.isnan(twap)

    def test_calculate_twap_single_trade(self):
        """Single trade should return that trade's price."""
        df = pd.DataFrame(
            {"timestamp": [datetime(2024, 1, 1, 10, 0, 0)], "price": [50.0]}
        )

        twap = calculate_twap(df)

        assert twap == 50.0


class TestCalculateTwapFromOrderbooks:
    """Tests for calculate_twap_from_orderbooks function."""

    def test_twap_from_orderbooks(self, orderbook_df):
        """TWAP from orderbooks should use mid_price column."""
        # 5 snapshots at 30s intervals over 2.5 minutes
        # With 1min resampling, should get 3 interval prices
        # Minute 1: last mid_price = 0.51 (10:00:30)
        # Minute 2: last mid_price = 0.51 (10:01:30)
        # Minute 3: last mid_price = 0.50 (10:02:00) - only one in this minute
        # TWAP = (0.51 + 0.51 + 0.50) / 3 = 0.506667

        twap = calculate_twap_from_orderbooks(orderbook_df, interval="1min")

        assert abs(twap - 0.5066666666666666) < 0.0001

    def test_twap_from_orderbooks_empty_df(self):
        """Empty DataFrame should return NaN."""
        df = pd.DataFrame({"timestamp": [], "mid_price": []})

        twap = calculate_twap_from_orderbooks(df)

        assert np.isnan(twap)

    def test_twap_from_orderbooks_single_snapshot(self):
        """Single snapshot should return that snapshot's mid_price."""
        df = pd.DataFrame(
            {"timestamp": [datetime(2024, 1, 1, 10, 0, 0)], "mid_price": [0.55]}
        )

        twap = calculate_twap_from_orderbooks(df)

        assert twap == 0.55


# =============================================================================
# Arrival Price Tests
# =============================================================================


class TestGetArrivalPrice:
    """Tests for get_arrival_price function."""

    def test_get_arrival_price_exact_match(self, mock_reader):
        """Should return mid_price when snapshot is at exact order time."""
        order_time = datetime(2024, 1, 1, 12, 0, 0)

        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame(
            {
                "timestamp": [order_time],
                "mid_price": [0.65],
            }
        )

        arrival = get_arrival_price(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            order_time=order_time,
        )

        assert arrival == 0.65

    def test_get_arrival_price_closest_before(self, mock_reader):
        """Should return snapshot closest to but not after order time."""
        order_time = datetime(2024, 1, 1, 12, 0, 0)

        # Snapshots: 2 seconds before and 4 seconds before
        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame(
            {
                "timestamp": [
                    order_time - timedelta(seconds=4),
                    order_time - timedelta(seconds=2),
                ],
                "mid_price": [0.60, 0.65],
            }
        )

        arrival = get_arrival_price(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            order_time=order_time,
        )

        # Should pick the closest (2 seconds before) with mid_price 0.65
        assert arrival == 0.65

    def test_get_arrival_price_none_found(self, mock_reader):
        """Should return None if no snapshot in window."""
        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame(
            {"timestamp": [], "mid_price": []}
        )

        arrival = get_arrival_price(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            order_time=datetime(2024, 1, 1, 12, 0, 0),
        )

        assert arrival is None

    def test_get_arrival_price_respects_lookback(self, mock_reader):
        """Should only search within lookback window."""
        order_time = datetime(2024, 1, 1, 12, 0, 0)

        # Call with default lookback (5 seconds)
        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame()

        get_arrival_price(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            order_time=order_time,
            lookback_seconds=10,
        )

        # Verify the query window is correct
        call_args = mock_reader.query_orderbook_snapshots.call_args
        assert call_args.kwargs["start_time"] == order_time - timedelta(seconds=10)

    def test_get_arrival_price_excludes_future_snapshots(self, mock_reader):
        """Should not use snapshots after order time."""
        order_time = datetime(2024, 1, 1, 12, 0, 0)

        # Snapshots: one before and one slightly after order_time
        mock_reader.query_orderbook_snapshots.return_value = pd.DataFrame(
            {
                "timestamp": [
                    order_time - timedelta(seconds=2),
                    order_time + timedelta(milliseconds=100),
                ],
                "mid_price": [0.60, 0.70],
            }
        )

        arrival = get_arrival_price(
            reader=mock_reader,
            market_id="market-123",
            token_id="token-yes",
            order_time=order_time,
        )

        # Should use the one before (0.60), not the one after (0.70)
        assert arrival == 0.60
