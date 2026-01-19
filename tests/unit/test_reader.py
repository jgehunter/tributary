"""Unit tests for QuestDBReader."""

from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from tributary.analytics.reader import QuestDBReader
from tributary.core.config import QuestDBConfig


@pytest.fixture
def config():
    """Create a test QuestDB configuration."""
    return QuestDBConfig(
        host="localhost",
        pg_port=8812,
        pg_user="admin",
        pg_password="quest",
    )


@pytest.fixture
def reader(config):
    """Create a QuestDBReader instance."""
    return QuestDBReader(config)


class TestConnectionManagement:
    """Tests for connection management."""

    def test_is_connected_before_connect(self, reader):
        """Reader should not be connected initially."""
        assert reader.is_connected is False

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_connect_success(self, mock_connect, reader):
        """Test successful connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        reader.connect()

        mock_connect.assert_called_once_with(
            host="localhost",
            port=8812,
            user="admin",
            password="quest",
            database="qdb",
        )
        # Verify autocommit is set (QuestDB requirement)
        assert mock_conn.autocommit is True

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_is_connected_after_connect(self, mock_connect, reader):
        """Reader should be connected after connect()."""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_connect.return_value = mock_conn

        reader.connect()

        assert reader.is_connected is True

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_close_clears_connection(self, mock_connect, reader):
        """Connection should be cleared after close()."""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_connect.return_value = mock_conn

        reader.connect()
        assert reader.is_connected is True

        reader.close()
        assert reader.is_connected is False
        mock_conn.close.assert_called_once()


class TestQueryTrades:
    """Tests for query_trades method."""

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_query_trades_returns_dataframe(self, mock_connect, reader):
        """query_trades should return a DataFrame."""
        # Setup mock connection and cursor
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("timestamp",), ("market_id",), ("token_id",), ("trade_id",),
            ("price",), ("size",), ("side",), ("value",),
        ]
        mock_cursor.fetchall.return_value = [
            (datetime(2024, 1, 15, 12, 0, 0), "market-123", "token-1", "trade-001",
             0.65, 100.0, "buy", 65.0),
        ]

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()
        df = reader.query_trades(
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["market_id"] == "market-123"
        assert df.iloc[0]["price"] == 0.65

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_query_trades_with_token_id(self, mock_connect, reader):
        """query_trades should filter by token_id when provided."""
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("timestamp",), ("market_id",), ("token_id",), ("trade_id",),
            ("price",), ("size",), ("side",), ("value",),
        ]
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()
        reader.query_trades(
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            token_id="token-yes",
        )

        # Verify the query includes token_id filter
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]

        assert "token_id = %s" in query
        assert "token-yes" in params

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_query_trades_empty_result(self, mock_connect, reader):
        """query_trades should return empty DataFrame, not error."""
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("timestamp",), ("market_id",), ("token_id",), ("trade_id",),
            ("price",), ("size",), ("side",), ("value",),
        ]
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()
        df = reader.query_trades(
            market_id="nonexistent-market",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestQueryOrderbookSnapshots:
    """Tests for query_orderbook_snapshots method."""

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_query_orderbooks_returns_dataframe(self, mock_connect, reader):
        """query_orderbook_snapshots should return a DataFrame."""
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("timestamp",), ("market_id",), ("token_id",), ("best_bid",),
            ("best_ask",), ("mid_price",), ("spread",), ("spread_bps",),
            ("bid_prices",), ("bid_sizes",), ("ask_prices",), ("ask_sizes",),
        ]
        mock_cursor.fetchall.return_value = [
            (datetime(2024, 1, 15, 12, 0, 0), "market-123", "token-1", 0.64,
             0.66, 0.65, 0.02, 307.7,
             "[0.64, 0.63]", "[500.0, 300.0]", "[0.66, 0.67]", "[300.0, 400.0]"),
        ]

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()
        df = reader.query_orderbook_snapshots(
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["best_bid"] == 0.64
        assert df.iloc[0]["best_ask"] == 0.66

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_query_orderbooks_parses_json_columns(self, mock_connect, reader):
        """JSON columns should be parsed to Python lists."""
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("timestamp",), ("market_id",), ("token_id",), ("best_bid",),
            ("best_ask",), ("mid_price",), ("spread",), ("spread_bps",),
            ("bid_prices",), ("bid_sizes",), ("ask_prices",), ("ask_sizes",),
        ]
        mock_cursor.fetchall.return_value = [
            (datetime(2024, 1, 15, 12, 0, 0), "market-123", "token-1", 0.64,
             0.66, 0.65, 0.02, 307.7,
             "[0.64, 0.63, 0.62]", "[500.0, 300.0, 200.0]",
             "[0.66, 0.67, 0.68]", "[300.0, 400.0, 250.0]"),
        ]

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()
        df = reader.query_orderbook_snapshots(
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
        )

        # Verify JSON columns are parsed to lists
        assert isinstance(df.iloc[0]["bid_prices"], list)
        assert df.iloc[0]["bid_prices"] == [0.64, 0.63, 0.62]
        assert isinstance(df.iloc[0]["bid_sizes"], list)
        assert df.iloc[0]["bid_sizes"] == [500.0, 300.0, 200.0]
        assert isinstance(df.iloc[0]["ask_prices"], list)
        assert df.iloc[0]["ask_prices"] == [0.66, 0.67, 0.68]
        assert isinstance(df.iloc[0]["ask_sizes"], list)
        assert df.iloc[0]["ask_sizes"] == [300.0, 400.0, 250.0]


class TestQueryVwapSampled:
    """Tests for query_vwap_sampled method."""

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_vwap_sampled_uses_sample_by(self, mock_connect, reader):
        """VWAP query should use SAMPLE BY clause."""
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("timestamp",), ("vwap",), ("volume",), ("trade_count",),
        ]
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()
        reader.query_vwap_sampled(
            market_id="market-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            interval="1h",
        )

        # Verify SAMPLE BY is in the query
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]

        assert "SAMPLE BY" in query
        assert "ALIGN TO CALENDAR" in query

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_vwap_sampled_interval_interpolation(self, mock_connect, reader):
        """Interval should be interpolated into SAMPLE BY clause."""
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("timestamp",), ("vwap",), ("volume",), ("trade_count",),
        ]
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()

        # Test different intervals
        for interval in ["15m", "1h", "1d", "30s"]:
            reader.query_vwap_sampled(
                market_id="market-123",
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 31),
                interval=interval,
            )

            call_args = mock_cursor.execute.call_args
            query = call_args[0][0]

            assert f"SAMPLE BY {interval}" in query

    def test_vwap_sampled_invalid_interval_raises(self, reader):
        """Invalid interval format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            reader.query_vwap_sampled(
                market_id="market-123",
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 31),
                interval="invalid",
            )

    def test_vwap_sampled_sql_injection_prevention(self, reader):
        """Malicious interval should be rejected."""
        malicious_intervals = [
            "1h; DROP TABLE trades;",
            "1h--",
            "1h'",
            "'; DELETE FROM trades; --",
        ]

        for interval in malicious_intervals:
            with pytest.raises(ValueError, match="Invalid interval format"):
                reader.query_vwap_sampled(
                    market_id="market-123",
                    start_time=datetime(2024, 1, 1),
                    end_time=datetime(2024, 1, 31),
                    interval=interval,
                )


class TestExecuteQuery:
    """Tests for execute_query method."""

    def test_execute_query_not_connected_raises(self, reader):
        """execute_query should raise if not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            reader.execute_query("SELECT 1")

    @patch("tributary.analytics.reader.psycopg2.connect")
    def test_execute_query_returns_dataframe(self, mock_connect, reader):
        """execute_query should return a DataFrame."""
        mock_cursor = MagicMock()
        mock_cursor.description = [("value",)]
        mock_cursor.fetchall.return_value = [(1,)]

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        reader.connect()
        df = reader.execute_query("SELECT 1 as value")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["value"] == 1
