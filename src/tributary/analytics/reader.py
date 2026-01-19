"""QuestDB reader for querying market data."""

import json
import logging
import re
from datetime import datetime
from typing import Optional, List, Any

import pandas as pd
import psycopg2

from tributary.core.config import QuestDBConfig

logger = logging.getLogger(__name__)


class QuestDBReader:
    """
    QuestDB reader for querying historical market data.

    Uses psycopg2 for PostgreSQL wire protocol access to QuestDB.
    Mirrors QuestDBWriter's connection pattern for consistency.

    Example:
        reader = QuestDBReader(config)
        reader.connect()
        trades_df = reader.query_trades(
            market_id="condition-123",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
        )
        reader.close()
    """

    # Valid interval patterns for SAMPLE BY
    _VALID_INTERVAL_PATTERN = re.compile(r"^\d+[smhd]$")

    def __init__(self, config: QuestDBConfig):
        """
        Initialize the QuestDB reader.

        Args:
            config: QuestDB configuration with connection details
        """
        self.config = config
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """
        Establish connection to QuestDB.

        Uses the PostgreSQL wire protocol on pg_port (default 8812).
        Sets autocommit=True as required by QuestDB (no transaction support).
        """
        try:
            self._conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.pg_port,
                user=self.config.pg_user,
                password=self.config.pg_password,
                database="qdb",
            )
            # QuestDB requires autocommit mode - it doesn't support transactions
            self._conn.autocommit = True
            logger.info(
                f"Connected to QuestDB at {self.config.host}:{self.config.pg_port}"
            )
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to QuestDB: {e}")
            self._conn = None
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
                logger.info("QuestDB reader connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._conn = None

    @property
    def is_connected(self) -> bool:
        """Check if reader is connected to the database."""
        return self._conn is not None and not self._conn.closed

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        Execute a query and return results as a DataFrame.

        Args:
            query: SQL query string with %s placeholders for parameters
            params: Optional tuple of parameter values

        Returns:
            DataFrame with query results

        Raises:
            RuntimeError: If not connected
            psycopg2.Error: On query execution failure
        """
        if not self.is_connected:
            raise RuntimeError("Reader not connected. Call connect() first.")

        with self._conn.cursor() as cur:
            cur.execute(query, params)

            # Handle queries that return no rows
            if cur.description is None:
                return pd.DataFrame()

            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            return pd.DataFrame(rows, columns=columns)

    def query_trades(
        self,
        market_id: str,
        start_time: datetime,
        end_time: datetime,
        token_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query trades for a market within a time range.

        Args:
            market_id: Market identifier (conditionId for Polymarket)
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)
            token_id: Optional token/outcome identifier to filter by

        Returns:
            DataFrame with columns: timestamp, market_id, token_id, trade_id,
            price, size, side, value
        """
        query = """
            SELECT
                timestamp,
                market_id,
                token_id,
                trade_id,
                price,
                size,
                side,
                value
            FROM trades
            WHERE market_id = %s
              AND timestamp >= %s
              AND timestamp < %s
        """
        params: List[Any] = [market_id, start_time, end_time]

        if token_id is not None:
            query += " AND token_id = %s"
            params.append(token_id)

        query += " ORDER BY timestamp ASC"

        return self.execute_query(query, tuple(params))

    def query_orderbook_snapshots(
        self,
        market_id: str,
        start_time: datetime,
        end_time: datetime,
        token_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query orderbook snapshots for a market within a time range.

        Args:
            market_id: Market identifier (conditionId for Polymarket)
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)
            token_id: Optional token/outcome identifier to filter by

        Returns:
            DataFrame with columns: timestamp, market_id, token_id, best_bid,
            best_ask, mid_price, spread, spread_bps, bid_prices, bid_sizes,
            ask_prices, ask_sizes

        Note:
            bid_prices, bid_sizes, ask_prices, ask_sizes are returned as Python
            lists (parsed from JSON storage format).
        """
        query = """
            SELECT
                timestamp,
                market_id,
                token_id,
                best_bid,
                best_ask,
                mid_price,
                spread,
                spread_bps,
                bid_prices,
                bid_sizes,
                ask_prices,
                ask_sizes
            FROM orderbook_snapshots
            WHERE market_id = %s
              AND timestamp >= %s
              AND timestamp < %s
        """
        params: List[Any] = [market_id, start_time, end_time]

        if token_id is not None:
            query += " AND token_id = %s"
            params.append(token_id)

        query += " ORDER BY timestamp ASC"

        df = self.execute_query(query, tuple(params))

        # Parse JSON columns to Python lists
        if not df.empty:
            for col in ["bid_prices", "bid_sizes", "ask_prices", "ask_sizes"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x
                    )

        return df

    def query_vwap_sampled(
        self,
        market_id: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Query time-bucketed VWAP using QuestDB's SAMPLE BY.

        Efficiently computes volume-weighted average price aggregated by time
        intervals using QuestDB's native time-series capabilities.

        Args:
            market_id: Market identifier (conditionId for Polymarket)
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)
            interval: Time bucket interval (e.g., "1h", "15m", "1d")
                     Must be a number followed by s/m/h/d

        Returns:
            DataFrame with columns: timestamp, vwap, volume, trade_count

        Raises:
            ValueError: If interval format is invalid
        """
        # Validate interval format to prevent SQL injection
        # Valid formats: 1h, 15m, 30s, 1d
        if not self._VALID_INTERVAL_PATTERN.match(interval):
            raise ValueError(
                f"Invalid interval format: {interval}. "
                f"Must be a number followed by s/m/h/d (e.g., '1h', '15m')"
            )

        # Interval must be string-interpolated (QuestDB limitation with SAMPLE BY)
        # This is safe because we validated the format above
        query = f"""
            SELECT
                timestamp,
                sum(price * size) / sum(size) as vwap,
                sum(size) as volume,
                count() as trade_count
            FROM trades
            WHERE market_id = %s
              AND timestamp >= %s
              AND timestamp < %s
            SAMPLE BY {interval} ALIGN TO CALENDAR
        """

        return self.execute_query(query, (market_id, start_time, end_time))
