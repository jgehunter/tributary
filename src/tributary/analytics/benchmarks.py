"""Benchmark calculations for execution quality measurement.

Provides VWAP, TWAP, and arrival price calculations used for measuring
execution quality and implementation shortfall.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from tributary.analytics.reader import QuestDBReader


def calculate_vwap(trades_df: pd.DataFrame) -> float:
    """
    Calculate volume-weighted average price from trades.

    Standard VWAP formula: sum(price * size) / sum(size)

    Args:
        trades_df: DataFrame with 'price' and 'size' columns

    Returns:
        VWAP as float, or NaN if empty or zero total volume

    Example:
        >>> df = pd.DataFrame({'price': [100, 200], 'size': [10, 20]})
        >>> calculate_vwap(df)
        166.66666666666666
    """
    if trades_df.empty:
        return float("nan")

    total_volume = trades_df["size"].sum()
    if total_volume == 0:
        return float("nan")

    return (trades_df["price"] * trades_df["size"]).sum() / total_volume


def calculate_cumulative_vwap(trades_df: pd.DataFrame) -> pd.Series:
    """
    Calculate running VWAP over time (cumulative).

    At each trade, computes the VWAP of all trades up to and including that trade.

    Args:
        trades_df: DataFrame with 'price', 'size', and 'timestamp' columns

    Returns:
        Series indexed same as input, with cumulative VWAP at each trade.
        Returns empty Series if input is empty.

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
        ...     'price': [100, 200],
        ...     'size': [10, 20]
        ... })
        >>> calculate_cumulative_vwap(df)
        0    100.000000
        1    166.666667
        dtype: float64
    """
    if trades_df.empty:
        return pd.Series(dtype=float)

    # Sort by timestamp to ensure correct cumulative calculation
    df = trades_df.sort_values("timestamp").reset_index(drop=True)

    # Calculate cumulative sums
    cumulative_value = (df["price"] * df["size"]).cumsum()
    cumulative_volume = df["size"].cumsum()

    # Compute cumulative VWAP
    cumulative_vwap = cumulative_value / cumulative_volume

    return cumulative_vwap


def calculate_vwap_for_period(
    reader: QuestDBReader,
    market_id: str,
    token_id: str,
    start_time: datetime,
    end_time: datetime,
) -> float:
    """
    Query trades and calculate VWAP for a specific period.

    Convenience function that wraps reader.query_trades() and calculate_vwap().

    Args:
        reader: Connected QuestDBReader instance
        market_id: Market identifier (e.g., conditionId for Polymarket)
        token_id: Token/outcome identifier
        start_time: Start of time range (inclusive)
        end_time: End of time range (exclusive)

    Returns:
        VWAP for the period, or NaN if no trades found

    Example:
        >>> reader = QuestDBReader(config)
        >>> reader.connect()
        >>> vwap = calculate_vwap_for_period(
        ...     reader, "market-123", "token-yes",
        ...     datetime(2024, 1, 1), datetime(2024, 1, 2)
        ... )
    """
    trades_df = reader.query_trades(
        market_id=market_id,
        start_time=start_time,
        end_time=end_time,
        token_id=token_id,
    )

    return calculate_vwap(trades_df)


def calculate_twap(trades_df: pd.DataFrame, interval: str = "1min") -> float:
    """
    Calculate time-weighted average price from trades.

    Resamples trades to regular intervals, takes the last price per interval,
    and returns the mean of those interval prices.

    Args:
        trades_df: DataFrame with 'timestamp' and 'price' columns
        interval: Pandas resample interval (e.g., "1min", "5min", "1h")

    Returns:
        TWAP as float, or NaN if empty or no valid intervals

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=5, freq='30s'),
        ...     'price': [100, 101, 102, 103, 104]
        ... })
        >>> calculate_twap(df, interval="1min")  # ~101.5
    """
    if trades_df.empty:
        return float("nan")

    # Set timestamp as index for resampling
    df = trades_df.set_index("timestamp")

    # Resample and take last price per interval, drop empty intervals
    interval_prices = df["price"].resample(interval).last().dropna()

    if interval_prices.empty:
        return float("nan")

    return interval_prices.mean()


def calculate_twap_from_orderbooks(
    orderbooks_df: pd.DataFrame, interval: str = "1min"
) -> float:
    """
    Calculate time-weighted average price from orderbook mid-prices.

    More accurate for illiquid markets where trades are sparse.
    Uses mid_price column from orderbook snapshots.

    Args:
        orderbooks_df: DataFrame with 'timestamp' and 'mid_price' columns
        interval: Pandas resample interval (e.g., "1min", "5min", "1h")

    Returns:
        TWAP as float, or NaN if empty or no valid intervals

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=5, freq='30s'),
        ...     'mid_price': [0.50, 0.51, 0.52, 0.51, 0.50]
        ... })
        >>> calculate_twap_from_orderbooks(df, interval="1min")  # ~0.51
    """
    if orderbooks_df.empty:
        return float("nan")

    # Set timestamp as index for resampling
    df = orderbooks_df.set_index("timestamp")

    # Resample and take last mid_price per interval, drop empty intervals
    interval_prices = df["mid_price"].resample(interval).last().dropna()

    if interval_prices.empty:
        return float("nan")

    return interval_prices.mean()


def get_arrival_price(
    reader: QuestDBReader,
    market_id: str,
    token_id: str,
    order_time: datetime,
    lookback_seconds: int = 5,
) -> Optional[float]:
    """
    Get mid-price at order submission time (arrival price).

    Queries for the orderbook snapshot closest to (but not after) the order time.
    The arrival price is used as the reference for measuring implementation shortfall.

    Args:
        reader: Connected QuestDBReader instance
        market_id: Market identifier (e.g., conditionId for Polymarket)
        token_id: Token/outcome identifier
        order_time: Time when order was submitted
        lookback_seconds: How far back to search for snapshot (default 5 seconds)

    Returns:
        Mid-price at order submission time, or None if no snapshot found in window

    Example:
        >>> reader = QuestDBReader(config)
        >>> reader.connect()
        >>> arrival = get_arrival_price(
        ...     reader, "market-123", "token-yes",
        ...     datetime(2024, 1, 1, 12, 0, 0)
        ... )
        >>> arrival  # 0.65 or None
    """
    # Define search window: [order_time - lookback, order_time]
    window_start = order_time - timedelta(seconds=lookback_seconds)
    window_end = order_time + timedelta(milliseconds=1)  # Inclusive of order_time

    # Query orderbook snapshots in the window
    snapshots_df = reader.query_orderbook_snapshots(
        market_id=market_id,
        start_time=window_start,
        end_time=window_end,
        token_id=token_id,
    )

    if snapshots_df.empty:
        return None

    # Get the snapshot closest to (but not after) order_time
    # Snapshots are ordered by timestamp ASC, so we want the last one <= order_time
    valid_snapshots = snapshots_df[snapshots_df["timestamp"] <= order_time]

    if valid_snapshots.empty:
        return None

    # Return the mid_price of the closest snapshot (last one in ascending order)
    return float(valid_snapshots.iloc[-1]["mid_price"])
