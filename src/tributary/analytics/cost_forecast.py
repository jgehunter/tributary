"""Orderbook-based cost forecasting using walk-the-book algorithm.

This module provides the PRIMARY method for estimating execution costs in thin
liquidity markets like Polymarket. Unlike model-based approaches (square-root law),
this directly measures available liquidity at each price level.

Example:
    # Direct estimation from orderbook data
    forecast = estimate_slippage_from_orderbook(
        order_size=10000,
        side='buy',
        bid_prices=[0.52, 0.51, 0.50],
        bid_sizes=[5000, 8000, 10000],
        ask_prices=[0.53, 0.54, 0.55],
        ask_sizes=[5000, 8000, 10000],
    )
    print(f"Expected slippage: {forecast.slippage_bps:.2f} bps")

    # Convenience function with QuestDBReader
    forecast = forecast_execution_cost(
        reader=reader,
        market_id="condition-123",
        token_id="token-yes",
        order_size=10000,
        side='buy',
    )
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from tributary.analytics.reader import QuestDBReader


@dataclass
class CostForecast:
    """
    Result of orderbook-based execution cost estimation.

    Attributes:
        mid_price: Reference price (best bid + best ask) / 2
        expected_execution_price: VWAP across levels consumed
        slippage_bps: Expected slippage in basis points (positive = cost)
        levels_consumed: Number of orderbook levels used
        fully_filled: Whether order can be completely filled
        unfilled_size: Remaining size if partially filled
        total_cost: Total dollar cost (size * expected_price)
    """

    mid_price: float
    expected_execution_price: float
    slippage_bps: float
    levels_consumed: int
    fully_filled: bool
    unfilled_size: float
    total_cost: float


def estimate_slippage_from_orderbook(
    order_size: float,
    side: str,
    bid_prices: List[float],
    bid_sizes: List[float],
    ask_prices: List[float],
    ask_sizes: List[float],
) -> CostForecast:
    """
    Estimate execution cost by walking the orderbook.

    Simulates order execution across orderbook levels to estimate expected
    slippage for a given order size. This is the PRIMARY method for thin
    liquidity markets where model-based approaches are unreliable.

    Args:
        order_size: Size to execute (in base currency/shares)
        side: Trade direction - 'buy' or 'sell'
        bid_prices: Bid prices in descending order (best bid first)
        bid_sizes: Sizes at each bid level
        ask_prices: Ask prices in ascending order (best ask first)
        ask_sizes: Sizes at each ask level

    Returns:
        CostForecast with execution estimates

    Raises:
        ValueError: If side is not 'buy' or 'sell'

    Note:
        - Positive slippage_bps = cost (unfavorable execution)
        - Negative slippage_bps = gain (favorable execution, rare)
        - For buy orders: walks ask side (ascending prices)
        - For sell orders: walks bid side (descending prices)
    """
    # Validate side
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")

    # Handle zero order size
    if order_size <= 0:
        return CostForecast(
            mid_price=float("nan"),
            expected_execution_price=float("nan"),
            slippage_bps=float("nan"),
            levels_consumed=0,
            fully_filled=True,
            unfilled_size=0.0,
            total_cost=0.0,
        )

    # Handle empty orderbook
    if not bid_prices or not ask_prices:
        return CostForecast(
            mid_price=float("nan"),
            expected_execution_price=float("nan"),
            slippage_bps=float("nan"),
            levels_consumed=0,
            fully_filled=False,
            unfilled_size=order_size,
            total_cost=0.0,
        )

    # Calculate mid price
    mid_price = (bid_prices[0] + ask_prices[0]) / 2

    # Select relevant side of book
    if side == "buy":
        prices = ask_prices
        sizes = ask_sizes
    else:
        prices = bid_prices
        sizes = bid_sizes

    # Walk the book
    remaining = order_size
    total_cost = 0.0
    total_filled = 0.0
    levels_consumed = 0

    for price, size in zip(prices, sizes):
        if remaining <= 0:
            break

        fill_size = min(remaining, size)
        total_cost += fill_size * price
        total_filled += fill_size
        remaining -= fill_size
        levels_consumed += 1

    # Handle case where no fill occurred (shouldn't happen with valid orderbook)
    if total_filled == 0:
        return CostForecast(
            mid_price=mid_price,
            expected_execution_price=float("nan"),
            slippage_bps=float("nan"),
            levels_consumed=0,
            fully_filled=False,
            unfilled_size=order_size,
            total_cost=0.0,
        )

    # Calculate volume-weighted average execution price
    avg_price = total_cost / total_filled

    # Calculate slippage in basis points
    # Convention: positive = cost (paid more for buys, received less for sells)
    if side == "buy":
        slippage_bps = (avg_price - mid_price) / mid_price * 10000
    else:
        slippage_bps = (mid_price - avg_price) / mid_price * 10000

    return CostForecast(
        mid_price=mid_price,
        expected_execution_price=avg_price,
        slippage_bps=slippage_bps,
        levels_consumed=levels_consumed,
        fully_filled=remaining <= 0,
        unfilled_size=max(0.0, remaining),
        total_cost=total_cost,
    )


def forecast_execution_cost(
    reader: QuestDBReader,
    market_id: str,
    token_id: str,
    order_size: float,
    side: str,
    as_of_time: Optional[datetime] = None,
    lookback_seconds: int = 60,
) -> CostForecast:
    """
    Forecast execution cost using the latest orderbook snapshot.

    Convenience function that queries the orderbook and calls
    estimate_slippage_from_orderbook.

    Args:
        reader: Connected QuestDBReader instance
        market_id: Market identifier (conditionId for Polymarket)
        token_id: Token/outcome identifier
        order_size: Size to execute
        side: Trade direction - 'buy' or 'sell'
        as_of_time: Time to query orderbook at (default: now)
        lookback_seconds: How far back to search for orderbook (default: 60s)

    Returns:
        CostForecast with execution estimates

    Raises:
        ValueError: If no orderbook data available in the lookback window
    """
    if as_of_time is None:
        as_of_time = datetime.now(timezone.utc)

    # Query orderbook snapshots in the lookback window
    start_time = as_of_time - timedelta(seconds=lookback_seconds)
    end_time = as_of_time + timedelta(milliseconds=1)  # Include exact time

    df = reader.query_orderbook_snapshots(
        market_id=market_id,
        start_time=start_time,
        end_time=end_time,
        token_id=token_id,
    )

    if df.empty:
        raise ValueError(
            f"No orderbook data found for market {market_id}, token {token_id} "
            f"in {lookback_seconds}s window before {as_of_time}"
        )

    # Use the most recent snapshot
    latest = df.iloc[-1]

    return estimate_slippage_from_orderbook(
        order_size=order_size,
        side=side,
        bid_prices=latest["bid_prices"],
        bid_sizes=latest["bid_sizes"],
        ask_prices=latest["ask_prices"],
        ask_sizes=latest["ask_sizes"],
    )
