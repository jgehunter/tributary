"""Event types for execution simulation.

This module defines the core event types that flow through the event-driven
simulation engine. Events are immutable (frozen dataclasses) to ensure
reproducible simulations.

Event Types:
    MarketEvent: Orderbook state at a point in time (from historical replay)
    OrderEvent: Trading instruction from a strategy
    FillEvent: Execution result with slippage and fill details

Example:
    >>> from datetime import datetime, timezone
    >>> market = MarketEvent(
    ...     timestamp=datetime.now(timezone.utc),
    ...     market_id="condition-123",
    ...     token_id="token-yes",
    ...     mid_price=0.50,
    ...     bid_prices=(0.49, 0.48, 0.47),
    ...     bid_sizes=(1000.0, 2000.0, 3000.0),
    ...     ask_prices=(0.51, 0.52, 0.53),
    ...     ask_sizes=(1000.0, 2000.0, 3000.0),
    ... )
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketEvent:
    """Orderbook update from historical data replay.

    Captures the complete orderbook state at a specific point in time.
    Used as input to the fill model for simulating order execution.

    Attributes:
        timestamp: Time of the orderbook snapshot
        market_id: Market identifier (conditionId for Polymarket)
        token_id: Token/outcome identifier
        mid_price: Reference price (best bid + best ask) / 2
        bid_prices: Bid prices in descending order (best bid first)
        bid_sizes: Sizes at each bid level
        ask_prices: Ask prices in ascending order (best ask first)
        ask_sizes: Sizes at each ask level

    Note:
        Uses tuples instead of lists for immutability in frozen dataclass.
    """

    timestamp: datetime
    market_id: str
    token_id: str
    mid_price: float
    bid_prices: tuple[float, ...]
    bid_sizes: tuple[float, ...]
    ask_prices: tuple[float, ...]
    ask_sizes: tuple[float, ...]


@dataclass(frozen=True)
class OrderEvent:
    """Order to be executed by the simulation engine.

    Represents a trading instruction from an execution strategy.
    Orders are executed against MarketEvents to produce FillEvents.

    Attributes:
        timestamp: Time the order was generated
        strategy_name: Name of the execution strategy ('twap', 'vwap', etc.)
        slice_index: Index of this slice in the execution schedule (0-based)
        size: Size to execute (in base units, always positive)
        side: Trade direction - 'buy' or 'sell'
    """

    timestamp: datetime
    strategy_name: str
    slice_index: int
    size: float
    side: str


@dataclass(frozen=True)
class FillEvent:
    """Execution result from simulating an order.

    Contains the full details of how an order was filled against
    the orderbook, including slippage and partial fill information.

    Attributes:
        timestamp: Time of fill execution
        strategy_name: Strategy that generated the original order
        slice_index: Index of this slice in the execution schedule
        requested_size: Original order size requested
        filled_size: Actual size filled (may be less than requested)
        avg_price: Volume-weighted average execution price
        slippage_bps: Slippage in basis points (positive = cost)
        levels_consumed: Number of orderbook levels used for fill
        mid_price_at_fill: Mid price at time of execution (for cost calculation)

    Note:
        slippage_bps follows the convention: positive = unfavorable execution.
        For buys, this means paying more than mid price.
        For sells, this means receiving less than mid price.
    """

    timestamp: datetime
    strategy_name: str
    slice_index: int
    requested_size: float
    filled_size: float
    avg_price: float
    slippage_bps: float
    levels_consumed: int
    mid_price_at_fill: float
