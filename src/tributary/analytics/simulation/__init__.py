"""Simulation engine for backtesting execution strategies.

This package provides an event-driven simulation framework for testing
execution strategies against historical orderbook data.

Core Components:
    MarketEvent: Orderbook state at a point in time
    OrderEvent: Trading instruction from a strategy
    FillEvent: Execution result with slippage details
    FillModel: Realistic fill simulation with liquidity consumption

Example:
    >>> from datetime import datetime, timezone
    >>> from tributary.analytics.simulation import (
    ...     MarketEvent, OrderEvent, FillModel
    ... )
    >>>
    >>> # Create market snapshot
    >>> market = MarketEvent(
    ...     timestamp=datetime.now(timezone.utc),
    ...     market_id="test",
    ...     token_id="test",
    ...     mid_price=0.50,
    ...     bid_prices=(0.49, 0.48, 0.47),
    ...     bid_sizes=(1000.0, 2000.0, 3000.0),
    ...     ask_prices=(0.51, 0.52, 0.53),
    ...     ask_sizes=(1000.0, 2000.0, 3000.0),
    ... )
    >>>
    >>> # Create order
    >>> order = OrderEvent(
    ...     timestamp=datetime.now(timezone.utc),
    ...     strategy_name="twap",
    ...     slice_index=0,
    ...     size=500.0,
    ...     side="buy",
    ... )
    >>>
    >>> # Execute order
    >>> model = FillModel()
    >>> fill = model.execute(order, market)
    >>> print(f"Filled {fill.filled_size} at {fill.avg_price:.4f}")
"""

from tributary.analytics.simulation.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
)
from tributary.analytics.simulation.fill_model import FillModel
from tributary.analytics.simulation.engine import SimulationEngine
from tributary.analytics.simulation.runner import StrategyRun, StrategyRunner

__all__ = [
    "MarketEvent",
    "OrderEvent",
    "FillEvent",
    "FillModel",
    "SimulationEngine",
    "StrategyRun",
    "StrategyRunner",
]
