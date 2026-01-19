"""Event-driven simulation engine for execution backtesting.

This module provides the SimulationEngine class that orchestrates the
replay of historical market data against execution trajectories.

The engine processes market events in strict timestamp order, generating
orders from the provided execution trajectory and filling them against
the current orderbook state. This ensures no lookahead bias - strategies
only see past/current market state, never future data.

Key Design Principles:
    1. No lookahead: Orders execute against most recent market state
    2. Time ordering: Market data processed in timestamp order
    3. Trajectory mapping: Trade sizes from trajectory execute at scheduled times
    4. Fill model integration: Realistic execution with liquidity consumption

Example:
    >>> from datetime import datetime, timedelta, timezone
    >>> import pandas as pd
    >>> from tributary.analytics.simulation import SimulationEngine, FillModel
    >>> from tributary.analytics.optimization import generate_twap_trajectory
    >>>
    >>> # Create engine
    >>> engine = SimulationEngine(fill_model=FillModel())
    >>>
    >>> # Generate trajectory
    >>> trajectory = generate_twap_trajectory(order_size=1000, duration_periods=5)
    >>>
    >>> # Run simulation (requires market_data DataFrame)
    >>> fills = engine.run(
    ...     trajectory=trajectory,
    ...     market_data=market_data,
    ...     side='buy',
    ...     start_time=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    ...     interval=timedelta(seconds=1),
    ... )
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from tributary.analytics.optimization import ExecutionTrajectory
from tributary.analytics.simulation.events import FillEvent, MarketEvent, OrderEvent
from tributary.analytics.simulation.fill_model import FillModel


class SimulationEngine:
    """Event-driven execution simulation engine.

    Processes market events in timestamp order, generating orders from
    the provided execution trajectory and filling them against the orderbook.

    Key design:
    - Events processed in strict time order (no lookahead)
    - Orders generated at trajectory timestamps
    - Fill model handles execution against current market state

    Attributes:
        fill_model: FillModel instance for order execution simulation
    """

    def __init__(
        self,
        fill_model: Optional[FillModel] = None,
    ):
        """Initialize the simulation engine.

        Args:
            fill_model: FillModel for execution simulation. If None,
                creates a default FillModel instance.
        """
        self.fill_model = fill_model or FillModel()
        self._current_market: Optional[MarketEvent] = None

    def run(
        self,
        trajectory: ExecutionTrajectory,
        market_data: pd.DataFrame,
        side: str,
        start_time: datetime,
        interval: timedelta,
    ) -> list[FillEvent]:
        """Run simulation of a single strategy.

        Replays historical market data and executes the trajectory's trade
        slices at the scheduled times. Each slice executes against the most
        recent market state (no lookahead).

        Args:
            trajectory: ExecutionTrajectory with trade_sizes to execute
            market_data: DataFrame with orderbook snapshots.
                Required columns: timestamp, mid_price, bid_prices, bid_sizes,
                ask_prices, ask_sizes. Optional: market_id, token_id
            side: 'buy' or 'sell' for all orders
            start_time: When to begin execution (arrival time)
            interval: Time between execution slices

        Returns:
            List of FillEvent objects, one per executed slice

        Note:
            - Market data must contain 'timestamp' column
            - For each slice, uses most recent market state at or before slice time
            - Slices with no prior market data are skipped
            - Fill model tracks liquidity consumption across slices
        """
        # Step 1: Reset fill model state for fresh simulation
        self.fill_model.reset()
        self._current_market = None

        # Handle edge cases
        if market_data.empty:
            return []

        if len(trajectory.trade_sizes) == 0:
            return []

        # Check if all trade sizes are zero or NaN
        if all(
            size <= 0 or pd.isna(size)
            for size in trajectory.trade_sizes
        ):
            return []

        # Step 2: Ensure market data is sorted by timestamp
        market_data = market_data.sort_values("timestamp").reset_index(drop=True)

        # Step 3: Build schedule of order times from trajectory
        order_times = []
        for i in range(len(trajectory.trade_sizes)):
            order_time = start_time + (i * interval)
            order_times.append(order_time)

        # Step 4: Execute each slice against the appropriate market state
        fills: list[FillEvent] = []

        for slice_index, (order_time, trade_size) in enumerate(
            zip(order_times, trajectory.trade_sizes)
        ):
            # Skip zero or invalid trade sizes
            if trade_size <= 0 or pd.isna(trade_size):
                continue

            # Find most recent market state at or before order_time (no lookahead!)
            market_event = self._find_market_at_time(market_data, order_time)

            if market_event is None:
                # No market data before this slice time - skip
                continue

            # Create order event
            order = OrderEvent(
                timestamp=order_time,
                strategy_name=trajectory.strategy_name,
                slice_index=slice_index,
                size=trade_size,
                side=side,
            )

            # Execute via fill model
            fill = self.fill_model.execute(order, market_event)
            fills.append(fill)

        return fills

    def _find_market_at_time(
        self,
        market_data: pd.DataFrame,
        target_time: datetime,
    ) -> Optional[MarketEvent]:
        """Find most recent market state at or before target time.

        Args:
            market_data: Sorted DataFrame with timestamp column
            target_time: Time to find market state for

        Returns:
            MarketEvent or None if no data before target_time
        """
        # Filter to rows at or before target_time
        valid_rows = market_data[market_data["timestamp"] <= target_time]

        if valid_rows.empty:
            return None

        # Get most recent row
        row = valid_rows.iloc[-1]

        # Convert DataFrame row to MarketEvent
        return MarketEvent(
            timestamp=row["timestamp"],
            market_id=row.get("market_id", "unknown"),
            token_id=row.get("token_id", "unknown"),
            mid_price=row["mid_price"],
            bid_prices=tuple(row["bid_prices"]) if hasattr(row["bid_prices"], "__iter__") else (row["bid_prices"],),
            bid_sizes=tuple(row["bid_sizes"]) if hasattr(row["bid_sizes"], "__iter__") else (row["bid_sizes"],),
            ask_prices=tuple(row["ask_prices"]) if hasattr(row["ask_prices"], "__iter__") else (row["ask_prices"],),
            ask_sizes=tuple(row["ask_sizes"]) if hasattr(row["ask_sizes"], "__iter__") else (row["ask_sizes"],),
        )
