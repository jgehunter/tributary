"""Realistic fill model with liquidity consumption and recovery.

This module provides the FillModel class that simulates order execution
against an orderbook with realistic market microstructure effects:

- Walks the orderbook to fill orders (like estimate_slippage_from_orderbook)
- Tracks consumed liquidity per price level
- Models partial liquidity recovery between execution slices
- Differentiates aggressive (market order) vs patient (TWAP) execution

The fill model is essential for backtesting execution strategies because
it captures the market impact of repeated trading on the same orderbook.

Example:
    >>> from datetime import datetime, timezone
    >>> from tributary.analytics.simulation import FillModel, MarketEvent, OrderEvent
    >>>
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
    >>> model = FillModel()
    >>> order = OrderEvent(
    ...     timestamp=datetime.now(timezone.utc),
    ...     strategy_name="twap",
    ...     slice_index=0,
    ...     size=500.0,
    ...     side="buy",
    ... )
    >>> fill = model.execute(order, market)
    >>> print(f"Slippage: {fill.slippage_bps:.2f} bps")
"""

from __future__ import annotations

from datetime import datetime

from tributary.analytics.simulation.events import FillEvent, MarketEvent, OrderEvent


class FillModel:
    """Realistic fill model with liquidity consumption and recovery.

    Simulates order execution by:
    1. Adjusting orderbook by previously consumed liquidity
    2. Walking the adjusted orderbook to fill the order
    3. Tracking newly consumed liquidity per level
    4. Applying exponential recovery based on time elapsed

    This allows the simulation to capture:
    - Market impact: Large orders consume liquidity, causing worse fills
    - Recovery: Liquidity partially returns over time
    - Strategy differentiation: TWAP (patient) vs market order (aggressive)

    Attributes:
        recovery_rate: Maximum fraction of liquidity that recovers (0-1)
        half_life_ms: Time for half of recoverable liquidity to return
    """

    def __init__(
        self,
        recovery_rate: float = 0.5,
        half_life_ms: float = 1000.0,
    ):
        """Initialize the fill model.

        Args:
            recovery_rate: Maximum fraction of liquidity that recovers (0-1).
                Value of 0.5 means at most 50% of consumed liquidity returns.
            half_life_ms: Time in milliseconds for half of the recoverable
                liquidity to return. Default 1000ms (1 second).
        """
        self.recovery_rate = recovery_rate
        self.half_life_ms = half_life_ms

        # Track consumed liquidity per side per level index
        # Structure: {"buy": {0: amount, 1: amount, ...}, "sell": {...}}
        self._consumed: dict[str, dict[int, float]] = {"buy": {}, "sell": {}}
        self._last_execution_time: datetime | None = None

    def reset(self) -> None:
        """Reset consumed liquidity state for new simulation run.

        Call this when starting a new simulation or when assuming
        the orderbook has fully recovered (e.g., between TWAP slices
        with sufficient time gap).
        """
        self._consumed = {"buy": {}, "sell": {}}
        self._last_execution_time = None

    def _apply_recovery(self, current_time: datetime) -> None:
        """Apply liquidity recovery based on time elapsed.

        Uses exponential decay model:
            recovered = consumed * recovery_factor
            remaining_consumed = consumed - recovered

        Where recovery_factor depends on time elapsed and half_life.

        Args:
            current_time: Current simulation time
        """
        if self._last_execution_time is None:
            return

        # Calculate time elapsed in milliseconds
        elapsed_ms = (
            current_time - self._last_execution_time
        ).total_seconds() * 1000.0

        if elapsed_ms <= 0:
            return

        # Calculate recovery factor using exponential decay
        # At t=half_life, 50% of recoverable liquidity has returned
        # recovery_factor = how much of the recoverable portion has returned
        decay_factor = 0.5 ** (elapsed_ms / self.half_life_ms)
        recovery_factor = self.recovery_rate * (1 - decay_factor)

        # Apply recovery to both sides
        for side in ("buy", "sell"):
            for level in list(self._consumed[side].keys()):
                consumed = self._consumed[side][level]
                recovered = consumed * recovery_factor
                remaining = consumed - recovered

                if remaining < 1e-10:
                    # Fully recovered, remove from tracking
                    del self._consumed[side][level]
                else:
                    self._consumed[side][level] = remaining

    def _get_adjusted_sizes(
        self,
        original_sizes: tuple[float, ...],
        side: str,
    ) -> list[float]:
        """Get orderbook sizes adjusted for consumed liquidity.

        Args:
            original_sizes: Original sizes at each level
            side: Trade side ('buy' or 'sell')

        Returns:
            List of adjusted sizes (original - consumed)
        """
        adjusted = []
        for i, size in enumerate(original_sizes):
            consumed = self._consumed[side].get(i, 0.0)
            adjusted_size = max(0.0, size - consumed)
            adjusted.append(adjusted_size)
        return adjusted

    def execute(
        self,
        order: OrderEvent,
        market: MarketEvent,
    ) -> FillEvent:
        """Execute order against current market state.

        Simulates order execution with the following steps:
        1. Apply liquidity recovery if time has elapsed
        2. Adjust orderbook sizes by consumed amounts
        3. Walk the book to fill the order
        4. Track newly consumed liquidity
        5. Return FillEvent with execution details

        Args:
            order: The order to execute
            market: Current orderbook state

        Returns:
            FillEvent with execution results

        Note:
            For buy orders, walks the ask side (prices ascending).
            For sell orders, walks the bid side (prices descending).
        """
        # Step 1: Apply recovery based on time elapsed
        self._apply_recovery(order.timestamp)

        # Validate side
        side = order.side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {order.side}. Must be 'buy' or 'sell'")

        # Handle zero order size
        if order.size <= 0:
            return FillEvent(
                timestamp=order.timestamp,
                strategy_name=order.strategy_name,
                slice_index=order.slice_index,
                requested_size=order.size,
                filled_size=0.0,
                avg_price=float("nan"),
                slippage_bps=float("nan"),
                levels_consumed=0,
                mid_price_at_fill=market.mid_price,
            )

        # Select relevant side of book
        if side == "buy":
            prices = market.ask_prices
            original_sizes = market.ask_sizes
        else:
            prices = market.bid_prices
            original_sizes = market.bid_sizes

        # Handle empty orderbook
        if not prices or not original_sizes:
            return FillEvent(
                timestamp=order.timestamp,
                strategy_name=order.strategy_name,
                slice_index=order.slice_index,
                requested_size=order.size,
                filled_size=0.0,
                avg_price=float("nan"),
                slippage_bps=float("nan"),
                levels_consumed=0,
                mid_price_at_fill=float("nan"),
            )

        # Step 2: Get adjusted sizes (accounting for consumed liquidity)
        adjusted_sizes = self._get_adjusted_sizes(original_sizes, side)

        # Step 3: Walk the book
        remaining = order.size
        total_cost = 0.0
        total_filled = 0.0
        levels_consumed = 0

        for i, (price, available) in enumerate(zip(prices, adjusted_sizes)):
            if remaining <= 0:
                break
            if available <= 0:
                continue

            fill_size = min(remaining, available)
            total_cost += fill_size * price
            total_filled += fill_size
            remaining -= fill_size
            levels_consumed += 1

            # Step 4: Track consumed liquidity at this level
            if i not in self._consumed[side]:
                self._consumed[side][i] = 0.0
            self._consumed[side][i] += fill_size

        # Update last execution time
        self._last_execution_time = order.timestamp

        # Handle no fill (all liquidity consumed)
        if total_filled == 0:
            return FillEvent(
                timestamp=order.timestamp,
                strategy_name=order.strategy_name,
                slice_index=order.slice_index,
                requested_size=order.size,
                filled_size=0.0,
                avg_price=float("nan"),
                slippage_bps=float("nan"),
                levels_consumed=0,
                mid_price_at_fill=market.mid_price,
            )

        # Calculate average execution price
        avg_price = total_cost / total_filled

        # Calculate slippage in basis points
        # Convention: positive = cost (paid more for buys, received less for sells)
        if side == "buy":
            slippage_bps = (avg_price - market.mid_price) / market.mid_price * 10000
        else:
            slippage_bps = (market.mid_price - avg_price) / market.mid_price * 10000

        # Step 5: Return FillEvent
        return FillEvent(
            timestamp=order.timestamp,
            strategy_name=order.strategy_name,
            slice_index=order.slice_index,
            requested_size=order.size,
            filled_size=total_filled,
            avg_price=avg_price,
            slippage_bps=slippage_bps,
            levels_consumed=levels_consumed,
            mid_price_at_fill=market.mid_price,
        )
