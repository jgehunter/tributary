"""Multi-strategy runner for comparing execution strategies.

This module provides the StrategyRunner class that executes multiple
strategies on the same historical market data with isolated execution.

Key principle: Each strategy gets a fresh FillModel instance, ensuring
no cross-strategy impact. This enables fair comparison of strategies
that would otherwise interfere with each other's liquidity consumption.

Example:
    >>> from datetime import datetime, timedelta, timezone
    >>> import pandas as pd
    >>> from tributary.analytics.simulation import StrategyRunner
    >>> from tributary.analytics.optimization import (
    ...     generate_twap_trajectory,
    ...     generate_market_order_trajectory,
    ... )
    >>>
    >>> # Create strategies
    >>> twap = generate_twap_trajectory(order_size=1000, duration_periods=5)
    >>> market = generate_market_order_trajectory(order_size=1000)
    >>>
    >>> # Run comparison
    >>> runner = StrategyRunner()
    >>> results = runner.run_strategies(
    ...     strategies=[twap, market],
    ...     market_data=market_data,  # DataFrame with orderbook snapshots
    ...     side='buy',
    ...     start_time=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    ...     interval=timedelta(seconds=1),
    ... )
    >>>
    >>> # Compare results
    >>> for result in results:
    ...     total_slippage = sum(f.slippage_bps for f in result.fills)
    ...     print(f"{result.trajectory.strategy_name}: {total_slippage:.2f} bps")
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from tributary.analytics.optimization import ExecutionTrajectory
from tributary.analytics.simulation.engine import SimulationEngine
from tributary.analytics.simulation.events import FillEvent
from tributary.analytics.simulation.fill_model import FillModel


@dataclass
class StrategyRun:
    """Result of simulating one strategy.

    Captures the complete execution results for a single strategy,
    including the original trajectory and all fill events.

    Attributes:
        trajectory: Original execution trajectory that was simulated
        fills: List of FillEvents from simulation (one per executed slice)
        side: Trade direction used ('buy' or 'sell')
        start_time: Simulation start time (arrival time)
        interval: Time between execution slices
    """

    trajectory: ExecutionTrajectory
    fills: list[FillEvent]
    side: str
    start_time: datetime
    interval: timedelta

    @property
    def total_filled(self) -> float:
        """Total size filled across all slices."""
        return sum(f.filled_size for f in self.fills)

    @property
    def total_requested(self) -> float:
        """Total size requested across all slices."""
        return sum(f.requested_size for f in self.fills)

    @property
    def fill_rate(self) -> float:
        """Percentage of order that was filled."""
        if self.total_requested == 0:
            return 0.0
        return self.total_filled / self.total_requested * 100.0

    @property
    def weighted_slippage_bps(self) -> float:
        """Size-weighted average slippage in basis points.

        Returns NaN if no fills occurred.
        """
        if not self.fills or self.total_filled == 0:
            return float("nan")

        weighted_sum = sum(
            f.slippage_bps * f.filled_size
            for f in self.fills
            if f.filled_size > 0
        )
        return weighted_sum / self.total_filled


class StrategyRunner:
    """Run multiple strategies on the same historical data.

    Each strategy executes with:
    - Fresh fill model (isolated execution, no cross-strategy impact)
    - Same market data (fair comparison)
    - Same parameters (start time, interval, side)

    This enables apples-to-apples comparison of execution strategies,
    where each strategy sees the same initial orderbook state.

    Attributes:
        recovery_rate: Liquidity recovery rate for fill models
        half_life_ms: Recovery half-life for fill models
    """

    def __init__(
        self,
        recovery_rate: float = 0.5,
        half_life_ms: float = 1000.0,
    ):
        """Initialize the strategy runner.

        Args:
            recovery_rate: Maximum fraction of liquidity that recovers (0-1).
                Applied to each strategy's FillModel.
            half_life_ms: Time in milliseconds for half of recoverable
                liquidity to return. Applied to each strategy's FillModel.
        """
        self.recovery_rate = recovery_rate
        self.half_life_ms = half_life_ms

    def run_strategies(
        self,
        strategies: List[ExecutionTrajectory],
        market_data: pd.DataFrame,
        side: str,
        start_time: datetime,
        interval: timedelta,
    ) -> List[StrategyRun]:
        """Run all strategies on the same market data.

        Each strategy gets a fresh fill model - strategies do not impact
        each other's execution. This provides fair comparison where each
        strategy sees the same initial market state.

        Args:
            strategies: List of ExecutionTrajectory objects to simulate
            market_data: Historical orderbook data (same for all strategies)
                Required columns: timestamp, mid_price, bid_prices, bid_sizes,
                ask_prices, ask_sizes
            side: Trade direction for all strategies ('buy' or 'sell')
            start_time: Execution start time (arrival time)
            interval: Time between execution slices

        Returns:
            List of StrategyRun objects, one per strategy, in same order
            as input strategies

        Note:
            - Empty strategies list returns empty results
            - Each StrategyRun contains all fill events for that strategy
            - Results order matches input order for easy pairing
        """
        results: List[StrategyRun] = []

        for strategy in strategies:
            # Fresh fill model for each strategy (isolated execution)
            fill_model = FillModel(
                recovery_rate=self.recovery_rate,
                half_life_ms=self.half_life_ms,
            )
            engine = SimulationEngine(fill_model=fill_model)

            fills = engine.run(
                trajectory=strategy,
                market_data=market_data,
                side=side,
                start_time=start_time,
                interval=interval,
            )

            results.append(StrategyRun(
                trajectory=strategy,
                fills=fills,
                side=side,
                start_time=start_time,
                interval=interval,
            ))

        return results

    def run_single(
        self,
        strategy: ExecutionTrajectory,
        market_data: pd.DataFrame,
        side: str,
        start_time: datetime,
        interval: timedelta,
    ) -> StrategyRun:
        """Convenience method to run a single strategy.

        Args:
            strategy: ExecutionTrajectory to simulate
            market_data: Historical orderbook data
            side: Trade direction ('buy' or 'sell')
            start_time: Execution start time
            interval: Time between execution slices

        Returns:
            StrategyRun with simulation results
        """
        results = self.run_strategies(
            strategies=[strategy],
            market_data=market_data,
            side=side,
            start_time=start_time,
            interval=interval,
        )
        return results[0]
