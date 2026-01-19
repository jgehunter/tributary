"""Baseline execution strategies: TWAP, VWAP, and market order.

These strategies provide comparison baselines for Almgren-Chriss optimization.
Unlike benchmarks.py (which calculates achieved prices from historical trades),
this module generates FUTURE execution schedules.

Key distinction:
- Benchmarks (Phase 2): What price DID we achieve? (backward-looking)
- Strategies (Phase 3): What trades SHOULD we execute? (forward-looking)

TWAP: Time-Weighted Average Price
    Divides order evenly across time periods. Simplest baseline.
    Minimizes variance of execution price but ignores market impact.

VWAP: Volume-Weighted Average Price
    Weights execution by expected volume at each interval.
    Attempts to match natural volume pattern to reduce market impact.

Market Order: Immediate execution
    Execute everything at once. Maximum market impact but zero timing risk.
    Represents the "naive" baseline for comparison.

All strategies return ExecutionTrajectory for consistent comparison with
Almgren-Chriss optimal trajectories.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from tributary.analytics.optimization.almgren_chriss import ExecutionTrajectory
from tributary.analytics.reader import QuestDBReader


def generate_twap_trajectory(
    order_size: float,
    duration_periods: int,
    randomize: bool = False,
    random_pct: float = 0.1,
    seed: Optional[int] = None,
) -> ExecutionTrajectory:
    """
    Generate Time-Weighted Average Price execution trajectory.

    Divides order evenly across time periods. Optional randomization
    adds variation to slice sizes to avoid detection/gaming.

    Args:
        order_size: Total size to execute
        duration_periods: Number of execution intervals
        randomize: If True, add random variation to slice sizes
        random_pct: Maximum random variation as fraction (default 10%)
        seed: Optional random seed for reproducibility

    Returns:
        ExecutionTrajectory with strategy_name='twap'

    Note:
        - risk_aversion = 0 (risk-neutral strategy)
        - total_cost_estimate = 0 (no impact model, just schedule)

    Example:
        >>> traj = generate_twap_trajectory(order_size=1000, duration_periods=10)
        >>> print(traj.trade_sizes)  # All equal: [100, 100, ..., 100]

        >>> traj = generate_twap_trajectory(
        ...     order_size=1000, duration_periods=10, randomize=True, seed=42
        ... )
        >>> print(traj.trade_sizes)  # Varies around 100, sums to 1000
    """
    # Validate inputs
    if order_size <= 0:
        return ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="twap",
            total_cost_estimate=0.0,
            risk_aversion=0.0,
            params={"error": "Invalid order size (<= 0)"},
        )

    if duration_periods <= 0:
        return ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="twap",
            total_cost_estimate=0.0,
            risk_aversion=0.0,
            params={"error": "Invalid duration (<= 0)"},
        )

    # Calculate base slice size
    base_slice = order_size / duration_periods

    if randomize:
        # Add random variation to slice sizes
        rng = np.random.default_rng(seed)
        variations = rng.uniform(1 - random_pct, 1 + random_pct, duration_periods)
        trade_sizes = base_slice * variations

        # Adjust last slice to ensure exact total
        trade_sizes[-1] = order_size - trade_sizes[:-1].sum()
    else:
        # Equal slices
        trade_sizes = np.full(duration_periods, base_slice)

    # Calculate holdings from trade sizes
    # holdings[0] = order_size
    # holdings[i+1] = holdings[i] - trade_sizes[i]
    holdings = np.zeros(duration_periods + 1)
    holdings[0] = order_size
    for i in range(duration_periods):
        holdings[i + 1] = holdings[i] - trade_sizes[i]

    # Timestamps are just period indices
    timestamps = np.arange(duration_periods + 1, dtype=float)

    return ExecutionTrajectory(
        timestamps=timestamps,
        holdings=holdings,
        trade_sizes=trade_sizes,
        strategy_name="twap",
        total_cost_estimate=0.0,  # Schedule only, no cost model
        risk_aversion=0.0,
        params={"randomize": randomize, "random_pct": random_pct if randomize else None},
    )


def generate_market_order_trajectory(
    order_size: float,
) -> ExecutionTrajectory:
    """
    Generate market order trajectory (single immediate execution).

    Represents the "naive" baseline: execute everything immediately,
    accepting maximum market impact.

    Args:
        order_size: Total size to execute

    Returns:
        ExecutionTrajectory with strategy_name='market_order'

    Note:
        - Single period execution (duration=1)
        - risk_aversion = float('inf') (infinitely risk-averse about timing)
        - total_cost_estimate = 0 (actual cost depends on orderbook)

    Example:
        >>> traj = generate_market_order_trajectory(order_size=1000)
        >>> print(traj.trade_sizes)  # [1000]
        >>> print(traj.holdings)     # [1000, 0]
    """
    # Validate inputs
    if order_size <= 0:
        return ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="market_order",
            total_cost_estimate=0.0,
            risk_aversion=float("inf"),
            params={"error": "Invalid order size (<= 0)"},
        )

    # Single-slice immediate execution
    timestamps = np.array([0.0, 1.0])
    holdings = np.array([order_size, 0.0])
    trade_sizes = np.array([order_size])

    return ExecutionTrajectory(
        timestamps=timestamps,
        holdings=holdings,
        trade_sizes=trade_sizes,
        strategy_name="market_order",
        total_cost_estimate=0.0,
        risk_aversion=float("inf"),
    )


def generate_vwap_trajectory(
    order_size: float,
    volume_profile: np.ndarray,
) -> ExecutionTrajectory:
    """
    Generate Volume-Weighted Average Price execution trajectory.

    Weights execution by expected volume at each interval. Attempts to
    match the natural volume pattern to minimize market impact.

    Args:
        order_size: Total size to execute
        volume_profile: Expected volume at each interval (will be normalized)
                       Length determines number of execution periods

    Returns:
        ExecutionTrajectory with strategy_name='vwap'

    Note:
        - Falls back to TWAP if volume_profile sums to 0
        - risk_aversion = 0 (risk-neutral strategy)

    Example:
        >>> volume_profile = np.array([100, 200, 300, 200, 100])  # Bell curve
        >>> traj = generate_vwap_trajectory(order_size=1000, volume_profile=volume_profile)
        >>> print(traj.trade_sizes)  # Peak in middle: [111, 222, 333, 222, 111]
    """
    # Validate inputs
    if order_size <= 0:
        return ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="vwap",
            total_cost_estimate=0.0,
            risk_aversion=0.0,
            params={"error": "Invalid order size (<= 0)"},
        )

    if len(volume_profile) == 0:
        return ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="vwap",
            total_cost_estimate=0.0,
            risk_aversion=0.0,
            params={"error": "Empty volume profile"},
        )

    # Normalize volume profile to weights
    total_volume = float(np.sum(volume_profile))

    if total_volume == 0:
        # Fall back to TWAP (uniform weights)
        duration_periods = len(volume_profile)
        weights = np.full(duration_periods, 1.0 / duration_periods)
        fallback = True
    else:
        weights = volume_profile / total_volume
        fallback = False

    # Allocate order by weights
    trade_sizes = order_size * weights

    # Calculate holdings
    duration_periods = len(volume_profile)
    holdings = np.zeros(duration_periods + 1)
    holdings[0] = order_size
    for i in range(duration_periods):
        holdings[i + 1] = holdings[i] - trade_sizes[i]

    # Timestamps
    timestamps = np.arange(duration_periods + 1, dtype=float)

    return ExecutionTrajectory(
        timestamps=timestamps,
        holdings=holdings,
        trade_sizes=trade_sizes,
        strategy_name="vwap",
        total_cost_estimate=0.0,
        risk_aversion=0.0,
        params={
            "volume_profile_sum": total_volume,
            "fallback_to_twap": fallback,
        },
    )


def get_volume_profile_from_db(
    reader: QuestDBReader,
    market_id: str,
    start_time: datetime,
    end_time: datetime,
    interval: str = "1h",
) -> np.ndarray:
    """
    Query historical volume profile from QuestDB.

    Uses query_vwap_sampled() to get time-bucketed volumes.

    Args:
        reader: Connected QuestDBReader
        market_id: Market identifier
        start_time: Start of historical period
        end_time: End of historical period
        interval: Time bucket interval (e.g., "1h", "15m")

    Returns:
        numpy array of volumes per interval

    Note:
        Returns empty array if no data found.

    Example:
        >>> reader = QuestDBReader(config)
        >>> reader.connect()
        >>> profile = get_volume_profile_from_db(
        ...     reader, "market-123",
        ...     datetime(2024, 1, 1), datetime(2024, 1, 2),
        ...     interval="1h"
        ... )
        >>> traj = generate_vwap_trajectory(1000, profile)
    """
    df = reader.query_vwap_sampled(
        market_id=market_id,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )

    if df.empty:
        return np.array([])

    # Extract volume column as numpy array
    return df["volume"].to_numpy()
