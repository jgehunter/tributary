"""Trade scheduling optimizer for optimal execution.

Provides constraint-aware schedule optimization that wraps Almgren-Chriss
trajectory generation with practical execution constraints like participation
rate limits and minimum slice sizes.

Key insight: The optimal number of execution intervals is often determined by
participation constraints (don't be more than X% of volume) rather than
pure cost-risk optimization.

Key Functions:
    optimize_schedule: Generate optimal schedule respecting constraints
    calculate_optimal_intervals: Determine minimum intervals from participation constraint

Requirements Satisfied:
    OPT-06: Optimal slice timing and sizing with constraint handling
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    generate_ac_trajectory,
)


@dataclass(frozen=True)
class ScheduleConstraints:
    """Constraints for trade schedule optimization.

    Attributes:
        max_participation_rate: Maximum fraction of interval volume (default 0.10)
        min_slice_size: Minimum trade size per interval (default 0)
        max_slice_size: Maximum trade size per interval (default inf)
        min_intervals: Minimum number of execution intervals (default 1)
        max_intervals: Maximum number of execution intervals (default None = unlimited)
    """

    max_participation_rate: float = 0.10
    min_slice_size: float = 0.0
    max_slice_size: float = float("inf")
    min_intervals: int = 1
    max_intervals: Optional[int] = None


@dataclass(frozen=True)
class TradeSchedule:
    """Optimized trade schedule with constraint metadata.

    Attributes:
        trajectory: The ExecutionTrajectory with optimal parameters
        constraints: Constraints that were applied
        intervals_used: Actual number of intervals in schedule
        max_slice_pct: Largest slice as percentage of total
        meets_participation_constraint: Whether all slices respect participation limit
        warnings: Any constraint violations or adjustments made
    """

    trajectory: ExecutionTrajectory
    constraints: ScheduleConstraints
    intervals_used: int
    max_slice_pct: float
    meets_participation_constraint: bool
    warnings: tuple = field(default_factory=tuple)


def calculate_optimal_intervals(
    order_size: float,
    expected_interval_volume: float,
    max_participation_rate: float = 0.10,
) -> int:
    """
    Calculate minimum intervals needed to respect participation constraint.

    The participation constraint limits how much of each interval's volume
    we can trade. This sets a floor on the number of intervals needed.

    Args:
        order_size: Total size to execute
        expected_interval_volume: Expected market volume per interval
        max_participation_rate: Maximum fraction of volume per interval

    Returns:
        Minimum number of intervals needed (at least 1)

    Example:
        >>> # 10,000 shares, 100,000 volume/interval, 10% max participation
        >>> calculate_optimal_intervals(10000, 100000, 0.10)
        1  # Can trade 10K in one interval (10K = 10% of 100K)

        >>> # 50,000 shares, 100,000 volume/interval, 10% max participation
        >>> calculate_optimal_intervals(50000, 100000, 0.10)
        5  # Need 5 intervals (10K per interval max)
    """
    if order_size <= 0:
        return 1

    if expected_interval_volume <= 0:
        return 1

    if max_participation_rate <= 0:
        # Can't execute anything, but return 1 as minimum
        return 1

    # Max per-interval execution based on participation constraint
    max_per_interval = expected_interval_volume * max_participation_rate

    # Minimum intervals needed to execute order within participation limit
    min_intervals = math.ceil(order_size / max_per_interval)

    return max(min_intervals, 1)


def optimize_schedule(
    order_size: float,
    params: AlmgrenChrissParams,
    expected_interval_volume: float,
    risk_aversion: float = 1e-6,
    constraints: Optional[ScheduleConstraints] = None,
) -> TradeSchedule:
    """
    Optimize trade schedule given order and market parameters.

    Determines optimal number of intervals and generates trajectory
    while respecting participation rate and slice size constraints.

    Algorithm:
        1. Calculate minimum intervals from participation constraint
        2. Apply min/max interval constraints
        3. Generate A-C trajectory with final interval count
        4. Verify slices respect size constraints
        5. Return schedule with warnings if constraints violated

    Args:
        order_size: Total size to execute
        params: Calibrated AlmgrenChrissParams
        expected_interval_volume: Expected market volume per interval
        risk_aversion: Lambda for A-C optimization
        constraints: Optional ScheduleConstraints (uses defaults if None)

    Returns:
        TradeSchedule with optimized trajectory and constraint metadata

    Example:
        >>> params = calibrate_ac_params(50000, 0.02, 0.05, 0.50)
        >>> schedule = optimize_schedule(
        ...     order_size=5000,
        ...     params=params,
        ...     expected_interval_volume=5000,  # 5000 shares per interval
        ...     risk_aversion=1e-5,
        ... )
        >>> print(f"Intervals: {schedule.intervals_used}")
        >>> print(f"Max slice: {schedule.max_slice_pct:.1f}%")
    """
    # Use default constraints if not provided
    if constraints is None:
        constraints = ScheduleConstraints()

    warnings: List[str] = []

    # Validate inputs
    if order_size <= 0:
        error_trajectory = ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="scheduler",
            total_cost_estimate=float("nan"),
            risk_aversion=risk_aversion,
            params={"error": "Invalid order size (<= 0)"},
        )
        return TradeSchedule(
            trajectory=error_trajectory,
            constraints=constraints,
            intervals_used=0,
            max_slice_pct=float("nan"),
            meets_participation_constraint=False,
            warnings=("Invalid order size: must be > 0",),
        )

    if expected_interval_volume <= 0:
        error_trajectory = ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="scheduler",
            total_cost_estimate=float("nan"),
            risk_aversion=risk_aversion,
            params={"error": "Invalid expected_interval_volume (<= 0)"},
        )
        return TradeSchedule(
            trajectory=error_trajectory,
            constraints=constraints,
            intervals_used=0,
            max_slice_pct=float("nan"),
            meets_participation_constraint=False,
            warnings=("Invalid expected_interval_volume: must be > 0",),
        )

    # Step 1: Calculate minimum intervals from participation constraint
    min_intervals_participation = calculate_optimal_intervals(
        order_size,
        expected_interval_volume,
        constraints.max_participation_rate,
    )

    # Step 2: Apply interval constraints
    intervals = max(min_intervals_participation, constraints.min_intervals)

    if constraints.max_intervals is not None:
        if intervals > constraints.max_intervals:
            warnings.append(
                f"Participation constraint requires {intervals} intervals "
                f"but max_intervals is {constraints.max_intervals}. "
                f"Using max_intervals, participation constraint may be violated."
            )
        intervals = min(intervals, constraints.max_intervals)

    # Step 3: Generate A-C trajectory with final interval count
    trajectory = generate_ac_trajectory(
        order_size=order_size,
        duration_periods=intervals,
        params=params,
        risk_aversion=risk_aversion,
    )

    # Step 4: Calculate metrics
    if np.any(np.isnan(trajectory.trade_sizes)):
        max_slice_pct = float("nan")
        max_slice_rate = float("nan")
        meets_participation = False
    else:
        max_slice = float(np.max(trajectory.trade_sizes))
        max_slice_pct = max_slice / order_size * 100
        max_slice_rate = max_slice / expected_interval_volume
        meets_participation = max_slice_rate <= constraints.max_participation_rate

    # Step 5: Check slice size constraints
    if not np.any(np.isnan(trajectory.trade_sizes)):
        min_slice = float(np.min(trajectory.trade_sizes))
        max_slice = float(np.max(trajectory.trade_sizes))

        if min_slice < constraints.min_slice_size:
            warnings.append(
                f"Minimum slice ({min_slice:.2f}) is below min_slice_size constraint "
                f"({constraints.min_slice_size:.2f})."
            )

        if max_slice > constraints.max_slice_size:
            warnings.append(
                f"Maximum slice ({max_slice:.2f}) exceeds max_slice_size constraint "
                f"({constraints.max_slice_size:.2f})."
            )

        if not meets_participation:
            warnings.append(
                f"Maximum slice ({max_slice:.2f}) exceeds participation limit "
                f"({constraints.max_participation_rate * 100:.1f}% of {expected_interval_volume:.2f} = "
                f"{constraints.max_participation_rate * expected_interval_volume:.2f})."
            )

    return TradeSchedule(
        trajectory=trajectory,
        constraints=constraints,
        intervals_used=intervals,
        max_slice_pct=max_slice_pct,
        meets_participation_constraint=meets_participation,
        warnings=tuple(warnings),
    )
