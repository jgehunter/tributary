"""Optimization module for optimal execution strategies."""

from tributary.analytics.optimization.almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    calibrate_ac_params,
    generate_ac_trajectory,
)
from tributary.analytics.optimization.strategies import (
    generate_twap_trajectory,
    generate_vwap_trajectory,
    generate_market_order_trajectory,
    get_volume_profile_from_db,
)
from tributary.analytics.optimization.scheduler import (
    ScheduleConstraints,
    TradeSchedule,
    optimize_schedule,
    calculate_optimal_intervals,
)
from tributary.analytics.optimization.comparison import (
    StrategyComparison,
    compare_strategies,
    execution_profile_chart,
)

__all__ = [
    # Almgren-Chriss
    "AlmgrenChrissParams",
    "ExecutionTrajectory",
    "calibrate_ac_params",
    "generate_ac_trajectory",
    # Baseline strategies
    "generate_twap_trajectory",
    "generate_vwap_trajectory",
    "generate_market_order_trajectory",
    "get_volume_profile_from_db",
    # Scheduler
    "ScheduleConstraints",
    "TradeSchedule",
    "optimize_schedule",
    "calculate_optimal_intervals",
    # Comparison
    "StrategyComparison",
    "compare_strategies",
    "execution_profile_chart",
]
