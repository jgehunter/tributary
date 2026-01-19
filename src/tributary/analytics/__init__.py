"""Analytics module for querying and analyzing market data."""

from tributary.analytics.reader import QuestDBReader
from tributary.analytics.benchmarks import (
    calculate_vwap,
    calculate_cumulative_vwap,
    calculate_vwap_for_period,
    calculate_twap,
    calculate_twap_from_orderbooks,
    get_arrival_price,
)
from tributary.analytics.slippage import calculate_slippage_bps
from tributary.analytics.shortfall import (
    ShortfallComponents,
    decompose_implementation_shortfall,
)
from tributary.analytics.cost_forecast import (
    CostForecast,
    estimate_slippage_from_orderbook,
    forecast_execution_cost,
)
from tributary.analytics.impact import (
    ImpactEstimate,
    estimate_market_impact,
    CalibrationResult,
    calibrate_impact_parameters,
)
from tributary.analytics.optimization import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    calibrate_ac_params,
    generate_ac_trajectory,
    generate_twap_trajectory,
    generate_vwap_trajectory,
    generate_market_order_trajectory,
    get_volume_profile_from_db,
    ScheduleConstraints,
    TradeSchedule,
    optimize_schedule,
    calculate_optimal_intervals,
    StrategyComparison,
    compare_strategies,
    execution_profile_chart,
)
from tributary.analytics import simulation

__all__ = [
    # Reader
    "QuestDBReader",
    # Benchmarks
    "calculate_vwap",
    "calculate_cumulative_vwap",
    "calculate_vwap_for_period",
    "calculate_twap",
    "calculate_twap_from_orderbooks",
    "get_arrival_price",
    # Slippage & Shortfall
    "calculate_slippage_bps",
    "ShortfallComponents",
    "decompose_implementation_shortfall",
    # Cost Forecast
    "CostForecast",
    "estimate_slippage_from_orderbook",
    "forecast_execution_cost",
    # Market Impact
    "ImpactEstimate",
    "estimate_market_impact",
    "CalibrationResult",
    "calibrate_impact_parameters",
    # Optimization - Almgren-Chriss
    "AlmgrenChrissParams",
    "ExecutionTrajectory",
    "calibrate_ac_params",
    "generate_ac_trajectory",
    # Optimization - Baseline Strategies
    "generate_twap_trajectory",
    "generate_vwap_trajectory",
    "generate_market_order_trajectory",
    "get_volume_profile_from_db",
    # Optimization - Scheduler
    "ScheduleConstraints",
    "TradeSchedule",
    "optimize_schedule",
    "calculate_optimal_intervals",
    # Optimization - Comparison
    "StrategyComparison",
    "compare_strategies",
    "execution_profile_chart",
    # Simulation module
    "simulation",
]
