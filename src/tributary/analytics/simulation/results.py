"""Simulation result containers and comparison utilities.

Provides:
- SimulationResult: Complete result of simulating one strategy
- create_simulation_result: Create result from StrategyRun
- compare_simulation_results: Rank strategies by cost, risk, or risk-adjusted
- execution_chart_data: Generate visualization-ready DataFrame

These utilities enable proving that optimized strategies outperform naive
approaches, which is the core value proposition of the simulation engine.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .runner import StrategyRun
from .metrics import calculate_simulation_metrics


@dataclass(frozen=True)
class SimulationResult:
    """Complete result of simulating one strategy.

    Frozen for immutability. Provides all metrics needed for comparison.

    Attributes:
        strategy_name: Name of the strategy
        total_order_size: Original order size
        side: Trade direction ('buy' or 'sell')
        total_filled: Total size filled
        total_unfilled: Size not filled
        num_slices: Number of execution slices
        num_partial_fills: Slices with partial fills
        arrival_price: Mid-price at simulation start
        avg_execution_price: VWAP of execution
        implementation_shortfall_bps: Cost vs arrival price
        vwap_slippage_bps: Cost vs market VWAP
        total_cost_usd: Total cost in dollars
        cost_variance: Variance of per-slice slippage
        max_drawdown_bps: Worst cumulative cost during execution
        worst_slice_slippage_bps: Highest single-slice slippage
        fills: Tuple of FillEvent objects (tuple for frozen)
    """

    strategy_name: str
    total_order_size: float
    side: str

    # Execution summary
    total_filled: float
    total_unfilled: float
    num_slices: int
    num_partial_fills: int

    # Cost metrics
    arrival_price: float
    avg_execution_price: float
    implementation_shortfall_bps: float
    vwap_slippage_bps: float
    total_cost_usd: float

    # Risk metrics
    cost_variance: float
    max_drawdown_bps: float
    worst_slice_slippage_bps: float

    # Detailed data (tuple for frozen)
    fills: tuple

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled (0-100)."""
        if self.total_order_size == 0:
            return 0.0
        return self.total_filled / self.total_order_size * 100

    @property
    def risk_adjusted_score(self) -> float:
        """Risk-adjusted cost: IS / sqrt(variance). Lower is better.

        When variance is zero, returns raw IS (no adjustment).
        This metric balances cost against execution risk.
        """
        if self.cost_variance > 0:
            return self.implementation_shortfall_bps / np.sqrt(self.cost_variance)
        return self.implementation_shortfall_bps


def create_simulation_result(
    strategy_run: StrategyRun,
    arrival_price: float,
    market_vwap: float,
) -> SimulationResult:
    """
    Create SimulationResult from StrategyRun.

    Args:
        strategy_run: Result from StrategyRunner
        arrival_price: Mid-price at start of execution
        market_vwap: Market VWAP during execution period

    Returns:
        SimulationResult with all metrics
    """
    total_order_size = float(np.sum(strategy_run.trajectory.trade_sizes))

    metrics = calculate_simulation_metrics(
        fills=strategy_run.fills,
        arrival_price=arrival_price,
        total_order_size=total_order_size,
        side=strategy_run.side,
        market_vwap=market_vwap,
    )

    return SimulationResult(
        strategy_name=strategy_run.trajectory.strategy_name,
        total_order_size=total_order_size,
        side=strategy_run.side,
        total_filled=metrics["total_filled"],
        total_unfilled=metrics["total_unfilled"],
        num_slices=metrics["num_slices"],
        num_partial_fills=metrics["num_partial_fills"],
        arrival_price=arrival_price,
        avg_execution_price=metrics["avg_execution_price"],
        implementation_shortfall_bps=metrics["implementation_shortfall_bps"],
        vwap_slippage_bps=metrics["vwap_slippage_bps"],
        total_cost_usd=metrics["total_cost_usd"],
        cost_variance=metrics["cost_variance"],
        max_drawdown_bps=metrics["max_drawdown_bps"],
        worst_slice_slippage_bps=metrics["worst_slice_slippage_bps"],
        fills=tuple(strategy_run.fills),
    )


def compare_simulation_results(
    results: List[SimulationResult],
    rank_by: str = "risk_adjusted",
) -> pd.DataFrame:
    """
    Compare multiple strategy simulation results.

    Args:
        results: List of SimulationResult objects
        rank_by: 'cost' (IS only), 'risk' (variance), or 'risk_adjusted'

    Returns:
        DataFrame with comparison metrics, sorted by selected criterion (best first)
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append(
            {
                "strategy": r.strategy_name,
                "is_bps": r.implementation_shortfall_bps,
                "vwap_slip_bps": r.vwap_slippage_bps,
                "cost_variance": r.cost_variance,
                "max_drawdown_bps": r.max_drawdown_bps,
                "fill_rate_pct": r.fill_rate,
                "risk_adjusted_score": r.risk_adjusted_score,
                "total_cost_usd": r.total_cost_usd,
            }
        )

    df = pd.DataFrame(rows)

    # Sort by selected criterion (lower is better)
    sort_col = {
        "cost": "is_bps",
        "risk": "cost_variance",
        "risk_adjusted": "risk_adjusted_score",
    }.get(rank_by, "risk_adjusted_score")

    return df.sort_values(sort_col).reset_index(drop=True)


def execution_chart_data(
    results: List[SimulationResult],
) -> pd.DataFrame:
    """
    Generate long-format DataFrame for execution visualization.

    Returns DataFrame with columns:
    - timestamp: Execution time
    - strategy: Strategy name
    - holdings_pct: Remaining holdings as % of order (100 -> 0)
    - cumulative_cost_bps: Cost accumulated so far
    """
    rows = []

    for r in results:
        if not r.fills:
            continue

        remaining = r.total_order_size
        cumulative_cost = 0.0

        # Initial state
        first_fill = r.fills[0]
        rows.append(
            {
                "timestamp": first_fill.timestamp,
                "strategy": r.strategy_name,
                "holdings_pct": 100.0,
                "cumulative_cost_bps": 0.0,
            }
        )

        for fill in r.fills:
            remaining -= fill.filled_size
            # Weight slippage by fill proportion
            if r.total_order_size > 0:
                cumulative_cost += fill.slippage_bps * (
                    fill.filled_size / r.total_order_size
                )

            rows.append(
                {
                    "timestamp": fill.timestamp,
                    "strategy": r.strategy_name,
                    "holdings_pct": remaining / r.total_order_size * 100
                    if r.total_order_size > 0
                    else 0.0,
                    "cumulative_cost_bps": cumulative_cost,
                }
            )

    return pd.DataFrame(rows)
