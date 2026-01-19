"""Strategy comparison utilities for pre-simulation analysis.

Enables side-by-side comparison of execution strategies before
committing to simulation. Helps users understand trade-offs between
strategies without running full backtests.

Key insight: Before running expensive simulations, users should compare
expected costs, risk profiles, and execution patterns of different strategies.

Key Classes:
    StrategyComparison: Holds multiple trajectories for comparison

Key Functions:
    compare_strategies: Create comparison from multiple trajectories
    execution_profile_chart: Generate data for visualization

Requirements Satisfied:
    OPT-06: Strategy comparison and visualization data
"""

from dataclasses import dataclass, field
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd

from .almgren_chriss import ExecutionTrajectory


@dataclass
class StrategyComparison:
    """Comparison of multiple execution strategies.

    Attributes:
        strategies: List of ExecutionTrajectory objects to compare
        baseline_name: Name of baseline strategy for relative comparison
        order_size: Original order size (for consistency check)
    """

    strategies: List[ExecutionTrajectory]
    baseline_name: str = "twap"
    order_size: Optional[float] = None

    def summary_table(self) -> pd.DataFrame:
        """
        Generate comparison summary as DataFrame.

        Returns:
            DataFrame with columns:
            - strategy: Strategy name
            - risk_aversion: Lambda parameter
            - expected_cost_bps: Estimated cost (0 if not computed)
            - num_slices: Number of execution intervals
            - max_slice_pct: Largest slice as % of total
            - min_slice_pct: Smallest slice as % of total
            - front_loaded: Boolean, True if first slice > last slice

        Example:
            >>> comparison = compare_strategies(twap, vwap, ac)
            >>> print(comparison.summary_table())
               strategy  risk_aversion  expected_cost_bps  num_slices  ...
            0      twap            0.0                0.0          10  ...
            1      vwap            0.0                0.0          10  ...
            2  almgren_chriss   0.000001            25.3          10  ...
        """
        rows = []

        for traj in self.strategies:
            # Calculate order size from this trajectory if not NaN
            if not np.any(np.isnan(traj.trade_sizes)):
                traj_order_size = float(np.sum(traj.trade_sizes))
                max_slice = float(np.max(traj.trade_sizes))
                min_slice = float(np.min(traj.trade_sizes))
                num_slices = len(traj.trade_sizes)

                if traj_order_size > 0:
                    max_slice_pct = max_slice / traj_order_size * 100
                    min_slice_pct = min_slice / traj_order_size * 100
                else:
                    max_slice_pct = 0.0
                    min_slice_pct = 0.0

                # Check if front-loaded (first slice > last slice)
                if num_slices > 1:
                    front_loaded = traj.trade_sizes[0] > traj.trade_sizes[-1]
                else:
                    front_loaded = False
            else:
                num_slices = 0
                max_slice_pct = float("nan")
                min_slice_pct = float("nan")
                front_loaded = False

            row = {
                "strategy": traj.strategy_name,
                "risk_aversion": traj.risk_aversion,
                "expected_cost_bps": traj.total_cost_estimate,
                "num_slices": num_slices,
                "max_slice_pct": max_slice_pct,
                "min_slice_pct": min_slice_pct,
                "front_loaded": front_loaded,
            }
            rows.append(row)

        return pd.DataFrame(rows)


def compare_strategies(
    *trajectories: ExecutionTrajectory,
    baseline: str = "twap",
) -> StrategyComparison:
    """
    Create comparison from multiple trajectories.

    Args:
        *trajectories: ExecutionTrajectory objects to compare
        baseline: Name of baseline strategy for relative comparisons

    Returns:
        StrategyComparison with all strategies

    Raises:
        ValueError: If fewer than 2 strategies provided

    Example:
        >>> twap = generate_twap_trajectory(1000, 10)
        >>> vwap = generate_vwap_trajectory(1000, volume_profile)
        >>> ac = generate_ac_trajectory(1000, 10, params, 1e-5)
        >>> comparison = compare_strategies(twap, vwap, ac, baseline="twap")
        >>> print(comparison.summary_table())
    """
    if len(trajectories) < 2:
        raise ValueError("At least 2 strategies required for comparison")

    # Extract order size from first valid trajectory
    order_size = None
    for traj in trajectories:
        if not np.any(np.isnan(traj.trade_sizes)):
            order_size = float(np.sum(traj.trade_sizes))
            break

    # Check consistency of order sizes
    if order_size is not None:
        for traj in trajectories:
            if not np.any(np.isnan(traj.trade_sizes)):
                traj_size = float(np.sum(traj.trade_sizes))
                if abs(traj_size - order_size) > 1e-6:
                    warnings.warn(
                        f"Strategy '{traj.strategy_name}' has order size {traj_size:.2f} "
                        f"but expected {order_size:.2f}. Results may not be comparable.",
                        UserWarning,
                    )

    return StrategyComparison(
        strategies=list(trajectories),
        baseline_name=baseline,
        order_size=order_size,
    )


def execution_profile_chart(
    comparison: StrategyComparison,
) -> pd.DataFrame:
    """
    Generate data for execution profile visualization.

    Creates a "long format" DataFrame suitable for plotting with
    matplotlib, plotly, seaborn, or other visualization libraries.

    Returns DataFrame with columns:
    - period: Time period (0, 1, 2, ...)
    - strategy: Strategy name
    - holdings_pct: Remaining holdings as % of order size
    - trade_size_pct: Trade size as % of order size

    Use with plotting libraries (matplotlib, plotly) as desired.

    Example:
        >>> comparison = compare_strategies(twap, vwap, ac)
        >>> chart_data = execution_profile_chart(comparison)
        >>> # Plot with matplotlib
        >>> import matplotlib.pyplot as plt
        >>> for name, group in chart_data.groupby('strategy'):
        ...     plt.plot(group['period'], group['holdings_pct'], label=name)
        >>> plt.legend()
        >>> plt.show()
    """
    rows = []

    for traj in comparison.strategies:
        # Skip invalid trajectories
        if np.any(np.isnan(traj.holdings)):
            continue

        # Calculate order size for this trajectory
        traj_order_size = float(np.sum(traj.trade_sizes)) if not np.any(np.isnan(traj.trade_sizes)) else 1.0
        if traj_order_size == 0:
            traj_order_size = 1.0  # Avoid division by zero

        # Normalize holdings and trade sizes to percentages
        holdings_pct = traj.holdings / traj_order_size * 100
        trade_sizes_pct = traj.trade_sizes / traj_order_size * 100

        # Create rows for holdings at each timestamp
        for i, (ts, h) in enumerate(zip(traj.timestamps, holdings_pct)):
            row = {
                "period": int(ts),
                "strategy": traj.strategy_name,
                "holdings_pct": float(h),
                # Trade size is for the interval starting at this period
                # (except for the last timestamp which has no trade)
                "trade_size_pct": float(trade_sizes_pct[i]) if i < len(trade_sizes_pct) else 0.0,
            }
            rows.append(row)

    return pd.DataFrame(rows)
