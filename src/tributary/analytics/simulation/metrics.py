"""Metrics calculation for simulation results.

Calculates comprehensive execution metrics from fill events:
- Implementation shortfall (cost vs arrival price)
- VWAP slippage (cost vs market VWAP)
- Risk metrics (variance, max drawdown, worst slice)
- Execution quality metrics (fill rate, partial fills)

All costs follow the sign convention:
    - Positive = cost (unfavorable execution)
    - Negative = gain (favorable execution)
"""

from typing import Dict

import numpy as np

from .events import FillEvent


def calculate_simulation_metrics(
    fills: list[FillEvent],
    arrival_price: float,
    total_order_size: float,
    side: str,
    market_vwap: float,
) -> Dict[str, float]:
    """
    Calculate comprehensive execution metrics from simulation fills.

    Args:
        fills: List of FillEvent objects from simulation
        arrival_price: Mid-price at simulation start (benchmark)
        total_order_size: Original order size
        side: 'buy' or 'sell'
        market_vwap: Market VWAP during execution period (for VWAP slippage)

    Returns:
        Dict with metrics:
        - implementation_shortfall_bps: Cost vs arrival price
        - vwap_slippage_bps: Cost vs market VWAP
        - total_filled: Total size filled
        - total_unfilled: Size not filled
        - num_slices: Number of execution slices
        - num_partial_fills: Slices with partial fills
        - avg_execution_price: VWAP of execution
        - cost_variance: Variance of per-slice slippage
        - max_drawdown_bps: Worst cumulative cost during execution
        - worst_slice_slippage_bps: Highest single-slice slippage
        - total_cost_usd: Total cost in dollars

    Sign convention: Positive = cost (unfavorable)
    """
    # Handle empty fills
    if not fills:
        return {
            "implementation_shortfall_bps": float("nan"),
            "vwap_slippage_bps": float("nan"),
            "total_filled": 0.0,
            "total_unfilled": total_order_size,
            "num_slices": 0,
            "num_partial_fills": 0,
            "avg_execution_price": float("nan"),
            "cost_variance": float("nan"),
            "max_drawdown_bps": float("nan"),
            "worst_slice_slippage_bps": float("nan"),
            "total_cost_usd": 0.0,
        }

    # Extract data from fills
    prices = [f.avg_price for f in fills]
    sizes = [f.filled_size for f in fills]
    slippages = [f.slippage_bps for f in fills]

    # Calculate totals
    total_filled = sum(sizes)
    total_unfilled = total_order_size - total_filled
    num_slices = len(fills)
    num_partial_fills = sum(1 for f in fills if f.filled_size < f.requested_size)

    # Calculate VWAP of execution
    if total_filled > 0:
        total_value = sum(p * s for p, s in zip(prices, sizes))
        avg_execution_price = total_value / total_filled
    else:
        avg_execution_price = float("nan")

    # Implementation shortfall vs arrival price
    if side == "buy":
        is_bps = (avg_execution_price - arrival_price) / arrival_price * 10000
        vwap_slip_bps = (avg_execution_price - market_vwap) / market_vwap * 10000
    else:  # sell
        is_bps = (arrival_price - avg_execution_price) / arrival_price * 10000
        vwap_slip_bps = (market_vwap - avg_execution_price) / market_vwap * 10000

    # Risk metrics
    cost_variance = float(np.var(slippages)) if len(slippages) > 1 else 0.0
    worst_slice_slippage_bps = float(max(slippages)) if slippages else float("nan")

    # Max drawdown: worst cumulative cost during execution
    # Weight slippages by filled size proportion
    if total_order_size > 0:
        weighted_slippages = [
            s * (sz / total_order_size) for s, sz in zip(slippages, sizes)
        ]
        cumulative_costs = np.cumsum(weighted_slippages)
        max_drawdown_bps = (
            float(np.max(cumulative_costs)) if len(cumulative_costs) > 0 else 0.0
        )
    else:
        max_drawdown_bps = 0.0

    # Total cost in USD
    if total_filled > 0 and not np.isnan(avg_execution_price):
        if side == "buy":
            total_cost_usd = total_filled * (avg_execution_price - arrival_price)
        else:
            total_cost_usd = total_filled * (arrival_price - avg_execution_price)
    else:
        total_cost_usd = 0.0

    return {
        "implementation_shortfall_bps": is_bps if not np.isnan(is_bps) else float("nan"),
        "vwap_slippage_bps": vwap_slip_bps if not np.isnan(vwap_slip_bps) else float("nan"),
        "total_filled": total_filled,
        "total_unfilled": total_unfilled,
        "num_slices": num_slices,
        "num_partial_fills": num_partial_fills,
        "avg_execution_price": avg_execution_price,
        "cost_variance": cost_variance,
        "max_drawdown_bps": max_drawdown_bps,
        "worst_slice_slippage_bps": worst_slice_slippage_bps,
        "total_cost_usd": total_cost_usd,
    }
