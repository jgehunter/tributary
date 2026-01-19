"""Implementation shortfall decomposition using the Perold framework.

Provides functions to decompose execution costs into delay, trading impact,
spread, and opportunity cost components.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ShortfallComponents:
    """Implementation shortfall decomposition result.

    All cost components follow the sign convention:
        - Positive = cost (unfavorable)
        - Negative = gain (favorable)

    Attributes:
        delay_cost_bps: Cost from price movement between decision and order entry
        trading_cost_bps: Cost from market impact during execution
        spread_cost_bps: Cost from bid-ask spread crossing (half-spread)
        opportunity_cost_bps: Cost from unfilled portion
        total_bps: Sum of all components
        delay_cost_usd: Delay cost in dollar terms
        trading_cost_usd: Trading cost in dollar terms
        spread_cost_usd: Spread cost in dollar terms
        opportunity_cost_usd: Opportunity cost in dollar terms
        total_usd: Total cost in dollar terms
    """

    delay_cost_bps: float
    trading_cost_bps: float
    spread_cost_bps: float
    opportunity_cost_bps: float
    total_bps: float
    delay_cost_usd: float
    trading_cost_usd: float
    spread_cost_usd: float
    opportunity_cost_usd: float
    total_usd: float


def decompose_implementation_shortfall(
    decision_price: float,
    order_entry_price: float,
    execution_prices: list[float],
    execution_sizes: list[float],
    total_order_size: float,
    closing_price: float,
    side: str,
    spread_at_entry: Optional[float] = None,
) -> ShortfallComponents:
    """
    Decompose implementation shortfall into cost components.

    Based on Perold (1988) and Wagner & Edwards (1993) framework.

    Implementation Shortfall = Delay Cost + Trading Cost + Spread Cost + Opportunity Cost

    Components:
        - Delay Cost: Price movement from decision time to order entry
        - Trading Cost: Price movement from order entry to execution (market impact)
        - Spread Cost: Half the bid-ask spread crossed when executing
        - Opportunity Cost: Cost of unfilled portion (price moved away)

    Sign convention:
        - Positive values = cost (unfavorable execution)
        - Negative values = gain (favorable execution)
        - Consistent for both buy and sell sides

    Args:
        decision_price: Mid-price when trading decision was made
        order_entry_price: Mid-price when order was submitted
        execution_prices: List of fill prices
        execution_sizes: List of fill sizes (must match execution_prices length)
        total_order_size: Originally intended order size
        closing_price: Price at end of evaluation period
        side: Trade direction ('buy' or 'sell')
        spread_at_entry: Bid-ask spread at order entry time (optional)

    Returns:
        ShortfallComponents with all cost components in bps and USD

    Raises:
        ValueError: If side is not 'buy' or 'sell'
        ValueError: If execution_prices and execution_sizes have different lengths

    Examples:
        >>> # Full execution buy order
        >>> result = decompose_implementation_shortfall(
        ...     decision_price=100.0,
        ...     order_entry_price=101.0,
        ...     execution_prices=[102.0],
        ...     execution_sizes=[1000.0],
        ...     total_order_size=1000.0,
        ...     closing_price=103.0,
        ...     side='buy',
        ...     spread_at_entry=0.20,
        ... )
        >>> result.delay_cost_bps  # Price moved up 1% before entry
        100.0
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

    if len(execution_prices) != len(execution_sizes):
        raise ValueError(
            f"execution_prices length ({len(execution_prices)}) must match "
            f"execution_sizes length ({len(execution_sizes)})"
        )

    executed_size = sum(execution_sizes)
    unfilled_size = total_order_size - executed_size

    # Calculate weighted average execution price
    if executed_size > 0:
        avg_execution_price = (
            sum(p * s for p, s in zip(execution_prices, execution_sizes))
            / executed_size
        )
    else:
        avg_execution_price = decision_price

    # Delay cost: movement between decision and order entry
    delay_cost_usd = executed_size * (order_entry_price - decision_price)

    # Trading cost: movement from order entry to execution
    trading_cost_usd = executed_size * (avg_execution_price - order_entry_price)

    # Spread cost: half-spread crossed
    if spread_at_entry is not None:
        spread_cost_usd = executed_size * (spread_at_entry / 2)
    else:
        spread_cost_usd = 0.0

    # Opportunity cost: unfilled portion
    opportunity_cost_usd = unfilled_size * (closing_price - decision_price)

    # Flip signs for sells (receiving less than expected is a cost)
    if side == "sell":
        delay_cost_usd = -delay_cost_usd
        trading_cost_usd = -trading_cost_usd
        opportunity_cost_usd = -opportunity_cost_usd

    total_usd = delay_cost_usd + trading_cost_usd + spread_cost_usd + opportunity_cost_usd

    # Convert to basis points
    notional = total_order_size * decision_price

    if notional == 0:
        return ShortfallComponents(
            delay_cost_bps=0.0,
            trading_cost_bps=0.0,
            spread_cost_bps=0.0,
            opportunity_cost_bps=0.0,
            total_bps=0.0,
            delay_cost_usd=delay_cost_usd,
            trading_cost_usd=trading_cost_usd,
            spread_cost_usd=spread_cost_usd,
            opportunity_cost_usd=opportunity_cost_usd,
            total_usd=total_usd,
        )

    return ShortfallComponents(
        delay_cost_bps=delay_cost_usd / notional * 10000,
        trading_cost_bps=trading_cost_usd / notional * 10000,
        spread_cost_bps=spread_cost_usd / notional * 10000,
        opportunity_cost_bps=opportunity_cost_usd / notional * 10000,
        total_bps=total_usd / notional * 10000,
        delay_cost_usd=delay_cost_usd,
        trading_cost_usd=trading_cost_usd,
        spread_cost_usd=spread_cost_usd,
        opportunity_cost_usd=opportunity_cost_usd,
        total_usd=total_usd,
    )
