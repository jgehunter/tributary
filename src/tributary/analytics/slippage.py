"""Slippage calculation for execution quality measurement.

Provides functions to calculate execution slippage in basis points,
measuring the difference between expected and actual execution prices.
"""


def calculate_slippage_bps(
    execution_price: float,
    benchmark_price: float,
    side: str,
) -> float:
    """
    Calculate slippage in basis points.

    Slippage measures the difference between the actual execution price
    and a benchmark price (arrival price, VWAP, TWAP, etc.).

    Sign convention:
        - Positive value = cost (unfavorable execution)
        - Negative value = gain (favorable execution)

    For buys: paying MORE than benchmark is a cost (positive slippage)
    For sells: receiving LESS than benchmark is a cost (positive slippage)

    Args:
        execution_price: Actual average execution price
        benchmark_price: Reference price (arrival, VWAP, or TWAP)
        side: Trade direction ('buy' or 'sell')

    Returns:
        Slippage in basis points (1 bp = 0.01%). Returns NaN if
        benchmark_price is zero or invalid.

    Raises:
        ValueError: If side is not 'buy' or 'sell'

    Examples:
        >>> # Buy order: paid 102 when benchmark was 100 -> 200 bps cost
        >>> calculate_slippage_bps(102, 100, 'buy')
        200.0

        >>> # Sell order: received 98 when benchmark was 100 -> 200 bps cost
        >>> calculate_slippage_bps(98, 100, 'sell')
        200.0

        >>> # Favorable buy: paid 98 when benchmark was 100 -> -200 bps (gain)
        >>> calculate_slippage_bps(98, 100, 'buy')
        -200.0
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

    if benchmark_price == 0:
        return float("nan")

    raw_slippage = (execution_price - benchmark_price) / benchmark_price * 10000

    # For sells, flip sign: getting less than expected is a cost
    if side == "sell":
        return -raw_slippage

    return raw_slippage
