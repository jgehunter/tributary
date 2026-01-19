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

__all__ = [
    "QuestDBReader",
    "calculate_vwap",
    "calculate_cumulative_vwap",
    "calculate_vwap_for_period",
    "calculate_twap",
    "calculate_twap_from_orderbooks",
    "get_arrival_price",
    "calculate_slippage_bps",
    "ShortfallComponents",
    "decompose_implementation_shortfall",
]
