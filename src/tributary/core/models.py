"""Core data models for market data."""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, computed_field


class AssetType(str, Enum):
    """Supported asset types."""

    PREDICTION_MARKET = "prediction_market"
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"


class Exchange(str, Enum):
    """Supported exchanges."""

    POLYMARKET = "polymarket"
    BINANCE = "binance"


class Side(str, Enum):
    """Trade side."""

    BUY = "buy"
    SELL = "sell"


class Market(BaseModel):
    """Unified market representation across exchanges."""

    market_id: str
    market_slug: str
    question: str
    asset_type: AssetType
    exchange: Exchange
    creation_time: datetime
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trade(BaseModel):
    """Unified trade representation across exchanges."""

    market_id: str  # Market-level ID (e.g., conditionId for Polymarket)
    token_id: str  # Outcome-level ID (e.g., asset_id/clobTokenId for Polymarket)
    trade_id: str
    timestamp: datetime
    price: float
    size: float
    side: Side
    is_buyer_maker: Optional[bool] = None  # None = unknown, can be inferred later

    @computed_field
    @property
    def value(self) -> float:
        """Computed trade value (price * size)."""
        return self.price * self.size


class OrderBookSnapshot(BaseModel):
    """Point-in-time orderbook snapshot."""

    market_id: str  # Market-level ID (e.g., conditionId for Polymarket)
    token_id: str  # Outcome-level ID (e.g., asset_id/clobTokenId for Polymarket)
    timestamp: datetime
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    bid_prices: List[float] = Field(default_factory=list)
    bid_sizes: List[float] = Field(default_factory=list)
    ask_prices: List[float] = Field(default_factory=list)
    ask_sizes: List[float] = Field(default_factory=list)

    @computed_field
    @property
    def mid_price(self) -> float:
        """Mid price between best bid and ask."""
        # If one side is empty (using defaults 0/1), mid_price is less meaningful
        return (self.best_bid + self.best_ask) / 2

    @computed_field
    @property
    def spread(self) -> float:
        """Spread between best ask and bid."""
        return self.best_ask - self.best_bid

    @computed_field
    @property
    def spread_bps(self) -> float:
        """Spread in basis points. Returns NaN if orderbook is one-sided or invalid."""
        # Guard against one-sided orderbooks (defaults: bid=0, ask=1)
        if self.best_bid <= 0 or self.best_ask >= 1:
            return float('nan')
        if self.mid_price == 0:
            return float('nan')
        return (self.spread / self.mid_price) * 10000

    @computed_field
    @property
    def is_two_sided(self) -> bool:
        """True if orderbook has both bids and asks."""
        return self.best_bid > 0 and self.best_ask < 1

    @computed_field
    @property
    def total_bid_volume(self) -> float:
        """Total volume on bid side."""
        return sum(self.bid_sizes)

    @computed_field
    @property
    def total_ask_volume(self) -> float:
        """Total volume on ask side."""
        return sum(self.ask_sizes)
