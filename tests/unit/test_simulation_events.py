"""Tests for simulation event types.

Tests that MarketEvent, OrderEvent, and FillEvent are correctly defined
as frozen (immutable) dataclasses with proper type hints.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from tributary.analytics.simulation import FillEvent, MarketEvent, OrderEvent


class TestMarketEvent:
    """Tests for MarketEvent dataclass."""

    @pytest.fixture
    def sample_market_event(self) -> MarketEvent:
        """Create a sample market event for testing."""
        return MarketEvent(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            market_id="condition-123",
            token_id="token-yes",
            mid_price=0.50,
            bid_prices=(0.49, 0.48, 0.47),
            bid_sizes=(1000.0, 2000.0, 3000.0),
            ask_prices=(0.51, 0.52, 0.53),
            ask_sizes=(1000.0, 2000.0, 3000.0),
        )

    def test_market_event_instantiation(self, sample_market_event: MarketEvent) -> None:
        """Test MarketEvent can be instantiated with valid data."""
        assert sample_market_event.timestamp == datetime(
            2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
        )
        assert sample_market_event.market_id == "condition-123"
        assert sample_market_event.token_id == "token-yes"
        assert sample_market_event.mid_price == 0.50
        assert sample_market_event.bid_prices == (0.49, 0.48, 0.47)
        assert sample_market_event.bid_sizes == (1000.0, 2000.0, 3000.0)
        assert sample_market_event.ask_prices == (0.51, 0.52, 0.53)
        assert sample_market_event.ask_sizes == (1000.0, 2000.0, 3000.0)

    def test_market_event_is_frozen(self, sample_market_event: MarketEvent) -> None:
        """Test MarketEvent is immutable (frozen)."""
        with pytest.raises(FrozenInstanceError):
            sample_market_event.mid_price = 0.60  # type: ignore[misc]

    def test_market_event_tuple_fields(self) -> None:
        """Test tuple fields work correctly."""
        event = MarketEvent(
            timestamp=datetime.now(timezone.utc),
            market_id="test",
            token_id="test",
            mid_price=0.50,
            bid_prices=(0.49,),
            bid_sizes=(1000.0,),
            ask_prices=(0.51,),
            ask_sizes=(1000.0,),
        )
        assert len(event.bid_prices) == 1
        assert len(event.ask_prices) == 1

    def test_market_event_empty_tuples(self) -> None:
        """Test MarketEvent can handle empty orderbook."""
        event = MarketEvent(
            timestamp=datetime.now(timezone.utc),
            market_id="test",
            token_id="test",
            mid_price=float("nan"),
            bid_prices=(),
            bid_sizes=(),
            ask_prices=(),
            ask_sizes=(),
        )
        assert len(event.bid_prices) == 0
        assert len(event.ask_prices) == 0


class TestOrderEvent:
    """Tests for OrderEvent dataclass."""

    @pytest.fixture
    def sample_order_event(self) -> OrderEvent:
        """Create a sample order event for testing."""
        return OrderEvent(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            strategy_name="twap",
            slice_index=0,
            size=1000.0,
            side="buy",
        )

    def test_order_event_instantiation(self, sample_order_event: OrderEvent) -> None:
        """Test OrderEvent can be instantiated with valid data."""
        assert sample_order_event.timestamp == datetime(
            2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
        )
        assert sample_order_event.strategy_name == "twap"
        assert sample_order_event.slice_index == 0
        assert sample_order_event.size == 1000.0
        assert sample_order_event.side == "buy"

    def test_order_event_is_frozen(self, sample_order_event: OrderEvent) -> None:
        """Test OrderEvent is immutable (frozen)."""
        with pytest.raises(FrozenInstanceError):
            sample_order_event.size = 2000.0  # type: ignore[misc]

    def test_order_event_sell_side(self) -> None:
        """Test OrderEvent works for sell orders."""
        order = OrderEvent(
            timestamp=datetime.now(timezone.utc),
            strategy_name="vwap",
            slice_index=5,
            size=500.0,
            side="sell",
        )
        assert order.side == "sell"
        assert order.slice_index == 5


class TestFillEvent:
    """Tests for FillEvent dataclass."""

    @pytest.fixture
    def sample_fill_event(self) -> FillEvent:
        """Create a sample fill event for testing."""
        return FillEvent(
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            strategy_name="twap",
            slice_index=0,
            requested_size=1000.0,
            filled_size=1000.0,
            avg_price=0.51,
            slippage_bps=20.0,
            levels_consumed=1,
            mid_price_at_fill=0.50,
        )

    def test_fill_event_instantiation(self, sample_fill_event: FillEvent) -> None:
        """Test FillEvent can be instantiated with valid data."""
        assert sample_fill_event.timestamp == datetime(
            2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
        )
        assert sample_fill_event.strategy_name == "twap"
        assert sample_fill_event.slice_index == 0
        assert sample_fill_event.requested_size == 1000.0
        assert sample_fill_event.filled_size == 1000.0
        assert sample_fill_event.avg_price == 0.51
        assert sample_fill_event.slippage_bps == 20.0
        assert sample_fill_event.levels_consumed == 1
        assert sample_fill_event.mid_price_at_fill == 0.50

    def test_fill_event_is_frozen(self, sample_fill_event: FillEvent) -> None:
        """Test FillEvent is immutable (frozen)."""
        with pytest.raises(FrozenInstanceError):
            sample_fill_event.filled_size = 500.0  # type: ignore[misc]

    def test_fill_event_partial_fill(self) -> None:
        """Test FillEvent can represent partial fills."""
        fill = FillEvent(
            timestamp=datetime.now(timezone.utc),
            strategy_name="market",
            slice_index=0,
            requested_size=10000.0,
            filled_size=6000.0,
            avg_price=0.53,
            slippage_bps=60.0,
            levels_consumed=3,
            mid_price_at_fill=0.50,
        )
        assert fill.filled_size < fill.requested_size
        assert fill.levels_consumed == 3

    def test_fill_event_zero_fill(self) -> None:
        """Test FillEvent can represent no fill (empty orderbook)."""
        fill = FillEvent(
            timestamp=datetime.now(timezone.utc),
            strategy_name="twap",
            slice_index=0,
            requested_size=1000.0,
            filled_size=0.0,
            avg_price=float("nan"),
            slippage_bps=float("nan"),
            levels_consumed=0,
            mid_price_at_fill=float("nan"),
        )
        assert fill.filled_size == 0.0
        assert fill.levels_consumed == 0


class TestEventInteroperability:
    """Tests for event type interactions."""

    def test_event_types_hashable(self) -> None:
        """Test frozen events can be used in sets/dicts."""
        market = MarketEvent(
            timestamp=datetime.now(timezone.utc),
            market_id="test",
            token_id="test",
            mid_price=0.50,
            bid_prices=(0.49,),
            bid_sizes=(1000.0,),
            ask_prices=(0.51,),
            ask_sizes=(1000.0,),
        )
        order = OrderEvent(
            timestamp=datetime.now(timezone.utc),
            strategy_name="twap",
            slice_index=0,
            size=1000.0,
            side="buy",
        )
        fill = FillEvent(
            timestamp=datetime.now(timezone.utc),
            strategy_name="twap",
            slice_index=0,
            requested_size=1000.0,
            filled_size=1000.0,
            avg_price=0.51,
            slippage_bps=20.0,
            levels_consumed=1,
            mid_price_at_fill=0.50,
        )

        # Should not raise - frozen dataclasses are hashable
        event_set = {market, order, fill}
        assert len(event_set) == 3

    def test_event_equality(self) -> None:
        """Test event equality based on content."""
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        order1 = OrderEvent(
            timestamp=ts,
            strategy_name="twap",
            slice_index=0,
            size=1000.0,
            side="buy",
        )
        order2 = OrderEvent(
            timestamp=ts,
            strategy_name="twap",
            slice_index=0,
            size=1000.0,
            side="buy",
        )
        order3 = OrderEvent(
            timestamp=ts,
            strategy_name="twap",
            slice_index=1,  # Different slice
            size=1000.0,
            side="buy",
        )

        assert order1 == order2
        assert order1 != order3
