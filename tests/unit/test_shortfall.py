"""Unit tests for implementation shortfall decomposition."""

import math

import pytest

from tributary.analytics.shortfall import (
    ShortfallComponents,
    decompose_implementation_shortfall,
)


# =============================================================================
# ShortfallComponents Tests
# =============================================================================


class TestShortfallComponents:
    """Tests for ShortfallComponents dataclass."""

    def test_shortfall_components_creation(self):
        """ShortfallComponents should store all cost components."""
        components = ShortfallComponents(
            delay_cost_bps=100.0,
            trading_cost_bps=50.0,
            spread_cost_bps=10.0,
            opportunity_cost_bps=25.0,
            total_bps=185.0,
            delay_cost_usd=100.0,
            trading_cost_usd=50.0,
            spread_cost_usd=10.0,
            opportunity_cost_usd=25.0,
            total_usd=185.0,
        )

        assert components.delay_cost_bps == 100.0
        assert components.trading_cost_bps == 50.0
        assert components.spread_cost_bps == 10.0
        assert components.opportunity_cost_bps == 25.0
        assert components.total_bps == 185.0


# =============================================================================
# decompose_implementation_shortfall Tests
# =============================================================================


class TestDecomposeImplementationShortfall:
    """Tests for decompose_implementation_shortfall function."""

    def test_full_execution_buy_side(self):
        """Full execution buy order with all components."""
        # Decision price: 100, Entry price: 101 (delay cost)
        # Execution price: 102 (trading cost)
        # Spread: 0.20 (spread cost = 0.10 per unit)
        # All 1000 units filled, no opportunity cost
        # Notional: 1000 * 100 = 100,000

        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=101.0,
            execution_prices=[102.0],
            execution_sizes=[1000.0],
            total_order_size=1000.0,
            closing_price=103.0,
            side="buy",
            spread_at_entry=0.20,
        )

        # Delay: 1000 * (101 - 100) = 1000 USD = 100 bps
        assert result.delay_cost_usd == 1000.0
        assert result.delay_cost_bps == 100.0

        # Trading: 1000 * (102 - 101) = 1000 USD = 100 bps
        assert result.trading_cost_usd == 1000.0
        assert result.trading_cost_bps == 100.0

        # Spread: 1000 * 0.10 = 100 USD = 10 bps
        assert result.spread_cost_usd == 100.0
        assert result.spread_cost_bps == 10.0

        # Opportunity: 0 units unfilled
        assert result.opportunity_cost_usd == 0.0
        assert result.opportunity_cost_bps == 0.0

        # Total: 100 + 100 + 10 + 0 = 210 bps
        assert result.total_bps == 210.0

    def test_full_execution_sell_side(self):
        """Full execution sell order with sign flipping."""
        # For sells, price moving up is unfavorable
        # Decision: 100, Entry: 99 (price dropped, good for sell -> negative delay)
        # Execution: 98 (price dropped more -> positive trading cost because we got less)

        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=99.0,
            execution_prices=[98.0],
            execution_sizes=[1000.0],
            total_order_size=1000.0,
            closing_price=97.0,
            side="sell",
            spread_at_entry=0.20,
        )

        # Delay: 1000 * (99 - 100) = -1000, but for sell flip sign: +1000 (cost)
        # Wait - price went DOWN from 100 to 99. For a sell, this means we could
        # sell at a worse price. So this is favorable (gain).
        # Raw: 1000 * (99 - 100) = -1000. Flip for sell: +1000? No.
        # Let me reconsider: For sell, if price dropped before entry, we enter at
        # a lower price which is bad. But the formula flips the sign.
        # Raw delay: executed_size * (entry - decision) = 1000 * (99-100) = -1000
        # For sell, we flip: -(-1000) = 1000
        # This means price dropping is a COST for the unfilled portion logic, but
        # for delay cost on executed portion: if price drops before we enter,
        # that's favorable for a sell! So the sign should be negative (gain).
        # Hmm, let me check the formula more carefully:
        # delay_cost = executed_size * (entry - decision) = 1000 * (-1) = -1000
        # For sell: delay_cost = -(-1000) = 1000
        # This says delay was a cost of $1000. But if I'm selling and price
        # went from 100 to 99 before I entered, I'm selling at a worse price...
        # Actually for a SELL: I want to sell HIGH. Price dropping is bad.
        # So the $1000 cost is correct.
        assert result.delay_cost_usd == 1000.0
        assert result.delay_cost_bps == 100.0

        # Trading: 1000 * (98 - 99) = -1000, flip for sell: +1000
        # Price dropped further during execution, so we received less -> cost
        assert result.trading_cost_usd == 1000.0
        assert result.trading_cost_bps == 100.0

        # Spread cost is always positive (crossing spread is a cost)
        assert result.spread_cost_usd == 100.0
        assert result.spread_cost_bps == 10.0

        # Total: 100 + 100 + 10 = 210 bps
        assert result.total_bps == 210.0

    def test_partial_execution_opportunity_cost(self):
        """Partial execution should have opportunity cost."""
        # Only 500 of 1000 units filled
        # Closing price is 105, decision was 100
        # Unfilled 500 units miss out on 5 points each

        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=100.0,  # No delay
            execution_prices=[100.0],  # No trading impact
            execution_sizes=[500.0],
            total_order_size=1000.0,
            closing_price=105.0,
            side="buy",
            spread_at_entry=None,  # No spread cost
        )

        assert result.delay_cost_bps == 0.0
        assert result.trading_cost_bps == 0.0
        assert result.spread_cost_bps == 0.0

        # Opportunity: 500 * (105 - 100) = 2500 USD
        # Notional: 1000 * 100 = 100,000
        # Opportunity bps: 2500 / 100000 * 10000 = 250 bps
        assert result.opportunity_cost_usd == 2500.0
        assert result.opportunity_cost_bps == 250.0
        assert result.total_bps == 250.0

    def test_zero_execution_all_opportunity_cost(self):
        """Zero execution means 100% opportunity cost."""
        # Nothing filled, price moved from 100 to 110
        # All 1000 units have opportunity cost of 10 each

        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=100.0,
            execution_prices=[],
            execution_sizes=[],
            total_order_size=1000.0,
            closing_price=110.0,
            side="buy",
        )

        assert result.delay_cost_bps == 0.0
        assert result.trading_cost_bps == 0.0
        assert result.spread_cost_bps == 0.0

        # Opportunity: 1000 * (110 - 100) = 10,000 USD = 1000 bps
        assert result.opportunity_cost_usd == 10000.0
        assert result.opportunity_cost_bps == 1000.0
        assert result.total_bps == 1000.0

    def test_multiple_fills_at_different_prices(self):
        """Multiple fills should calculate VWAP execution price."""
        # Two fills: 300 @ 101, 700 @ 102
        # VWAP: (300*101 + 700*102) / 1000 = 101.7

        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=100.0,
            execution_prices=[101.0, 102.0],
            execution_sizes=[300.0, 700.0],
            total_order_size=1000.0,
            closing_price=100.0,
            side="buy",
        )

        # Trading cost: 1000 * (101.7 - 100) = 1700 USD = 170 bps
        assert abs(result.trading_cost_usd - 1700.0) < 0.001
        assert abs(result.trading_cost_bps - 170.0) < 0.001

    def test_no_price_movement_zero_delay(self):
        """No price movement between decision and entry means zero delay cost."""
        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=100.0,  # Same as decision
            execution_prices=[101.0],
            execution_sizes=[1000.0],
            total_order_size=1000.0,
            closing_price=100.0,
            side="buy",
        )

        assert result.delay_cost_bps == 0.0
        assert result.delay_cost_usd == 0.0

    def test_spread_cost_calculation(self):
        """Spread cost should be half the spread times executed size."""
        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=100.0,
            execution_prices=[100.0],
            execution_sizes=[1000.0],
            total_order_size=1000.0,
            closing_price=100.0,
            side="buy",
            spread_at_entry=0.50,  # 50 cents spread
        )

        # Spread cost: 1000 * 0.25 = 250 USD = 25 bps
        assert result.spread_cost_usd == 250.0
        assert result.spread_cost_bps == 25.0

    def test_missing_spread_zero_spread_cost(self):
        """Missing spread should result in zero spread cost."""
        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=100.0,
            execution_prices=[100.0],
            execution_sizes=[1000.0],
            total_order_size=1000.0,
            closing_price=100.0,
            side="buy",
            spread_at_entry=None,
        )

        assert result.spread_cost_bps == 0.0
        assert result.spread_cost_usd == 0.0

    def test_sign_convention_sell_all_costs_unfavorable(self):
        """Verify sign convention for sell side with unfavorable execution."""
        # For a sell: we want price to stay high or go higher
        # Price dropped at each stage -> all costs are positive

        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=99.0,  # Price dropped (bad for sell)
            execution_prices=[98.0],  # Price dropped more (bad for sell)
            execution_sizes=[800.0],
            total_order_size=1000.0,
            closing_price=97.0,  # Price dropped even more (bad for unfilled)
            side="sell",
            spread_at_entry=0.20,
        )

        # All components should be positive (costs) for this scenario
        assert result.delay_cost_bps > 0  # Price dropped before entry
        assert result.trading_cost_bps > 0  # Price dropped during execution
        assert result.spread_cost_bps > 0  # Spread is always positive
        assert result.opportunity_cost_bps > 0  # Price dropped, unfilled lost out
        assert result.total_bps > 0

    def test_empty_execution_lists(self):
        """Empty execution lists should work (zero execution case)."""
        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=100.0,
            execution_prices=[],
            execution_sizes=[],
            total_order_size=1000.0,
            closing_price=100.0,
            side="buy",
        )

        assert result.delay_cost_bps == 0.0
        assert result.trading_cost_bps == 0.0
        assert result.spread_cost_bps == 0.0
        # Price didn't move, so no opportunity cost either
        assert result.opportunity_cost_bps == 0.0
        assert result.total_bps == 0.0

    def test_invalid_side_raises_error(self):
        """Invalid side should raise ValueError."""
        with pytest.raises(ValueError, match="side must be 'buy' or 'sell'"):
            decompose_implementation_shortfall(
                decision_price=100.0,
                order_entry_price=100.0,
                execution_prices=[100.0],
                execution_sizes=[1000.0],
                total_order_size=1000.0,
                closing_price=100.0,
                side="invalid",
            )

    def test_mismatched_execution_lists_raises_error(self):
        """Mismatched prices and sizes lists should raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            decompose_implementation_shortfall(
                decision_price=100.0,
                order_entry_price=100.0,
                execution_prices=[100.0, 101.0],  # 2 prices
                execution_sizes=[1000.0],  # 1 size
                total_order_size=1000.0,
                closing_price=100.0,
                side="buy",
            )

    def test_zero_notional_returns_zero_bps(self):
        """Zero decision price (zero notional) should return zero bps."""
        result = decompose_implementation_shortfall(
            decision_price=0.0,
            order_entry_price=0.0,
            execution_prices=[1.0],
            execution_sizes=[100.0],
            total_order_size=100.0,
            closing_price=1.0,
            side="buy",
        )

        # Can't compute bps with zero notional
        assert result.delay_cost_bps == 0.0
        assert result.trading_cost_bps == 0.0
        assert result.total_bps == 0.0
        # But USD values are still computed
        assert result.trading_cost_usd == 100.0  # 100 * (1 - 0)

    def test_favorable_execution_negative_values(self):
        """Favorable execution should result in negative costs (gains)."""
        # Buy order where price dropped during execution (favorable!)
        result = decompose_implementation_shortfall(
            decision_price=100.0,
            order_entry_price=99.0,  # Price dropped (good for buy)
            execution_prices=[98.0],  # Price dropped more (good for buy)
            execution_sizes=[1000.0],
            total_order_size=1000.0,
            closing_price=100.0,
            side="buy",
        )

        # Delay: paid 99 instead of 100 -> gain -> negative
        assert result.delay_cost_bps == -100.0
        # Trading: paid 98 instead of 99 -> gain -> negative
        assert result.trading_cost_bps == -100.0
        # Total: negative (favorable overall)
        assert result.total_bps == -200.0
