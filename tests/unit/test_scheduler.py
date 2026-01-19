"""Unit tests for trade scheduler optimizer.

Tests the ScheduleConstraints, TradeSchedule dataclasses and the
optimize_schedule() function that wraps Almgren-Chriss trajectory
generation with participation and slice size constraints.
"""

import math

import numpy as np
import pytest

from tributary.analytics.optimization.scheduler import (
    ScheduleConstraints,
    TradeSchedule,
    calculate_optimal_intervals,
    optimize_schedule,
)
from tributary.analytics.optimization.almgren_chriss import (
    AlmgrenChrissParams,
    calibrate_ac_params,
)


# =============================================================================
# Test ScheduleConstraints
# =============================================================================


class TestScheduleConstraints:
    """Tests for ScheduleConstraints dataclass."""

    def test_constraints_defaults(self):
        """Test default constraint values."""
        constraints = ScheduleConstraints()
        assert constraints.max_participation_rate == 0.10
        assert constraints.min_slice_size == 0.0
        assert constraints.max_slice_size == float("inf")
        assert constraints.min_intervals == 1
        assert constraints.max_intervals is None

    def test_constraints_custom_values(self):
        """Test custom constraint values."""
        constraints = ScheduleConstraints(
            max_participation_rate=0.05,
            min_slice_size=100.0,
            max_slice_size=10000.0,
            min_intervals=5,
            max_intervals=20,
        )
        assert constraints.max_participation_rate == 0.05
        assert constraints.min_slice_size == 100.0
        assert constraints.max_slice_size == 10000.0
        assert constraints.min_intervals == 5
        assert constraints.max_intervals == 20

    def test_constraints_is_frozen(self):
        """Test that constraints are immutable."""
        constraints = ScheduleConstraints()
        with pytest.raises(Exception):  # FrozenInstanceError
            constraints.max_participation_rate = 0.05


# =============================================================================
# Test TradeSchedule
# =============================================================================


class TestTradeSchedule:
    """Tests for TradeSchedule dataclass."""

    @pytest.fixture
    def sample_params(self):
        """Create sample A-C params for tests."""
        return calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )

    def test_schedule_creation(self, sample_params):
        """Test TradeSchedule can be created with all fields."""
        from tributary.analytics.optimization.almgren_chriss import (
            generate_ac_trajectory,
        )

        trajectory = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=sample_params,
            risk_aversion=1e-6,
        )
        constraints = ScheduleConstraints()

        schedule = TradeSchedule(
            trajectory=trajectory,
            constraints=constraints,
            intervals_used=10,
            max_slice_pct=12.5,
            meets_participation_constraint=True,
            warnings=(),
        )

        assert schedule.trajectory == trajectory
        assert schedule.constraints == constraints
        assert schedule.intervals_used == 10
        assert schedule.max_slice_pct == 12.5
        assert schedule.meets_participation_constraint is True
        assert schedule.warnings == ()

    def test_schedule_is_frozen(self, sample_params):
        """Test that schedule is immutable."""
        from tributary.analytics.optimization.almgren_chriss import (
            generate_ac_trajectory,
        )

        trajectory = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=sample_params,
            risk_aversion=1e-6,
        )
        schedule = TradeSchedule(
            trajectory=trajectory,
            constraints=ScheduleConstraints(),
            intervals_used=10,
            max_slice_pct=12.5,
            meets_participation_constraint=True,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            schedule.intervals_used = 20


# =============================================================================
# Test calculate_optimal_intervals
# =============================================================================


class TestCalculateOptimalIntervals:
    """Tests for calculate_optimal_intervals helper function."""

    def test_calculate_intervals_basic(self):
        """Test basic interval calculation."""
        # 10,000 order, 100,000 volume, 10% max = need 1 interval
        intervals = calculate_optimal_intervals(
            order_size=10000,
            expected_interval_volume=100000,
            max_participation_rate=0.10,
        )
        assert intervals == 1

    def test_calculate_intervals_large_order(self):
        """Test interval calculation for large order."""
        # 50,000 order, 100,000 volume, 10% max = need 5 intervals
        intervals = calculate_optimal_intervals(
            order_size=50000,
            expected_interval_volume=100000,
            max_participation_rate=0.10,
        )
        assert intervals == 5

    def test_calculate_intervals_small_order(self):
        """Test interval calculation for small order."""
        # 500 order, 100,000 volume, 10% max = need 1 interval
        intervals = calculate_optimal_intervals(
            order_size=500,
            expected_interval_volume=100000,
            max_participation_rate=0.10,
        )
        assert intervals == 1

    def test_calculate_intervals_exact_fit(self):
        """Test when order exactly fits participation limit."""
        # 10,000 order, 100,000 volume, 10% max = exactly 1 interval
        intervals = calculate_optimal_intervals(
            order_size=10000,
            expected_interval_volume=100000,
            max_participation_rate=0.10,
        )
        assert intervals == 1

        # Just over the limit needs 2 intervals
        intervals = calculate_optimal_intervals(
            order_size=10001,
            expected_interval_volume=100000,
            max_participation_rate=0.10,
        )
        assert intervals == 2

    def test_calculate_intervals_invalid_order_size(self):
        """Test with invalid order size."""
        assert calculate_optimal_intervals(0, 100000, 0.10) == 1
        assert calculate_optimal_intervals(-100, 100000, 0.10) == 1

    def test_calculate_intervals_invalid_volume(self):
        """Test with invalid volume."""
        assert calculate_optimal_intervals(1000, 0, 0.10) == 1
        assert calculate_optimal_intervals(1000, -100, 0.10) == 1

    def test_calculate_intervals_low_participation_rate(self):
        """Test with low participation rate."""
        # 10,000 order, 100,000 volume, 1% max = need 10 intervals
        intervals = calculate_optimal_intervals(
            order_size=10000,
            expected_interval_volume=100000,
            max_participation_rate=0.01,
        )
        assert intervals == 10


# =============================================================================
# Test optimize_schedule
# =============================================================================


class TestOptimizeSchedule:
    """Tests for optimize_schedule function."""

    @pytest.fixture
    def sample_params(self):
        """Create sample A-C params for tests."""
        return calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )

    def test_optimize_basic(self, sample_params):
        """Test basic schedule optimization."""
        schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=10000,  # 10% is 1000, our order size
            risk_aversion=1e-6,
        )

        assert schedule.trajectory is not None
        assert schedule.trajectory.strategy_name == "almgren_chriss"
        assert schedule.intervals_used >= 1
        assert schedule.max_slice_pct > 0
        assert len(schedule.warnings) == 0 or schedule.meets_participation_constraint

    def test_optimize_respects_participation_rate(self, sample_params):
        """Test that participation constraint is respected with TWAP (risk-neutral)."""
        # Use risk_aversion=0 for TWAP which has uniform slices
        schedule = optimize_schedule(
            order_size=5000,
            params=sample_params,
            expected_interval_volume=1000,  # 10% is only 100
            risk_aversion=0,  # TWAP for uniform execution
        )

        # With 5000 order and only 100 per interval allowed, need at least 50 intervals
        assert schedule.intervals_used >= 50
        # TWAP with 50 intervals gives 100 per slice, exactly meeting 10% constraint
        assert schedule.meets_participation_constraint

    def test_optimize_calculates_intervals_for_risk_averse(self, sample_params):
        """Test interval calculation with risk aversion (may slightly exceed participation)."""
        schedule = optimize_schedule(
            order_size=5000,
            params=sample_params,
            expected_interval_volume=1000,  # 10% is only 100
            risk_aversion=1e-6,  # Non-zero risk aversion front-loads
        )

        # Should still calculate 50 intervals based on uniform assumption
        assert schedule.intervals_used >= 50
        # With front-loading, first slice may exceed participation limit - this is noted in warnings
        if not schedule.meets_participation_constraint:
            assert len(schedule.warnings) > 0

    def test_optimize_minimum_intervals_from_participation(self, sample_params):
        """Test minimum intervals derived from participation constraint."""
        # Order = 2000, volume = 1000, 10% max = 100 per interval
        # Need ceil(2000 / 100) = 20 intervals
        schedule = optimize_schedule(
            order_size=2000,
            params=sample_params,
            expected_interval_volume=1000,
            risk_aversion=1e-6,
        )

        assert schedule.intervals_used >= 20

    def test_optimize_respects_min_intervals_constraint(self, sample_params):
        """Test that min_intervals constraint is respected."""
        constraints = ScheduleConstraints(min_intervals=15)

        schedule = optimize_schedule(
            order_size=100,  # Small order, would normally need 1 interval
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
            constraints=constraints,
        )

        assert schedule.intervals_used >= 15

    def test_optimize_respects_max_intervals_constraint(self, sample_params):
        """Test that max_intervals constraint is respected."""
        constraints = ScheduleConstraints(max_intervals=5)

        schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=100,  # Would need 100 intervals
            risk_aversion=1e-6,
            constraints=constraints,
        )

        # Should be capped at 5 with warning
        assert schedule.intervals_used == 5
        assert any("max_intervals" in w for w in schedule.warnings)

    def test_optimize_large_order_needs_more_intervals(self, sample_params):
        """Test that large orders use more intervals."""
        small_schedule = optimize_schedule(
            order_size=100,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
        )

        large_schedule = optimize_schedule(
            order_size=10000,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
        )

        # Large order should need more intervals to stay within participation
        assert large_schedule.intervals_used >= small_schedule.intervals_used

    def test_optimize_small_order_single_interval(self, sample_params):
        """Test that small orders can use single interval."""
        schedule = optimize_schedule(
            order_size=100,
            params=sample_params,
            expected_interval_volume=100000,  # 10% = 10,000, way more than needed
            risk_aversion=1e-6,
        )

        assert schedule.intervals_used == 1
        assert schedule.meets_participation_constraint

    def test_optimize_warns_on_constraint_violation(self, sample_params):
        """Test warnings when constraints are violated."""
        # Set impossible constraints
        constraints = ScheduleConstraints(
            min_slice_size=500,  # Each slice must be at least 500
            max_slice_size=50,  # Each slice must be at most 50 (impossible!)
        )

        schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
            constraints=constraints,
        )

        # Should have warnings about violated constraints
        assert len(schedule.warnings) > 0

    def test_optimize_invalid_order_size(self, sample_params):
        """Test with invalid order size."""
        schedule = optimize_schedule(
            order_size=0,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
        )

        assert np.isnan(schedule.trajectory.trade_sizes).all()
        assert "order size" in schedule.warnings[0].lower()

        schedule = optimize_schedule(
            order_size=-100,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
        )

        assert np.isnan(schedule.trajectory.trade_sizes).all()
        assert "order size" in schedule.warnings[0].lower()

    def test_optimize_invalid_volume(self, sample_params):
        """Test with invalid expected volume."""
        schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=0,
            risk_aversion=1e-6,
        )

        assert np.isnan(schedule.trajectory.trade_sizes).all()
        assert "volume" in schedule.warnings[0].lower()

        schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=-100,
            risk_aversion=1e-6,
        )

        assert np.isnan(schedule.trajectory.trade_sizes).all()
        assert "volume" in schedule.warnings[0].lower()

    def test_optimize_default_constraints(self, sample_params):
        """Test that default constraints are used when not provided."""
        schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
            constraints=None,
        )

        # Should use default constraints
        assert schedule.constraints.max_participation_rate == 0.10
        assert schedule.constraints.min_intervals == 1

    def test_optimize_trajectory_sums_to_order_size(self, sample_params):
        """Test that trajectory trade sizes sum to order size."""
        schedule = optimize_schedule(
            order_size=5000,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-6,
        )

        total_traded = np.sum(schedule.trajectory.trade_sizes)
        np.testing.assert_almost_equal(total_traded, 5000, decimal=6)

    def test_optimize_high_risk_aversion_front_loads(self, sample_params):
        """Test that high risk aversion front-loads execution."""
        # Low risk aversion (risk neutral)
        low_ra_schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=0,
            constraints=ScheduleConstraints(min_intervals=10),
        )

        # High risk aversion
        high_ra_schedule = optimize_schedule(
            order_size=1000,
            params=sample_params,
            expected_interval_volume=10000,
            risk_aversion=1e-3,
            constraints=ScheduleConstraints(min_intervals=10),
        )

        # High risk aversion should have larger first slice
        low_first = low_ra_schedule.trajectory.trade_sizes[0]
        high_first = high_ra_schedule.trajectory.trade_sizes[0]

        # Low RA (TWAP) should have equal slices, high RA should front-load
        assert high_first > low_first
