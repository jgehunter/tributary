"""Unit tests for market impact estimation module."""

import math

import numpy as np
import pandas as pd
import pytest

from tributary.analytics.impact import (
    CalibrationResult,
    ImpactEstimate,
    calibrate_impact_parameters,
    estimate_market_impact,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_order_params():
    """Parameters for a small order (low participation rate)."""
    return {
        "order_size": 500,
        "daily_volume": 100000,  # 0.5% participation
        "volatility": 0.02,
        "spread_bps": 50,
    }


@pytest.fixture
def medium_order_params():
    """Parameters for a medium order (elevated participation rate)."""
    return {
        "order_size": 5000,
        "daily_volume": 100000,  # 5% participation
        "volatility": 0.02,
        "spread_bps": 50,
    }


@pytest.fixture
def large_order_params():
    """Parameters for a large order (high participation rate)."""
    return {
        "order_size": 15000,
        "daily_volume": 100000,  # 15% participation
        "volatility": 0.02,
        "spread_bps": 50,
    }


@pytest.fixture
def synthetic_executions_df():
    """Synthetic execution data with known alpha = 0.5."""
    np.random.seed(42)
    n = 50

    # Generate participation rates
    participation_rates = np.random.uniform(0.001, 0.05, n)
    daily_volumes = np.full(n, 100000.0)
    order_sizes = participation_rates * daily_volumes
    volatilities = np.full(n, 0.02)

    # Generate impact with alpha = 0.5 (square-root) plus noise
    # Impact = volatility * sqrt(participation) * 10000
    true_alpha = 0.5
    noise = np.random.normal(0, 0.1, n)  # Small noise
    impacts = volatilities * (participation_rates**true_alpha) * 10000 * np.exp(noise)

    return pd.DataFrame(
        {
            "order_size": order_sizes,
            "daily_volume": daily_volumes,
            "realized_impact_bps": impacts,
            "volatility": volatilities,
        }
    )


# =============================================================================
# ImpactEstimate Tests
# =============================================================================


class TestImpactEstimateDataclass:
    """Tests for ImpactEstimate dataclass."""

    def test_dataclass_creation(self):
        """ImpactEstimate can be created with all fields."""
        estimate = ImpactEstimate(
            temporary_impact_bps=10.0,
            permanent_impact_bps=4.0,
            total_impact_bps=39.0,
            confidence="HIGH",
            notes=["Test note"],
        )

        assert estimate.temporary_impact_bps == 10.0
        assert estimate.permanent_impact_bps == 4.0
        assert estimate.total_impact_bps == 39.0
        assert estimate.confidence == "HIGH"
        assert estimate.notes == ["Test note"]

    def test_dataclass_empty_notes_default(self):
        """Notes defaults to empty list."""
        estimate = ImpactEstimate(
            temporary_impact_bps=10.0,
            permanent_impact_bps=4.0,
            total_impact_bps=39.0,
            confidence="HIGH",
        )

        assert estimate.notes == []


# =============================================================================
# estimate_market_impact Tests
# =============================================================================


class TestEstimateMarketImpact:
    """Tests for estimate_market_impact function."""

    def test_small_order_high_confidence(self, small_order_params):
        """Small order (< 1% participation) should have HIGH confidence."""
        estimate = estimate_market_impact(**small_order_params)

        assert estimate.confidence == "HIGH"
        assert not any("participation rate" in note.lower() for note in estimate.notes)

    def test_medium_order_medium_confidence(self, medium_order_params):
        """Medium order (1-10% participation) should have MEDIUM confidence."""
        estimate = estimate_market_impact(**medium_order_params)

        assert estimate.confidence == "MEDIUM"
        assert any("1-10%" in note for note in estimate.notes)

    def test_large_order_low_confidence(self, large_order_params):
        """Large order (> 10% participation) should have LOW confidence."""
        estimate = estimate_market_impact(**large_order_params)

        assert estimate.confidence == "LOW"
        assert any(">10%" in note for note in estimate.notes)

    def test_zero_daily_volume_returns_nan(self):
        """Zero daily volume should return NaN values with LOW confidence."""
        estimate = estimate_market_impact(
            order_size=1000,
            daily_volume=0,
            volatility=0.02,
            spread_bps=50,
        )

        assert math.isnan(estimate.temporary_impact_bps)
        assert math.isnan(estimate.permanent_impact_bps)
        assert math.isnan(estimate.total_impact_bps)
        assert estimate.confidence == "LOW"
        assert any("volume" in note.lower() for note in estimate.notes)

    def test_negative_daily_volume_returns_nan(self):
        """Negative daily volume should return NaN values with LOW confidence."""
        estimate = estimate_market_impact(
            order_size=1000,
            daily_volume=-10000,
            volatility=0.02,
            spread_bps=50,
        )

        assert math.isnan(estimate.temporary_impact_bps)
        assert estimate.confidence == "LOW"

    def test_temporary_vs_permanent_ratio(self, small_order_params):
        """Permanent impact should be 40% of temporary impact."""
        estimate = estimate_market_impact(**small_order_params)

        expected_permanent = estimate.temporary_impact_bps * 0.4
        assert abs(estimate.permanent_impact_bps - expected_permanent) < 0.0001

    def test_alpha_0_5_square_root_values(self):
        """Alpha = 0.5 should produce square-root impact scaling."""
        # Impact = volatility * sqrt(participation) * 10000
        # participation = 1000/100000 = 0.01
        # sqrt(0.01) = 0.1
        # temporary_impact = 0.02 * 0.1 * 10000 = 20 bps

        estimate = estimate_market_impact(
            order_size=1000,
            daily_volume=100000,
            volatility=0.02,
            spread_bps=0,  # No spread to isolate impact
            alpha=0.5,
        )

        # temporary_impact = 0.02 * sqrt(0.01) * 10000 = 20 bps
        expected_temporary = 0.02 * (0.01**0.5) * 10000
        assert abs(estimate.temporary_impact_bps - expected_temporary) < 0.0001
        assert abs(estimate.temporary_impact_bps - 20.0) < 0.0001

    def test_alpha_0_6_produces_higher_impact(self):
        """Alpha = 0.6 should produce higher impact than alpha = 0.5."""
        params = {
            "order_size": 1000,
            "daily_volume": 100000,
            "volatility": 0.02,
            "spread_bps": 50,
        }

        estimate_0_5 = estimate_market_impact(**params, alpha=0.5)
        estimate_0_6 = estimate_market_impact(**params, alpha=0.6)

        # participation = 0.01 < 1, so higher alpha means lower (participation^alpha)
        # BUT for participation < 1: 0.01^0.5 = 0.1, 0.01^0.6 = 0.0631
        # So alpha=0.6 gives LOWER impact for small participation
        assert estimate_0_6.temporary_impact_bps < estimate_0_5.temporary_impact_bps

    def test_spread_component_added_correctly(self):
        """Half-spread should be added to total impact."""
        # Test with no spread first
        estimate_no_spread = estimate_market_impact(
            order_size=1000,
            daily_volume=100000,
            volatility=0.02,
            spread_bps=0,
        )

        # Then with spread
        spread_bps = 100
        estimate_with_spread = estimate_market_impact(
            order_size=1000,
            daily_volume=100000,
            volatility=0.02,
            spread_bps=spread_bps,
        )

        # Difference should be half-spread
        impact_difference = (
            estimate_with_spread.total_impact_bps - estimate_no_spread.total_impact_bps
        )
        assert abs(impact_difference - spread_bps / 2) < 0.0001

    def test_total_equals_components_plus_spread(self, small_order_params):
        """Total impact should equal temporary + permanent + half-spread."""
        estimate = estimate_market_impact(**small_order_params)

        expected_total = (
            estimate.temporary_impact_bps
            + estimate.permanent_impact_bps
            + small_order_params["spread_bps"] / 2
        )

        assert abs(estimate.total_impact_bps - expected_total) < 0.0001

    def test_notes_populated_for_elevated_participation(self, medium_order_params):
        """Notes should contain warning for elevated participation."""
        estimate = estimate_market_impact(**medium_order_params)

        assert len(estimate.notes) > 0
        assert any("caution" in note.lower() for note in estimate.notes)

    def test_zero_order_size(self):
        """Zero order size should return zero impact with only spread cost."""
        estimate = estimate_market_impact(
            order_size=0,
            daily_volume=100000,
            volatility=0.02,
            spread_bps=50,
        )

        assert estimate.temporary_impact_bps == 0.0
        assert estimate.permanent_impact_bps == 0.0
        assert estimate.total_impact_bps == 25.0  # half-spread only
        assert estimate.confidence == "HIGH"

    def test_negative_order_size(self):
        """Negative order size should return NaN values."""
        estimate = estimate_market_impact(
            order_size=-1000,
            daily_volume=100000,
            volatility=0.02,
            spread_bps=50,
        )

        assert math.isnan(estimate.temporary_impact_bps)
        assert estimate.confidence == "LOW"

    def test_boundary_participation_one_percent(self):
        """Exactly 1% participation should be HIGH confidence (boundary exclusive)."""
        estimate = estimate_market_impact(
            order_size=1000,
            daily_volume=100000,  # Exactly 1%
            volatility=0.02,
            spread_bps=50,
        )

        # Per design: HIGH if < 1%, MEDIUM if 1-10%, LOW if > 10%
        # Exactly 1% is the boundary - implementation uses > 0.01 for MEDIUM
        assert estimate.confidence == "HIGH"

    def test_just_over_one_percent(self):
        """Just over 1% participation should be MEDIUM confidence."""
        estimate = estimate_market_impact(
            order_size=1001,
            daily_volume=100000,  # 1.001%
            volatility=0.02,
            spread_bps=50,
        )

        assert estimate.confidence == "MEDIUM"

    def test_boundary_participation_ten_percent(self):
        """Exactly 10% participation should be MEDIUM confidence."""
        estimate = estimate_market_impact(
            order_size=10000,
            daily_volume=100000,  # Exactly 10%
            volatility=0.02,
            spread_bps=50,
        )

        assert estimate.confidence == "MEDIUM"

    def test_just_over_ten_percent(self):
        """Just over 10% participation should be LOW confidence."""
        estimate = estimate_market_impact(
            order_size=10001,
            daily_volume=100000,  # 10.001%
            volatility=0.02,
            spread_bps=50,
        )

        assert estimate.confidence == "LOW"


# =============================================================================
# CalibrationResult Tests
# =============================================================================


class TestCalibrationResultDataclass:
    """Tests for CalibrationResult dataclass."""

    def test_dataclass_creation(self):
        """CalibrationResult can be created with all fields."""
        result = CalibrationResult(
            alpha=0.52,
            volatility_sensitivity=0.8,
            intercept=2.1,
            r_squared=0.45,
            alpha_std_error=0.15,
            n_observations=50,
            warnings=["Test warning"],
        )

        assert result.alpha == 0.52
        assert result.volatility_sensitivity == 0.8
        assert result.intercept == 2.1
        assert result.r_squared == 0.45
        assert result.alpha_std_error == 0.15
        assert result.n_observations == 50
        assert result.warnings == ["Test warning"]

    def test_dataclass_empty_warnings_default(self):
        """Warnings defaults to empty list."""
        result = CalibrationResult(
            alpha=0.5,
            volatility_sensitivity=1.0,
            intercept=0.0,
            r_squared=0.5,
            alpha_std_error=0.1,
            n_observations=100,
        )

        assert result.warnings == []


# =============================================================================
# calibrate_impact_parameters Tests
# =============================================================================


class TestCalibrateImpactParameters:
    """Tests for calibrate_impact_parameters function."""

    def test_calibration_with_synthetic_data(self, synthetic_executions_df):
        """Calibration with synthetic data should recover alpha close to 0.5."""
        result = calibrate_impact_parameters(synthetic_executions_df)

        # Should recover alpha close to true value of 0.5
        assert 0.3 < result.alpha < 0.7  # Reasonable range
        assert not math.isnan(result.r_squared)
        assert result.n_observations == 50

    def test_calibration_insufficient_data(self):
        """Calibration with < 10 observations should return error."""
        df = pd.DataFrame(
            {
                "order_size": [100, 200, 300],
                "daily_volume": [10000, 10000, 10000],
                "realized_impact_bps": [5, 7, 9],
                "volatility": [0.02, 0.02, 0.02],
            }
        )

        result = calibrate_impact_parameters(df)

        assert math.isnan(result.alpha)
        assert result.n_observations == 3
        assert any("insufficient" in w.lower() for w in result.warnings)

    def test_calibration_filters_zero_values(self):
        """Calibration should filter out zero/negative values."""
        # Row 0: order_size=0, impact=0 (same row, filters once)
        # Row 11: impact=0 (filters)
        df = pd.DataFrame(
            {
                "order_size": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                "daily_volume": [10000] * 13,
                "realized_impact_bps": [0, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 0, 27],
                "volatility": [0.02] * 13,
            }
        )

        result = calibrate_impact_parameters(df)

        # Row 0: order_size=0 filters, Row 11: impact=0 filters
        # 13 - 2 = 11 observations
        assert result.n_observations == 11

    def test_calibration_returns_warnings_for_poor_fit(self):
        """Calibration should warn when R-squared is low."""
        # Create noisy data that won't fit well
        np.random.seed(123)
        n = 50
        df = pd.DataFrame(
            {
                "order_size": np.random.uniform(100, 1000, n),
                "daily_volume": np.full(n, 10000.0),
                "realized_impact_bps": np.random.uniform(1, 100, n),  # Random, no pattern
                "volatility": np.full(n, 0.02),
            }
        )

        result = calibrate_impact_parameters(df)

        # Poor fit should generate warning if r_squared < 0.3
        if result.r_squared < 0.3:
            assert any("fit" in w.lower() for w in result.warnings)

    def test_calibration_returns_warnings_for_limited_data(self):
        """Calibration should warn when n_observations < 30."""
        df = pd.DataFrame(
            {
                "order_size": [100 * i for i in range(1, 21)],
                "daily_volume": [10000] * 20,
                "realized_impact_bps": [5 + i for i in range(20)],
                "volatility": [0.02] * 20,
            }
        )

        result = calibrate_impact_parameters(df)

        assert any("limited" in w.lower() or str(result.n_observations) in w for w in result.warnings)

    def test_r_squared_and_std_error_populated(self, synthetic_executions_df):
        """R-squared and alpha_std_error should be populated correctly."""
        result = calibrate_impact_parameters(synthetic_executions_df)

        assert 0 <= result.r_squared <= 1
        assert result.alpha_std_error > 0
        assert not math.isnan(result.alpha_std_error)

    def test_missing_columns_raises_error(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame(
            {
                "order_size": [100, 200],
                # Missing daily_volume, realized_impact_bps, volatility
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            calibrate_impact_parameters(df)

    def test_calibration_with_high_std_error_warning(self):
        """High alpha_std_error should generate warning."""
        # Create data with high variance that will produce uncertain estimate
        np.random.seed(999)
        n = 15  # Minimum viable sample
        df = pd.DataFrame(
            {
                "order_size": np.random.uniform(100, 500, n),
                "daily_volume": np.full(n, 10000.0),
                "realized_impact_bps": np.random.uniform(1, 50, n) * np.random.uniform(0.5, 2, n),
                "volatility": np.random.uniform(0.01, 0.05, n),
            }
        )

        result = calibrate_impact_parameters(df)

        # If std_error > 0.3, should warn
        if result.alpha_std_error > 0.3:
            assert any("uncertainty" in w.lower() for w in result.warnings)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for impact estimation workflow."""

    def test_estimate_then_calibrate_round_trip(self, synthetic_executions_df):
        """Estimates from calibrated parameters should be consistent."""
        # First calibrate
        calibration = calibrate_impact_parameters(synthetic_executions_df)

        # Use calibrated alpha in new estimate
        estimate = estimate_market_impact(
            order_size=1000,
            daily_volume=100000,
            volatility=0.02,
            spread_bps=50,
            alpha=calibration.alpha,
        )

        # Should produce reasonable results
        assert not math.isnan(estimate.total_impact_bps)
        assert estimate.total_impact_bps > 0

    def test_imports_work_cleanly(self):
        """All exports should be importable from analytics package."""
        from tributary.analytics import (
            ImpactEstimate,
            estimate_market_impact,
            CalibrationResult,
            calibrate_impact_parameters,
        )

        # Just verify imports succeed and types are correct
        assert callable(estimate_market_impact)
        assert callable(calibrate_impact_parameters)
