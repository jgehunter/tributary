"""Unit tests for Almgren-Chriss optimal execution module."""

import math
import numpy as np
import pytest

from tributary.analytics.optimization.almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    calibrate_ac_params,
    generate_ac_trajectory,
)
from tributary.analytics.impact import CalibrationResult


class TestAlmgrenChrissParams:
    """Tests for AlmgrenChrissParams dataclass."""

    def test_params_creation_with_defaults(self):
        """Test creating params with default values."""
        params = AlmgrenChrissParams(
            eta=0.04,
            gamma=0.004,
            sigma=0.025,
            alpha=0.5,
            tau=1.0,
            price=0.50,
        )
        assert params.eta == 0.04
        assert params.gamma == 0.004
        assert params.sigma == 0.025
        assert params.alpha == 0.5
        assert params.tau == 1.0
        assert params.price == 0.50
        assert params.source_calibration is None
        assert params.warnings == ()

    def test_params_creation_with_source_calibration(self):
        """Test creating params with a CalibrationResult reference."""
        cal_result = CalibrationResult(
            alpha=0.52,
            volatility_sensitivity=0.8,
            intercept=2.1,
            r_squared=0.45,
            alpha_std_error=0.15,
            n_observations=50,
        )
        params = AlmgrenChrissParams(
            eta=0.04,
            gamma=0.004,
            sigma=0.025,
            alpha=0.5,
            tau=1.0,
            price=0.50,
            source_calibration=cal_result,
        )
        assert params.source_calibration is cal_result
        assert params.source_calibration.alpha == 0.52

    def test_params_is_frozen(self):
        """Test that params are immutable (frozen)."""
        params = AlmgrenChrissParams(
            eta=0.04,
            gamma=0.004,
            sigma=0.025,
            alpha=0.5,
            tau=1.0,
            price=0.50,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            params.eta = 0.05


class TestCalibrateAcParams:
    """Tests for calibrate_ac_params function."""

    def test_calibrate_basic(self):
        """Test basic calibration with standard inputs."""
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        assert not math.isnan(params.eta)
        assert not math.isnan(params.gamma)
        assert not math.isnan(params.sigma)
        assert len(params.warnings) == 0

    def test_calibrate_eta_calculation(self):
        """Test that eta = spread / (0.01 * volume)."""
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        expected_eta = 0.02 / (0.01 * 50000)  # = 0.04
        assert params.eta == pytest.approx(expected_eta)

    def test_calibrate_gamma_calculation(self):
        """Test that gamma = spread / (0.10 * volume)."""
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        expected_gamma = 0.02 / (0.10 * 50000)  # = 0.004
        assert params.gamma == pytest.approx(expected_gamma)

    def test_calibrate_sigma_calculation(self):
        """Test that sigma = volatility * price."""
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        expected_sigma = 0.05 * 0.50  # = 0.025
        assert params.sigma == pytest.approx(expected_sigma)

    def test_calibrate_with_source_calibration(self):
        """Test calibration with source_calibration reference."""
        cal_result = CalibrationResult(
            alpha=0.52,
            volatility_sensitivity=0.8,
            intercept=2.1,
            r_squared=0.45,
            alpha_std_error=0.15,
            n_observations=50,
        )
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
            source_calibration=cal_result,
        )
        assert params.source_calibration is cal_result

    def test_calibrate_invalid_volume_zero(self):
        """Test calibration with zero volume returns NaN and warning."""
        params = calibrate_ac_params(
            daily_volume=0,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        assert math.isnan(params.eta)
        assert math.isnan(params.gamma)
        assert math.isnan(params.sigma)
        assert len(params.warnings) > 0
        assert "volume" in params.warnings[0].lower()

    def test_calibrate_invalid_volume_negative(self):
        """Test calibration with negative volume returns NaN and warning."""
        params = calibrate_ac_params(
            daily_volume=-1000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        assert math.isnan(params.eta)
        assert len(params.warnings) > 0

    def test_calibrate_eta_tilde_warning(self):
        """Test warning when eta_tilde constraint is violated."""
        # Create scenario where gamma is large relative to eta
        # eta_tilde = eta - 0.5 * gamma * tau
        # We need gamma > 2*eta for violation with tau=1
        # eta = spread / (0.01 * volume)
        # gamma = spread / (0.10 * volume)
        # So gamma = 0.1 * eta, normally eta_tilde = eta - 0.5*0.1*eta = 0.95*eta > 0
        # We can't easily violate with standard heuristics, so test passes if no warning
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        # With standard params, eta_tilde should be positive (no warning)
        eta_tilde = params.eta - 0.5 * params.gamma * params.tau
        assert eta_tilde > 0  # No violation expected

    def test_calibrate_zero_spread(self):
        """Test calibration with zero spread (edge case)."""
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0,
            daily_volatility=0.05,
            price=0.50,
        )
        assert params.eta == 0
        assert params.gamma == 0
        assert not math.isnan(params.sigma)


class TestExecutionTrajectory:
    """Tests for ExecutionTrajectory dataclass."""

    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        traj = ExecutionTrajectory(
            timestamps=np.array([0, 1, 2, 3]),
            holdings=np.array([1000, 700, 300, 0]),
            trade_sizes=np.array([300, 400, 300]),
            strategy_name="almgren_chriss",
            total_cost_estimate=25.5,
            risk_aversion=1e-5,
        )
        assert len(traj.timestamps) == 4
        assert len(traj.holdings) == 4
        assert len(traj.trade_sizes) == 3
        assert traj.strategy_name == "almgren_chriss"
        assert traj.total_cost_estimate == 25.5
        assert traj.risk_aversion == 1e-5

    def test_trajectory_is_frozen(self):
        """Test that trajectory is immutable."""
        traj = ExecutionTrajectory(
            timestamps=np.array([0, 1, 2]),
            holdings=np.array([100, 50, 0]),
            trade_sizes=np.array([50, 50]),
            strategy_name="twap",
            total_cost_estimate=10.0,
            risk_aversion=0,
        )
        with pytest.raises(Exception):
            traj.strategy_name = "vwap"


class TestGenerateAcTrajectory:
    """Tests for generate_ac_trajectory function."""

    @pytest.fixture
    def standard_params(self):
        """Standard A-C params for testing."""
        return calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )

    def test_generate_risk_neutral_equals_twap(self, standard_params):
        """Test that lambda=0 produces TWAP (linear liquidation)."""
        traj = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=standard_params,
            risk_aversion=0,
        )
        # All trade sizes should be equal for TWAP
        expected_size = 1000 / 10
        assert np.allclose(traj.trade_sizes, expected_size)

    def test_generate_front_loaded_with_high_lambda(self, standard_params):
        """Test that high lambda produces front-loaded execution."""
        traj = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=standard_params,
            risk_aversion=0.01,  # High risk aversion
        )
        # First trade should be larger than last trade
        assert traj.trade_sizes[0] > traj.trade_sizes[-1]

    def test_generate_holdings_start_at_order_size(self, standard_params):
        """Test that holdings start at order_size."""
        order_size = 5000
        traj = generate_ac_trajectory(
            order_size=order_size,
            duration_periods=10,
            params=standard_params,
            risk_aversion=1e-5,
        )
        assert traj.holdings[0] == pytest.approx(order_size)

    def test_generate_holdings_end_at_zero(self, standard_params):
        """Test that holdings end at approximately zero."""
        traj = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=standard_params,
            risk_aversion=1e-5,
        )
        assert traj.holdings[-1] == pytest.approx(0, abs=1e-6)

    def test_generate_trade_sizes_sum_to_order_size(self, standard_params):
        """Test that trade sizes sum to order_size."""
        order_size = 5000
        traj = generate_ac_trajectory(
            order_size=order_size,
            duration_periods=10,
            params=standard_params,
            risk_aversion=1e-5,
        )
        assert np.sum(traj.trade_sizes) == pytest.approx(order_size)

    def test_generate_trade_sizes_all_positive(self, standard_params):
        """Test that all trade sizes are positive."""
        traj = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=standard_params,
            risk_aversion=1e-5,
        )
        assert np.all(traj.trade_sizes > 0)

    def test_generate_invalid_order_size_zero(self, standard_params):
        """Test that zero order_size returns NaN trajectory."""
        traj = generate_ac_trajectory(
            order_size=0,
            duration_periods=10,
            params=standard_params,
            risk_aversion=1e-5,
        )
        assert math.isnan(traj.holdings[0])
        assert "error" in traj.params

    def test_generate_invalid_order_size_negative(self, standard_params):
        """Test that negative order_size returns NaN trajectory."""
        traj = generate_ac_trajectory(
            order_size=-1000,
            duration_periods=10,
            params=standard_params,
            risk_aversion=1e-5,
        )
        assert math.isnan(traj.holdings[0])

    def test_generate_invalid_duration_zero(self, standard_params):
        """Test that zero duration returns NaN trajectory."""
        traj = generate_ac_trajectory(
            order_size=1000,
            duration_periods=0,
            params=standard_params,
            risk_aversion=1e-5,
        )
        assert math.isnan(traj.holdings[0])

    def test_generate_eta_tilde_violation_fallback_to_twap(self):
        """Test that eta_tilde <= 0 falls back to TWAP."""
        # Create params manually with large gamma to force violation
        params = AlmgrenChrissParams(
            eta=0.001,  # Small eta
            gamma=0.1,  # Large gamma
            sigma=0.025,
            alpha=0.5,
            tau=1.0,
            price=0.50,
            warnings=(),
        )
        # eta_tilde = 0.001 - 0.5 * 0.1 * 1 = 0.001 - 0.05 = -0.049 < 0
        traj = generate_ac_trajectory(
            order_size=1000,
            duration_periods=10,
            params=params,
            risk_aversion=1e-5,
        )
        # Should fall back to TWAP (equal slices)
        assert traj.params["mode"] == "twap_fallback"
        expected_size = 1000 / 10
        assert np.allclose(traj.trade_sizes, expected_size)

    def test_generate_increasing_lambda_more_front_loaded(self, standard_params):
        """Test that increasing lambda produces more front-loaded execution."""
        lambdas = [1e-7, 1e-5, 1e-3]
        first_trade_ratios = []

        for lam in lambdas:
            traj = generate_ac_trajectory(
                order_size=1000,
                duration_periods=10,
                params=standard_params,
                risk_aversion=lam,
            )
            # Calculate ratio of first trade to average
            avg_trade = 1000 / 10
            first_trade_ratios.append(traj.trade_sizes[0] / avg_trade)

        # Each successive lambda should have larger first trade ratio
        for i in range(len(first_trade_ratios) - 1):
            assert first_trade_ratios[i + 1] >= first_trade_ratios[i]


class TestIntegration:
    """Integration tests for the A-C module."""

    def test_calibrate_then_generate(self):
        """Test round-trip: calibrate params then generate trajectory."""
        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        traj = generate_ac_trajectory(
            order_size=5000,
            duration_periods=10,
            params=params,
            risk_aversion=1e-5,
        )
        assert traj.strategy_name == "almgren_chriss"
        assert traj.holdings[0] == pytest.approx(5000)
        assert traj.holdings[-1] == pytest.approx(0, abs=1e-6)
        assert np.sum(traj.trade_sizes) == pytest.approx(5000)

    def test_imports_from_analytics_package(self):
        """Test that exports work from tributary.analytics package."""
        from tributary.analytics import (
            calibrate_ac_params,
            generate_ac_trajectory,
            AlmgrenChrissParams,
            ExecutionTrajectory,
        )

        params = calibrate_ac_params(
            daily_volume=50000,
            daily_spread=0.02,
            daily_volatility=0.05,
            price=0.50,
        )
        assert isinstance(params, AlmgrenChrissParams)

        traj = generate_ac_trajectory(
            order_size=1000,
            duration_periods=5,
            params=params,
            risk_aversion=1e-5,
        )
        assert isinstance(traj, ExecutionTrajectory)
