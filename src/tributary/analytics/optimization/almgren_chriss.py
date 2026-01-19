"""Almgren-Chriss optimal execution framework.

This module implements the Almgren-Chriss (2001) optimal execution model, which
balances market impact costs against timing risk through mean-variance optimization.

The model solves for the optimal execution trajectory that minimizes:
    E[Cost] + lambda * Var[Cost]

Where:
    - lambda is risk aversion (0 = risk-neutral/TWAP, higher = more front-loaded)
    - Cost includes temporary impact (from trading rate) and permanent impact
    - Variance captures timing risk from price volatility during execution

Key insight: Risk-averse traders front-load execution to reduce exposure to
adverse price movements, while risk-neutral traders execute linearly (TWAP).

Key Functions:
    calibrate_ac_params: Derive A-C parameters from market data
    generate_ac_trajectory: Compute optimal holdings/trade sizes

Requirements Satisfied:
    OPT-01: Almgren-Chriss parameter calibration
    OPT-02: Optimal execution trajectory generation
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tributary.analytics.impact import CalibrationResult


@dataclass(frozen=True)
class AlmgrenChrissParams:
    """Calibrated parameters for Almgren-Chriss optimal execution.

    Attributes:
        eta: Temporary impact coefficient (cost per unit traded per time)
        gamma: Permanent impact coefficient (persistent price move per unit)
        sigma: Volatility (price standard deviation per period)
        alpha: Impact exponent from square-root model (typically 0.5)
        tau: Time interval between trades (default 1.0)
        price: Reference price used for calibration (for cost conversion)
        source_calibration: Optional reference to CalibrationResult from impact.py
        warnings: Tuple of validation/calibration warnings (tuple for frozen)
    """

    eta: float
    gamma: float
    sigma: float
    alpha: float
    tau: float
    price: float
    source_calibration: Optional[CalibrationResult] = None
    warnings: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionTrajectory:
    """Optimal execution trajectory from any strategy.

    Attributes:
        timestamps: Time points (0, 1, 2, ..., T) or actual timestamps
        holdings: Remaining position at each time (starts at order_size, ends at 0)
        trade_sizes: Size to execute at each interval (length = T)
        strategy_name: 'almgren_chriss', 'twap', 'vwap', or 'market_order'
        total_cost_estimate: Expected total cost in basis points
        risk_aversion: Lambda parameter (0 for TWAP, inf for market order)
        params: Optional strategy-specific parameters (dict)
    """

    timestamps: np.ndarray
    holdings: np.ndarray
    trade_sizes: np.ndarray
    strategy_name: str
    total_cost_estimate: float
    risk_aversion: float
    params: Optional[dict] = None


def calibrate_ac_params(
    daily_volume: float,
    daily_spread: float,
    daily_volatility: float,
    price: float,
    alpha: float = 0.5,
    tau: float = 1.0,
    source_calibration: Optional[CalibrationResult] = None,
) -> AlmgrenChrissParams:
    """
    Calibrate Almgren-Chriss parameters from market data.

    Uses standard A-C heuristics:
    - eta: Trading 1% of daily volume causes temporary impact of full spread
    - gamma: Trading 10% of daily volume causes permanent impact of full spread

    Args:
        daily_volume: Average daily trading volume (in shares/contracts)
        daily_spread: Average bid-ask spread in price units (e.g., $0.02)
        daily_volatility: Daily volatility as decimal (e.g., 0.05 for 5%)
        price: Current asset price (used for absolute volatility calculation)
        alpha: Impact exponent (default 0.5 from square-root model)
        tau: Time interval between trades (default 1.0)
        source_calibration: Optional reference to existing CalibrationResult

    Returns:
        AlmgrenChrissParams with calibrated values and any warnings

    Example:
        >>> params = calibrate_ac_params(
        ...     daily_volume=50000,
        ...     daily_spread=0.02,
        ...     daily_volatility=0.05,
        ...     price=0.50,
        ... )
        >>> print(f"eta={params.eta:.6f}, gamma={params.gamma:.6f}")
    """
    warnings = []

    # Validate inputs
    if daily_volume <= 0:
        warnings.append(f"Invalid daily_volume ({daily_volume}): must be > 0")
        return AlmgrenChrissParams(
            eta=float("nan"),
            gamma=float("nan"),
            sigma=float("nan"),
            alpha=alpha,
            tau=tau,
            price=price,
            source_calibration=source_calibration,
            warnings=tuple(warnings),
        )

    if daily_spread < 0:
        warnings.append(f"Invalid daily_spread ({daily_spread}): must be >= 0")
        return AlmgrenChrissParams(
            eta=float("nan"),
            gamma=float("nan"),
            sigma=float("nan"),
            alpha=alpha,
            tau=tau,
            price=price,
            source_calibration=source_calibration,
            warnings=tuple(warnings),
        )

    if daily_volatility < 0:
        warnings.append(f"Invalid daily_volatility ({daily_volatility}): must be >= 0")
        return AlmgrenChrissParams(
            eta=float("nan"),
            gamma=float("nan"),
            sigma=float("nan"),
            alpha=alpha,
            tau=tau,
            price=price,
            source_calibration=source_calibration,
            warnings=tuple(warnings),
        )

    if price <= 0:
        warnings.append(f"Invalid price ({price}): must be > 0")
        return AlmgrenChrissParams(
            eta=float("nan"),
            gamma=float("nan"),
            sigma=float("nan"),
            alpha=alpha,
            tau=tau,
            price=price,
            source_calibration=source_calibration,
            warnings=tuple(warnings),
        )

    # Temporary impact: 1% ADV causes temp impact of full spread
    # eta = spread / (0.01 * daily_volume)
    eta = daily_spread / (0.01 * daily_volume)

    # Permanent impact: 10% ADV causes perm impact of full spread
    # gamma = spread / (0.10 * daily_volume)
    gamma = daily_spread / (0.10 * daily_volume)

    # Volatility in absolute price units (not percentage)
    sigma = daily_volatility * price

    # Validate eta_tilde constraint: eta_tilde = eta - 0.5 * gamma * tau > 0
    eta_tilde = eta - 0.5 * gamma * tau
    if eta_tilde <= 0:
        warnings.append(
            f"eta_tilde = {eta_tilde:.6f} <= 0: model constraints violated. "
            "Trajectory generation will fall back to TWAP."
        )

    return AlmgrenChrissParams(
        eta=eta,
        gamma=gamma,
        sigma=sigma,
        alpha=alpha,
        tau=tau,
        price=price,
        source_calibration=source_calibration,
        warnings=tuple(warnings),
    )


def generate_ac_trajectory(
    order_size: float,
    duration_periods: int,
    params: AlmgrenChrissParams,
    risk_aversion: float = 1e-6,
) -> ExecutionTrajectory:
    """
    Generate optimal execution trajectory using Almgren-Chriss solution.

    Computes the optimal trading trajectory that minimizes:
        E[cost] + lambda * Var[cost]

    The solution uses the hyperbolic sinh/cosh formula:
        holdings[t] = order_size * sinh(kappa * (T - t)) / sinh(kappa * T)

    Where kappa is the decay rate determined by risk aversion and market parameters.

    Special Cases:
        - risk_aversion = 0: Degenerates to TWAP (linear liquidation)
        - risk_aversion -> inf: Approaches immediate execution (market order)

    Args:
        order_size: Total position to liquidate (positive for sell orders)
        duration_periods: Number of trading intervals (T)
        params: Calibrated AlmgrenChrissParams
        risk_aversion: Lambda parameter for risk-return tradeoff (default 1e-6)

    Returns:
        ExecutionTrajectory with optimal holdings, trade sizes, and cost estimate

    Notes:
        - Holdings start at order_size and end at 0
        - Trade sizes are all positive (representing sell orders)
        - Total cost estimate is in basis points
        - If eta_tilde <= 0, falls back to TWAP with warning

    Example:
        >>> params = calibrate_ac_params(50000, 0.02, 0.05, 0.50)
        >>> traj = generate_ac_trajectory(
        ...     order_size=5000,
        ...     duration_periods=10,
        ...     params=params,
        ...     risk_aversion=1e-5,
        ... )
        >>> print(f"First trade: {traj.trade_sizes[0]:.0f}")
    """
    # Handle invalid inputs
    if order_size <= 0:
        return ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="almgren_chriss",
            total_cost_estimate=float("nan"),
            risk_aversion=risk_aversion,
            params={"error": "Invalid order size (<= 0)"},
        )

    if duration_periods <= 0:
        return ExecutionTrajectory(
            timestamps=np.array([np.nan]),
            holdings=np.array([np.nan]),
            trade_sizes=np.array([np.nan]),
            strategy_name="almgren_chriss",
            total_cost_estimate=float("nan"),
            risk_aversion=risk_aversion,
            params={"error": "Invalid duration (<= 0)"},
        )

    T = duration_periods
    tau = params.tau
    timestamps = np.arange(T + 1, dtype=float)

    # Check for risk-neutral case (TWAP)
    if risk_aversion == 0 or risk_aversion < 1e-10:
        # Linear liquidation: holdings[t] = order_size * (1 - t/T)
        holdings = order_size * (1 - timestamps / T)
        trade_sizes = np.full(T, order_size / T)

        # Estimate cost (temporary impact only for TWAP)
        # Cost = eta * sum(n_i^2) / tau where n_i = trade_sizes[i]
        temp_cost = params.eta * np.sum(trade_sizes**2) / tau
        perm_cost = 0.5 * params.gamma * order_size**2

        # Convert to bps
        if params.price > 0:
            total_cost_bps = (temp_cost + perm_cost) / (order_size * params.price) * 10000
        else:
            total_cost_bps = float("nan")

        return ExecutionTrajectory(
            timestamps=timestamps,
            holdings=holdings,
            trade_sizes=trade_sizes,
            strategy_name="almgren_chriss",
            total_cost_estimate=total_cost_bps,
            risk_aversion=risk_aversion,
            params={"mode": "twap_fallback", "reason": "risk_neutral"},
        )

    # Compute adjusted temporary impact coefficient
    eta_tilde = params.eta - 0.5 * params.gamma * tau

    # If eta_tilde <= 0, fall back to TWAP
    if eta_tilde <= 0:
        holdings = order_size * (1 - timestamps / T)
        trade_sizes = np.full(T, order_size / T)

        temp_cost = params.eta * np.sum(trade_sizes**2) / tau
        perm_cost = 0.5 * params.gamma * order_size**2

        if params.price > 0:
            total_cost_bps = (temp_cost + perm_cost) / (order_size * params.price) * 10000
        else:
            total_cost_bps = float("nan")

        return ExecutionTrajectory(
            timestamps=timestamps,
            holdings=holdings,
            trade_sizes=trade_sizes,
            strategy_name="almgren_chriss",
            total_cost_estimate=total_cost_bps,
            risk_aversion=risk_aversion,
            params={
                "mode": "twap_fallback",
                "reason": "eta_tilde_violation",
                "eta_tilde": eta_tilde,
            },
        )

    # Compute kappa (decay rate)
    # kappa_tilde^2 = (lambda * sigma^2) / eta_tilde
    kappa_tilde_sq = (risk_aversion * params.sigma**2) / eta_tilde

    # kappa = arccosh(1 + 0.5 * kappa_tilde^2 * tau^2) / tau
    cosh_arg = 0.5 * kappa_tilde_sq * tau**2 + 1
    kappa = np.arccosh(cosh_arg) / tau

    # Generate optimal trajectory using sinh formula
    # holdings[t] = X * sinh(kappa * (T - t)) / sinh(kappa * T)
    sinh_kT = np.sinh(kappa * T)

    # Avoid division by zero for very small kappa*T
    if abs(sinh_kT) < 1e-10:
        # Fall back to TWAP for numerical stability
        holdings = order_size * (1 - timestamps / T)
    else:
        holdings = order_size * np.sinh(kappa * (T - timestamps)) / sinh_kT

    # Compute trade sizes as negative of holdings differences
    trade_sizes = -np.diff(holdings)

    # Estimate total cost using A-C formula
    # E[cost] = 0.5 * gamma * X^2 + eta_tilde * sum(n_i^2) / tau
    temp_cost = eta_tilde * np.sum(trade_sizes**2) / tau
    perm_cost = 0.5 * params.gamma * order_size**2

    if params.price > 0:
        total_cost_bps = (temp_cost + perm_cost) / (order_size * params.price) * 10000
    else:
        total_cost_bps = float("nan")

    return ExecutionTrajectory(
        timestamps=timestamps,
        holdings=holdings,
        trade_sizes=trade_sizes,
        strategy_name="almgren_chriss",
        total_cost_estimate=total_cost_bps,
        risk_aversion=risk_aversion,
        params={
            "kappa": kappa,
            "eta_tilde": eta_tilde,
            "sinh_kT": sinh_kT,
        },
    )
