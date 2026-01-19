"""Market impact estimation using square-root model.

This module provides model-based market impact forecasting as SECONDARY validation
for orderbook-based estimates. The square-root law (Almgren et al. 2005) is
well-established for equity markets with deep liquidity.

WARNING: These models are calibrated for equity markets. For thin liquidity markets
like prediction markets (Polymarket, Kalshi), orderbook-based estimation should be
used as the PRIMARY method. Model-based impact serves as a sanity check, not truth.

Key Functions:
    estimate_market_impact: Estimate temp/perm impact from order parameters
    calibrate_impact_parameters: Fit model to historical execution data

Requirements Satisfied:
    COST-06: Estimate temporary market impact from historical data
    COST-07: Estimate permanent market impact from historical data
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class ImpactEstimate:
    """Result of market impact estimation.

    Attributes:
        temporary_impact_bps: Temporary impact in basis points (reverts after execution)
        permanent_impact_bps: Permanent impact in basis points (persists - information content)
        total_impact_bps: Total expected impact (temporary + permanent + half-spread)
        confidence: Confidence level ('HIGH', 'MEDIUM', 'LOW') based on participation rate
        notes: List of warnings and contextual information

    Example:
        >>> estimate = ImpactEstimate(
        ...     temporary_impact_bps=15.5,
        ...     permanent_impact_bps=6.2,
        ...     total_impact_bps=46.7,
        ...     confidence='MEDIUM',
        ...     notes=['Elevated participation rate (1-10%): use with caution']
        ... )
    """

    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    confidence: str
    notes: List[str] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Result of impact model parameter calibration.

    Contains the fitted parameters from OLS regression on historical execution data.

    Attributes:
        alpha: Impact exponent (0.5 = square-root law)
        volatility_sensitivity: Beta coefficient for volatility term
        intercept: Regression intercept (constant term)
        r_squared: Model fit quality (0-1)
        alpha_std_error: Standard error of alpha estimate
        n_observations: Number of data points used in calibration
        warnings: List of data quality warnings

    Example:
        >>> result = CalibrationResult(
        ...     alpha=0.52,
        ...     volatility_sensitivity=0.8,
        ...     intercept=2.1,
        ...     r_squared=0.45,
        ...     alpha_std_error=0.15,
        ...     n_observations=50,
        ...     warnings=[]
        ... )
    """

    alpha: float
    volatility_sensitivity: float
    intercept: float
    r_squared: float
    alpha_std_error: float
    n_observations: int
    warnings: List[str] = field(default_factory=list)


def estimate_market_impact(
    order_size: float,
    daily_volume: float,
    volatility: float,
    spread_bps: float,
    alpha: float = 0.5,
) -> ImpactEstimate:
    """
    Estimate market impact using the square-root model.

    Based on Almgren et al. (2005) empirical findings for equity markets.

    WARNING: This model is calibrated for equity markets with deep liquidity.
    For thin liquidity markets (prediction markets, small-cap crypto),
    use orderbook-based estimation as the PRIMARY method. This model
    should be used as SECONDARY validation only.

    Formula:
        temporary_impact = volatility * (order_size / daily_volume)^alpha * 10000
        permanent_impact = temporary_impact * permanent_ratio (default 40%)
        total_impact = temporary + permanent + half_spread

    Args:
        order_size: Total size of the order to execute
        daily_volume: Average daily volume for the instrument
        volatility: Daily volatility as a decimal (e.g., 0.02 for 2%)
        spread_bps: Current bid-ask spread in basis points
        alpha: Impact exponent, default 0.5 for square-root law
            (typical range 0.5-0.6 in empirical studies)

    Returns:
        ImpactEstimate with temporary, permanent, and total impact in bps,
        confidence level, and any warnings

    Confidence Levels:
        - HIGH: participation rate < 1%
        - MEDIUM: participation rate 1-10%
        - LOW: participation rate > 10% (model unreliable)

    Example:
        >>> estimate = estimate_market_impact(
        ...     order_size=1000,
        ...     daily_volume=100000,
        ...     volatility=0.02,
        ...     spread_bps=50
        ... )
        >>> print(f"Total impact: {estimate.total_impact_bps:.1f} bps")
        Total impact: 62.8 bps
    """
    notes: List[str] = []

    # Handle invalid inputs
    if daily_volume <= 0:
        return ImpactEstimate(
            temporary_impact_bps=float("nan"),
            permanent_impact_bps=float("nan"),
            total_impact_bps=float("nan"),
            confidence="LOW",
            notes=["Invalid daily volume (<= 0)"],
        )

    if order_size < 0:
        return ImpactEstimate(
            temporary_impact_bps=float("nan"),
            permanent_impact_bps=float("nan"),
            total_impact_bps=float("nan"),
            confidence="LOW",
            notes=["Invalid order size (< 0)"],
        )

    # Handle zero order size
    if order_size == 0:
        return ImpactEstimate(
            temporary_impact_bps=0.0,
            permanent_impact_bps=0.0,
            total_impact_bps=spread_bps / 2,  # Just the spread cost
            confidence="HIGH",
            notes=["Zero order size: only spread cost applies"],
        )

    # Calculate participation rate
    participation_rate = order_size / daily_volume

    # Determine confidence based on participation rate
    if participation_rate > 0.10:
        confidence = "LOW"
        notes.append(
            "High participation rate (>10%): model unreliable for thin liquidity"
        )
    elif participation_rate > 0.01:
        confidence = "MEDIUM"
        notes.append("Elevated participation rate (1-10%): use with caution")
    else:
        confidence = "HIGH"

    # Square-root model for temporary impact
    # Impact = sigma * (Q/V)^alpha * 10000 (convert to bps)
    temporary_impact_bps = volatility * (participation_rate**alpha) * 10000

    # Permanent impact: typically 30-50% of temporary for equities
    # Use 40% as conservative default for thin markets
    permanent_ratio = 0.4
    permanent_impact_bps = temporary_impact_bps * permanent_ratio

    # Add half-spread as execution cost
    half_spread_bps = spread_bps / 2

    total_impact_bps = temporary_impact_bps + permanent_impact_bps + half_spread_bps

    return ImpactEstimate(
        temporary_impact_bps=temporary_impact_bps,
        permanent_impact_bps=permanent_impact_bps,
        total_impact_bps=total_impact_bps,
        confidence=confidence,
        notes=notes,
    )


def calibrate_impact_parameters(executions_df: pd.DataFrame) -> CalibrationResult:
    """
    Calibrate square-root impact model parameters from historical executions.

    Uses OLS regression on log-transformed data to estimate the impact exponent (alpha)
    and volatility sensitivity from actual execution data.

    Regression model:
        log(realized_impact_bps) = c + alpha * log(participation_rate) + beta * log(volatility)

    Where:
        - participation_rate = order_size / daily_volume
        - alpha is the impact exponent (0.5 = square-root law)
        - beta is the volatility sensitivity

    Args:
        executions_df: DataFrame with required columns:
            - order_size: Size of each executed order
            - daily_volume: Average daily volume at time of execution
            - realized_impact_bps: Actual impact observed in basis points
            - volatility: Daily volatility (as decimal) at time of execution

    Returns:
        CalibrationResult with fitted parameters and diagnostics

    Raises:
        ValueError: If required columns are missing from DataFrame

    Warnings Generated:
        - "Insufficient data": n_observations < 30
        - "High parameter uncertainty": alpha_std_error > 0.3
        - "Poor model fit": r_squared < 0.3

    Example:
        >>> df = pd.DataFrame({
        ...     'order_size': [100, 200, 300, 400, 500] * 10,
        ...     'daily_volume': [10000] * 50,
        ...     'realized_impact_bps': [5, 7, 9, 10, 11] * 10,
        ...     'volatility': [0.02] * 50
        ... })
        >>> result = calibrate_impact_parameters(df)
        >>> print(f"Alpha: {result.alpha:.2f} +/- {result.alpha_std_error:.2f}")
    """
    required_columns = {"order_size", "daily_volume", "realized_impact_bps", "volatility"}
    missing_columns = required_columns - set(executions_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = executions_df.copy()
    warnings: List[str] = []

    # Filter out invalid values
    df = df[df["order_size"] > 0]
    df = df[df["daily_volume"] > 0]
    df = df[df["realized_impact_bps"] > 0]
    df = df[df["volatility"] > 0]

    # Calculate participation rate
    df["participation_rate"] = df["order_size"] / df["daily_volume"]

    # Minimum observations check
    if len(df) < 10:
        return CalibrationResult(
            alpha=float("nan"),
            volatility_sensitivity=float("nan"),
            intercept=float("nan"),
            r_squared=float("nan"),
            alpha_std_error=float("nan"),
            n_observations=len(df),
            warnings=["Insufficient data (need 10+ observations after filtering)"],
        )

    # Log-transform for regression
    df["log_impact"] = np.log(df["realized_impact_bps"])
    df["log_participation"] = np.log(df["participation_rate"])
    df["log_volatility"] = np.log(df["volatility"])

    # Build regression matrix with explicit column names
    X = df[["log_participation", "log_volatility"]].copy()
    X.insert(0, "const", 1.0)  # Add constant column explicitly
    y = df["log_impact"]

    # Fit OLS model
    model = sm.OLS(y, X).fit()

    # Extract parameters using column names
    alpha = model.params["log_participation"]
    volatility_sensitivity = model.params["log_volatility"]
    intercept = model.params["const"]
    r_squared = model.rsquared
    alpha_std_error = model.bse["log_participation"]
    n_observations = len(df)

    # Generate warnings for data quality issues
    if n_observations < 30:
        warnings.append(f"Limited data ({n_observations} observations): results may be unstable")

    if alpha_std_error > 0.3:
        warnings.append(
            f"High parameter uncertainty (alpha std error: {alpha_std_error:.2f}): "
            "estimate unreliable"
        )

    if r_squared < 0.3:
        warnings.append(
            f"Poor model fit (R-squared: {r_squared:.2f}): "
            "impact may not follow square-root law"
        )

    return CalibrationResult(
        alpha=alpha,
        volatility_sensitivity=volatility_sensitivity,
        intercept=intercept,
        r_squared=r_squared,
        alpha_std_error=alpha_std_error,
        n_observations=n_observations,
        warnings=warnings,
    )
