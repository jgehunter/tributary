---
phase: 02-cost-analytics
plan: 03
subsystem: analytics
tags: [market-impact, square-root-model, calibration, statsmodels, ols-regression]

dependency_graph:
  requires:
    - 02-01 (slippage calculation patterns)
    - 02-02 (cost forecast module structure)
  provides:
    - ImpactEstimate dataclass
    - estimate_market_impact function
    - CalibrationResult dataclass
    - calibrate_impact_parameters function
  affects:
    - Phase 3 Almgren-Chriss optimization (uses calibrated alpha)
    - Cost comparison workflows (model vs orderbook estimates)

tech_stack:
  added:
    - statsmodels>=0.14 (OLS regression for calibration)
    - pandas>=2.0 (moved to core dependencies)
    - numpy>=1.26 (moved to core dependencies)
  patterns:
    - Square-root impact model (Almgren 2005)
    - Log-transform regression for impact calibration
    - Confidence-based output (HIGH/MEDIUM/LOW participation thresholds)

key_files:
  created:
    - src/tributary/analytics/impact.py
    - tests/unit/test_impact.py
  modified:
    - src/tributary/analytics/__init__.py
    - pyproject.toml

decisions:
  - pattern: "Participation rate thresholds"
    choice: "HIGH (<1%), MEDIUM (1-10%), LOW (>10%)"
    rationale: "Follows standard institutional trading thresholds; >10% typically overwhelms thin orderbooks"
  - pattern: "Permanent impact ratio"
    choice: "40% of temporary (fixed default)"
    rationale: "Conservative default for thin markets; equity literature shows 30-50% range"
  - pattern: "Boundary conditions"
    choice: "Exactly 1% stays HIGH (exclusive boundary)"
    rationale: "Consistent with standard interval notation [0, 0.01) = HIGH"

metrics:
  duration: "6m"
  completed: "2026-01-19"
  tests_added: 31
  tests_passing: 31
---

# Phase 2 Plan 3: Market Impact Estimation Summary

Square-root impact model with temp/perm decomposition and parameter calibration from historical data using statsmodels OLS.

## What Was Built

### ImpactEstimate Dataclass
```python
@dataclass
class ImpactEstimate:
    temporary_impact_bps: float   # Reverts after execution
    permanent_impact_bps: float   # Persists (information content)
    total_impact_bps: float       # temporary + permanent + half-spread
    confidence: str               # 'HIGH', 'MEDIUM', 'LOW'
    notes: List[str]              # Warnings and context
```

### estimate_market_impact Function
- Implements square-root law: `impact = volatility * (participation)^alpha * 10000`
- Default alpha=0.5 (square-root), configurable up to 0.6
- Permanent impact = 40% of temporary (conservative for thin markets)
- Adds half-spread as execution cost
- Confidence levels based on participation rate:
  - HIGH: < 1%
  - MEDIUM: 1-10%
  - LOW: > 10% (model unreliable)

### CalibrationResult Dataclass
```python
@dataclass
class CalibrationResult:
    alpha: float                  # Impact exponent
    volatility_sensitivity: float # Beta for volatility term
    intercept: float              # Regression constant
    r_squared: float              # Model fit quality
    alpha_std_error: float        # Uncertainty in alpha
    n_observations: int           # Data points used
    warnings: List[str]           # Data quality warnings
```

### calibrate_impact_parameters Function
- Uses statsmodels OLS regression on log-transformed data
- `log(impact) = c + alpha*log(participation) + beta*log(volatility)`
- Filters zero/negative values before regression
- Requires minimum 10 observations
- Generates warnings for:
  - `alpha_std_error > 0.3` (unreliable estimate)
  - `r_squared < 0.3` (poor model fit)
  - `n_observations < 30` (limited data)

## Key Design Decisions

### Secondary Validation Pattern
The module includes prominent warnings that the square-root model is calibrated for equity markets with deep liquidity. For prediction markets (Polymarket), orderbook-based estimation should be the PRIMARY method. This model serves as a sanity check, not truth.

### Confidence Thresholds
Standard institutional trading thresholds were adopted:
- < 1%: Model is reliable, fits historical equity data well
- 1-10%: Use with caution, impact uncertainty increases
- > 10%: Model breaks down; thin orderbooks get overwhelmed

### Permanent Impact Ratio
Used 40% as conservative default. Equity research shows 30-50% range. For thin liquidity markets, permanent impact tends to be higher (more information content in trades).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] statsmodels column name compatibility**
- **Found during:** Task 2 implementation
- **Issue:** `sm.add_constant()` creates column names that vary by pandas version
- **Fix:** Explicitly create 'const' column instead of using `sm.add_constant()`
- **Files modified:** `src/tributary/analytics/impact.py`
- **Commit:** f6b5ea4

## Requirements Satisfied

| Requirement | Description | Status |
|-------------|-------------|--------|
| COST-06 | Estimate temporary market impact from historical data | SATISFIED |
| COST-07 | Estimate permanent market impact from historical data | SATISFIED |

## Test Coverage

31 unit tests covering:
- ImpactEstimate dataclass creation and defaults
- estimate_market_impact with various participation rates
- Confidence level thresholds (HIGH/MEDIUM/LOW)
- Edge cases (zero volume, negative inputs, boundary conditions)
- Temporary vs permanent ratio validation
- Alpha parameter effects (0.5 vs 0.6)
- Spread component added correctly
- Total equals sum of components
- CalibrationResult dataclass creation
- calibrate_impact_parameters with synthetic data
- Insufficient data handling (< 10 observations)
- Zero value filtering
- Warning generation for poor fit
- Warning generation for limited data
- Warning generation for high std_error
- Integration tests (round-trip, imports)

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/tributary/analytics/impact.py` | Created | 329 |
| `tests/unit/test_impact.py` | Created | 544 |
| `src/tributary/analytics/__init__.py` | Modified | +20 |
| `pyproject.toml` | Modified | +5 |

## Commits

| Hash | Message |
|------|---------|
| f6b5ea4 | feat(02-03): add market impact estimation module |

## Next Phase Readiness

Phase 2 now has all core analytics components:
- 02-01: Slippage calculation, implementation shortfall
- 02-02: Orderbook-based cost forecasting (PRIMARY)
- 02-03: Model-based impact estimation (SECONDARY)

Ready for Phase 3: Almgren-Chriss optimal execution using calibrated alpha parameter from this module.

## Usage Example

```python
from tributary.analytics import (
    estimate_market_impact,
    calibrate_impact_parameters,
)

# Estimate impact for a single order
estimate = estimate_market_impact(
    order_size=1000,
    daily_volume=100000,
    volatility=0.02,  # 2% daily volatility
    spread_bps=50,
)
print(f"Total impact: {estimate.total_impact_bps:.1f} bps")
print(f"Confidence: {estimate.confidence}")
# Output: Total impact: 53.0 bps, Confidence: HIGH

# Calibrate from historical data
import pandas as pd
executions = pd.DataFrame({
    'order_size': [...],
    'daily_volume': [...],
    'realized_impact_bps': [...],
    'volatility': [...]
})
result = calibrate_impact_parameters(executions)
print(f"Calibrated alpha: {result.alpha:.2f}")
```
