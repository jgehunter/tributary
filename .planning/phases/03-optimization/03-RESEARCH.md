# Phase 3: Optimization - Research

**Researched:** 2026-01-19
**Domain:** Optimal execution algorithms, trade scheduling, execution strategy comparison
**Confidence:** HIGH

## Summary

Phase 3 implements optimal execution trajectory generation using the Almgren-Chriss framework alongside baseline execution strategies (TWAP, VWAP, market order). The project already has the foundational components from Phase 2: the square-root impact model (`impact.py`), calibration infrastructure (`CalibrationResult`), and benchmark calculations (`benchmarks.py`).

The Almgren-Chriss model is the standard academic framework for optimal execution, balancing market impact costs against timing risk through a mean-variance optimization. The key insight is that the optimal trajectory depends on risk aversion: aggressive traders front-load execution to reduce price uncertainty, while risk-neutral traders execute linearly (TWAP).

For this project's prediction market context, the existing `calibrate_impact_parameters()` function already provides the `alpha` (impact exponent) via OLS regression. The Almgren-Chriss implementation needs additional parameters (`eta` for temporary impact, `gamma` for permanent impact, `sigma` for volatility) which can be derived from the same historical data.

**Primary recommendation:** Build a pure-Python Almgren-Chriss implementation using NumPy (no external package dependencies), leveraging the existing `CalibrationResult` from `impact.py` and extending it with the additional parameters needed for trajectory optimization.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=1.26 | Trajectory calculation, matrix math | Already in project, hyperbolic functions needed |
| pandas | >=2.0 | Time series, volume profiles | Already in project |
| scipy | >=1.11 | Optimization (optional, for advanced scheduling) | Already available via notebooks optional |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statsmodels | >=0.14 | Parameter regression (already used) | Impact calibration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom implementation | `almgren-chriss` PyPI package | Package is GPL-licensed, limited maintenance, custom impl gives full control |
| NumPy-only | scipy.optimize.minimize | Only needed for complex constraint optimization, not basic A-C |

**Installation:**
No new dependencies required. All needed libraries already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure
```
src/tributary/analytics/
├── impact.py           # Existing: CalibrationResult, estimate_market_impact()
├── cost_forecast.py    # Existing: walk-the-book estimation
├── benchmarks.py       # Existing: VWAP, TWAP calculation
├── slippage.py         # Existing: slippage calculation
├── shortfall.py        # Existing: implementation shortfall decomposition
├── optimization/       # NEW: Phase 3 module
│   ├── __init__.py
│   ├── almgren_chriss.py  # OPT-01, OPT-02: A-C calibration and trajectories
│   ├── strategies.py      # OPT-03, OPT-04, OPT-05: TWAP, VWAP, market order
│   ├── scheduler.py       # OPT-06: Trade scheduling optimizer
│   └── comparison.py      # Strategy output comparison
└── reader.py           # Existing: QuestDB data access
```

### Pattern 1: Dataclass-Based Results (Consistent with Phase 2)
**What:** Use frozen dataclasses for all result types, matching existing patterns
**When to use:** All function return types for trajectories, schedules, comparisons
**Example:**
```python
# Source: Project convention from Phase 2
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass(frozen=True)
class ExecutionTrajectory:
    """Optimal execution trajectory from Almgren-Chriss or baseline strategy.

    Attributes:
        timestamps: Execution time points
        holdings: Remaining position at each time (starts at order_size, ends at 0)
        trade_sizes: Size to execute at each interval
        strategy_name: 'almgren_chriss', 'twap', 'vwap', or 'market_order'
        total_cost_estimate: Expected total cost in basis points
        risk_aversion: Lambda parameter (0 for TWAP, inf for market order)
    """
    timestamps: np.ndarray
    holdings: np.ndarray
    trade_sizes: np.ndarray
    strategy_name: str
    total_cost_estimate: float
    risk_aversion: float
```

### Pattern 2: Parameter Calibration Extension
**What:** Extend existing `CalibrationResult` or create companion dataclass for A-C specific params
**When to use:** When calibrating Almgren-Chriss parameters from historical data
**Example:**
```python
# Source: Extension of existing impact.py pattern
@dataclass
class AlmgrenChrissParams:
    """Calibrated parameters for Almgren-Chriss optimal execution.

    Attributes:
        eta: Temporary impact coefficient ($/share per share/time)
        gamma: Permanent impact coefficient ($/share per share)
        sigma: Daily volatility ($/share)
        alpha: Impact exponent from square-root model (typically 0.5)
        tau: Time interval between trades (in trading periods)
        source_calibration: Reference to underlying CalibrationResult
    """
    eta: float
    gamma: float
    sigma: float
    alpha: float
    tau: float
    source_calibration: CalibrationResult
    warnings: List[str] = field(default_factory=list)
```

### Pattern 3: Strategy Protocol (Duck Typing)
**What:** Common interface for all execution strategies
**When to use:** Strategy comparison, backtesting preparation
**Example:**
```python
# Source: Python Protocol pattern for strategy abstraction
from typing import Protocol

class ExecutionStrategy(Protocol):
    """Protocol for execution strategies."""

    def generate_trajectory(
        self,
        order_size: float,
        duration_periods: int,
        side: str,
    ) -> ExecutionTrajectory:
        """Generate execution trajectory for the given order."""
        ...
```

### Anti-Patterns to Avoid
- **Hardcoded parameters:** Always derive eta, gamma, sigma from data; don't use equity-market defaults for prediction markets
- **Single risk-aversion value:** Expose lambda as a parameter, don't bake in assumptions
- **Ignoring position sign:** Respect buy vs sell direction throughout (positive slippage = cost convention)
- **Mixing time units:** Be explicit about whether duration is in minutes, hours, or periods

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hyperbolic trajectory | Manual sinh/cosh calculation | `np.sinh()`, `np.cosh()` | Numerical stability, edge cases |
| VWAP calculation | Custom aggregation | Existing `calculate_vwap()` in benchmarks.py | Already implemented and tested |
| TWAP calculation | Custom time weighting | Existing `calculate_twap()` in benchmarks.py | Already implemented and tested |
| Impact calibration | New regression | Existing `calibrate_impact_parameters()` | Alpha already calibrated, extend don't replace |
| Volatility estimation | Simple std dev | Parkinson or Garman-Klass estimator | Captures intraday moves better |

**Key insight:** Phase 2 already provides VWAP/TWAP *benchmarks* (measuring achieved prices). Phase 3 needs VWAP/TWAP *strategies* (generating schedules to achieve those prices). These are different: benchmarks look backward, strategies look forward.

## Common Pitfalls

### Pitfall 1: Confusing Benchmark vs Strategy
**What goes wrong:** Using `calculate_vwap()` to generate execution schedules
**Why it happens:** Same names used for different concepts
**How to avoid:**
- Benchmarks (Phase 2): Calculate price from historical trades
- Strategies (Phase 3): Generate future trade schedules
- Naming: `vwap_benchmark()` vs `vwap_strategy()` or `generate_vwap_trajectory()`
**Warning signs:** Strategy function takes historical trades as input instead of volume profile

### Pitfall 2: Invalid Almgren-Chriss Parameters
**What goes wrong:** Model produces NaN or negative trajectories
**Why it happens:** eta_tilde = eta - 0.5 * gamma * tau must be > 0
**How to avoid:**
- Validate: `eta_tilde = eta - 0.5 * gamma * tau; assert eta_tilde > 0`
- If violated, fall back to TWAP (risk-neutral solution)
**Warning signs:** `sinh(kappa * T)` approaching zero, divide-by-zero warnings

### Pitfall 3: Parameter Scale Mismatch
**What goes wrong:** Trajectories are absurdly aggressive or passive
**Why it happens:** Parameters estimated with wrong units (bps vs decimal, daily vs per-period)
**How to avoid:**
- Document units explicitly in docstrings
- Validate parameters produce sensible trajectories for test cases
- Compare against TWAP baseline (A-C should be between TWAP and market order)
**Warning signs:** All trades in first interval, or trajectory barely differs from TWAP

### Pitfall 4: VWAP Strategy Without Volume Profile
**What goes wrong:** VWAP strategy doesn't match actual volume patterns
**Why it happens:** Using uniform distribution instead of historical volume profile
**How to avoid:**
- Query historical volume profile from QuestDB
- Use `query_vwap_sampled()` to get hourly/minute volume distributions
- Weight slice sizes by predicted volume
**Warning signs:** VWAP strategy looks identical to TWAP

### Pitfall 5: Sign Convention Inconsistency
**What goes wrong:** Buy orders show negative costs, sell orders show positive gains
**Why it happens:** Mixing conventions between modules
**How to avoid:**
- Maintain project convention: positive slippage = cost
- All strategies should produce costs that are comparable
- Document: "A lower total_cost_estimate is better for both buy and sell"
**Warning signs:** Comparing strategies and market order appears cheapest

## Code Examples

Verified patterns from research:

### Almgren-Chriss Trajectory Calculation
```python
# Source: QuantJourney implementation + academic papers
import numpy as np
from dataclasses import dataclass
from typing import List

def calculate_ac_trajectory(
    order_size: float,
    duration_periods: int,
    eta: float,
    gamma: float,
    sigma: float,
    risk_aversion: float,
    tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate optimal Almgren-Chriss execution trajectory.

    Args:
        order_size: Total shares/contracts to execute (X_0)
        duration_periods: Number of trading periods (T)
        eta: Temporary impact coefficient
        gamma: Permanent impact coefficient
        sigma: Volatility (per period)
        risk_aversion: Lambda - higher = more front-loaded
        tau: Time between periods (default 1.0)

    Returns:
        (holdings, trade_sizes): Arrays of remaining holdings and trade sizes

    Note:
        - risk_aversion = 0 produces TWAP (linear liquidation)
        - risk_aversion -> inf produces immediate execution
    """
    T = duration_periods
    X = order_size

    # Handle risk-neutral case (TWAP)
    if risk_aversion == 0 or risk_aversion < 1e-10:
        holdings = np.array([X * (1 - t/T) for t in range(T + 1)])
        trade_sizes = np.diff(holdings) * -1  # Positive trade sizes
        return holdings, trade_sizes

    # Adjusted parameters
    eta_tilde = eta - 0.5 * gamma * tau

    # Validate model constraint
    if eta_tilde <= 0:
        raise ValueError(
            f"Invalid parameters: eta_tilde = {eta_tilde:.6f} <= 0. "
            f"Reduce gamma or increase eta."
        )

    # Calculate kappa (decay rate)
    kappa_tilde_sq = (risk_aversion * sigma**2) / eta_tilde
    kappa = np.arccosh(0.5 * kappa_tilde_sq * tau**2 + 1) / tau

    # Generate trajectory using sinh formula
    holdings = np.array([
        X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        for t in range(T + 1)
    ])

    # Trade sizes are the differences (negative diff = selling/reducing position)
    trade_sizes = np.diff(holdings) * -1  # Make positive for execution

    return holdings, trade_sizes
```

### TWAP Strategy Generation
```python
# Source: Standard TWAP algorithm
def generate_twap_trajectory(
    order_size: float,
    duration_periods: int,
    randomize: bool = False,
    random_pct: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Time-Weighted Average Price execution trajectory.

    Divides order evenly across time periods.

    Args:
        order_size: Total size to execute
        duration_periods: Number of execution intervals
        randomize: If True, add random variation to slice sizes
        random_pct: Maximum random variation as fraction (default 10%)

    Returns:
        (holdings, trade_sizes): Remaining holdings and trade sizes per period
    """
    base_slice = order_size / duration_periods

    if randomize:
        # Add randomization to avoid detection
        rng = np.random.default_rng()
        variations = rng.uniform(1 - random_pct, 1 + random_pct, duration_periods)
        trade_sizes = np.array([base_slice * v for v in variations])
        # Adjust last slice to ensure exact total
        trade_sizes[-1] = order_size - trade_sizes[:-1].sum()
    else:
        trade_sizes = np.full(duration_periods, base_slice)

    # Calculate holdings from trade sizes
    holdings = np.zeros(duration_periods + 1)
    holdings[0] = order_size
    for i in range(duration_periods):
        holdings[i + 1] = holdings[i] - trade_sizes[i]

    return holdings, trade_sizes
```

### VWAP Strategy Generation
```python
# Source: Standard VWAP algorithm with volume profile
def generate_vwap_trajectory(
    order_size: float,
    volume_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Volume-Weighted Average Price execution trajectory.

    Weights execution by predicted volume at each interval.

    Args:
        order_size: Total size to execute
        volume_profile: Expected volume at each interval (will be normalized)

    Returns:
        (holdings, trade_sizes): Remaining holdings and trade sizes per period
    """
    # Normalize volume profile to weights
    total_volume = volume_profile.sum()
    if total_volume == 0:
        # Fall back to TWAP if no volume data
        weights = np.ones(len(volume_profile)) / len(volume_profile)
    else:
        weights = volume_profile / total_volume

    # Allocate order by volume weights
    trade_sizes = order_size * weights

    # Calculate holdings
    holdings = np.zeros(len(volume_profile) + 1)
    holdings[0] = order_size
    for i in range(len(volume_profile)):
        holdings[i + 1] = holdings[i] - trade_sizes[i]

    return holdings, trade_sizes
```

### Parameter Calibration from Historical Data
```python
# Source: Almgren-Chriss paper heuristics + project's existing calibration
def calibrate_almgren_chriss_params(
    daily_volume: float,
    daily_spread: float,
    daily_volatility: float,
    price: float,
    alpha: float = 0.5,
    tau: float = 1.0,
) -> AlmgrenChrissParams:
    """
    Calibrate Almgren-Chriss parameters from market data.

    Uses Almgren-Chriss heuristics:
    - eta: Trading 1% of daily volume causes temp impact of full spread
    - gamma: Trading 10% of daily volume causes perm impact of full spread

    Args:
        daily_volume: Average daily trading volume
        daily_spread: Average bid-ask spread in price units
        daily_volatility: Daily volatility as decimal (e.g., 0.02 for 2%)
        price: Current asset price
        alpha: Impact exponent (default 0.5 from square-root model)
        tau: Time interval between trades

    Returns:
        AlmgrenChrissParams with calibrated values
    """
    warnings = []

    # Temporary impact: 1% ADV -> full spread
    eta = daily_spread / (0.01 * daily_volume)

    # Permanent impact: 10% ADV -> full spread
    gamma = daily_spread / (0.10 * daily_volume)

    # Volatility in $/share (absolute, not percentage)
    sigma = daily_volatility * price

    # Validate eta_tilde constraint
    eta_tilde = eta - 0.5 * gamma * tau
    if eta_tilde <= 0:
        warnings.append(
            f"eta_tilde = {eta_tilde:.6f} <= 0: model constraints violated. "
            f"Consider larger tau or use TWAP instead."
        )

    return AlmgrenChrissParams(
        eta=eta,
        gamma=gamma,
        sigma=sigma,
        alpha=alpha,
        tau=tau,
        source_calibration=None,  # Populated if using existing CalibrationResult
        warnings=warnings,
    )
```

### Strategy Comparison
```python
# Source: Implementation shortfall framework
@dataclass
class StrategyComparison:
    """Comparison of execution strategies before simulation."""
    strategies: List[ExecutionTrajectory]
    baseline_strategy: str  # Reference strategy for comparison

    def summary_table(self) -> pd.DataFrame:
        """Generate comparison summary."""
        rows = []
        for strat in self.strategies:
            rows.append({
                'strategy': strat.strategy_name,
                'risk_aversion': strat.risk_aversion,
                'expected_cost_bps': strat.total_cost_estimate,
                'num_slices': len(strat.trade_sizes),
                'max_slice_pct': strat.trade_sizes.max() / strat.trade_sizes.sum() * 100,
                'front_loaded': strat.trade_sizes[0] > strat.trade_sizes[-1],
            })
        return pd.DataFrame(rows)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed TWAP | Adaptive TWAP with randomization | Industry standard | Reduces detection, gaming |
| Single asset A-C | Multi-asset A-C with correlations | 2001 paper already covered | Project scope is single-asset |
| Linear impact | Square-root impact (alpha=0.5) | Empirical validation ~2005 | Already in Phase 2 impact.py |
| Risk-neutral only | Risk-averse with lambda | A-C framework | Core of Phase 3 |

**Deprecated/outdated:**
- Static parameters: Modern implementations re-calibrate continuously from recent data
- Ignoring spread: Half-spread cost should be included in total cost estimate

## Open Questions

Things that couldn't be fully resolved:

1. **Prediction market volume profile patterns**
   - What we know: VWAP needs volume profile, Polymarket has distinct patterns (event-driven spikes)
   - What's unclear: Whether historical intraday volume patterns are predictive for prediction markets
   - Recommendation: Start with simple hourly profiles, validate against actual execution in Phase 4

2. **Lambda (risk aversion) selection**
   - What we know: Lambda = 0 is TWAP, higher lambda is more aggressive
   - What's unclear: What lambda value is appropriate for prediction markets with thin liquidity
   - Recommendation: Expose as user parameter, suggest starting with low values (1e-6 range) and tuning

3. **Market order baseline definition**
   - What we know: Need a "naive" baseline for comparison
   - What's unclear: Should market order be single slice or just very aggressive?
   - Recommendation: Define as executing 100% immediately (single slice, maximum impact)

## Sources

### Primary (HIGH confidence)
- Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions" - Core framework
- [almgren-chriss PyPI package](https://pypi.org/project/almgren-chriss/) - API reference for standard implementation
- [QuestDB Almgren-Chriss glossary](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/) - Parameter definitions
- [Dean Markwick: Solving the Almgren-Chriss Model](https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html) - Mathematical derivation
- [QuantJourney implementation](https://quantjourney.substack.com/p/how-to-sell-stocks-wisely-the-code) - Python class structure
- [SciPy optimize documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html) - Optimization functions

### Secondary (MEDIUM confidence)
- [Alpaca TWAP/VWAP tutorial](https://alpaca.markets/learn/algorithmic-trading-with-twap-and-vwap-using-alpaca) - TWAP/VWAP formulas
- [EODHD execution strategies](https://eodhd.medium.com/advanced-trading-strategies-maximizing-profits-with-vwap-twap-and-pov-using-python-987e0ead97f1) - Python patterns
- [Wikipedia TWAP](https://en.wikipedia.org/wiki/Time-weighted_average_price) - Basic definitions
- [TradersPost TWAP guide](https://blog.traderspost.io/article/twap-trading-strategies-guide) - Slice sizing heuristics

### Tertiary (LOW confidence)
- [Quantum Medium A-C article](https://quantum-blog.medium.com/almgren-chriss-optimal-execution-model-5a85b66555d2) - Parameter calibration heuristics (needs validation)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses existing project dependencies
- Architecture: HIGH - Follows Phase 2 patterns exactly
- Almgren-Chriss math: HIGH - Well-documented academic framework
- Parameter calibration: MEDIUM - Heuristics may need tuning for prediction markets
- TWAP/VWAP strategies: HIGH - Standard, well-documented algorithms

**Research date:** 2026-01-19
**Valid until:** 60 days (stable academic framework, implementation details settled)
