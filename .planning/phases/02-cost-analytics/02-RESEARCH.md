# Phase 2: Cost Analytics - Research

**Researched:** 2026-01-19
**Domain:** Transaction Cost Analysis, Market Impact Estimation, Execution Cost Forecasting
**Confidence:** HIGH

## Summary

Phase 2 builds the cost analytics layer on top of the Phase 1 foundation (QuestDBReader, VWAP/TWAP/arrival price benchmarks). The core deliverables are: slippage calculation in basis points, implementation shortfall decomposition (Perold framework), temporary vs permanent market impact estimation, and execution cost forecasting from orderbook depth.

The standard approach uses the Perold framework for implementation shortfall decomposition into timing, impact, spread, and opportunity cost components. For market impact estimation, the square-root law is the standard model, but must be adapted for thin liquidity markets like Polymarket where standard equity assumptions break down. The Talos three-component model (spread + physical impact + time risk) provides a better framework for crypto/prediction markets.

Cost forecasting from orderbook depth uses a "walk the book" approach - simulating order execution across price levels to calculate expected slippage. This is more reliable than model-based forecasting for thin liquidity markets where impact models are poorly calibrated.

**Primary recommendation:** Use orderbook-based slippage estimation as the primary cost forecasting method. Model-based impact estimation (square-root law) should be implemented but treated as secondary validation, not primary truth.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 2.x | DataFrame operations, time series | Foundation for all analytics |
| numpy | 1.26+ | Array operations, vectorized math | Performance for large datasets |
| statsmodels | 0.14+ | OLS regression for impact calibration | Standard for econometric analysis |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.optimize | 1.17+ | Parameter calibration | Fitting impact curves to data |
| almgren-chriss | 0.1+ | Reference implementation | Optimal execution (Phase 3) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| statsmodels OLS | sklearn LinearRegression | statsmodels provides richer diagnostics (p-values, confidence intervals) |
| Custom slippage | tcapy | tcapy is FX-specific and stale (2021); custom is more appropriate for prediction markets |

**Installation:**
```bash
pip install statsmodels scipy
# almgren-chriss is for Phase 3
```

## Architecture Patterns

### Recommended Module Structure
```
src/tributary/analytics/
    __init__.py           # (existing) Exports
    reader.py             # (existing) QuestDB queries
    benchmarks.py         # (existing) VWAP, TWAP, arrival price
    slippage.py           # (new) Slippage calculation in bps
    shortfall.py          # (new) Implementation shortfall decomposition
    impact.py             # (new) Market impact estimation
    cost_forecast.py      # (new) Execution cost forecasting
```

### Pattern 1: Slippage Calculation in Basis Points

**What:** Calculate the difference between expected and actual execution price, expressed in basis points (1/100th of a percent).

**When to use:** Measuring actual execution quality against benchmarks (arrival price, VWAP, TWAP).

**Formula:**
```
slippage_bps = ((execution_price - benchmark_price) / benchmark_price) * 10000

# For buys: positive slippage = worse execution (paid more)
# For sells: positive slippage = worse execution (received less)
# Convention: Use signed slippage where positive = cost
```

**Example:**
```python
def calculate_slippage_bps(
    execution_price: float,
    benchmark_price: float,
    side: str  # 'buy' or 'sell'
) -> float:
    """
    Calculate slippage in basis points.

    Positive value = cost (unfavorable execution)
    Negative value = gain (favorable execution)

    Args:
        execution_price: Actual average execution price
        benchmark_price: Reference price (arrival, VWAP, or TWAP)
        side: Trade direction ('buy' or 'sell')

    Returns:
        Slippage in basis points (1 bp = 0.01%)
    """
    if benchmark_price == 0:
        return float('nan')

    raw_slippage = (execution_price - benchmark_price) / benchmark_price * 10000

    # For sells, flip sign (getting less than expected is a cost)
    if side == 'sell':
        return -raw_slippage
    return raw_slippage
```

### Pattern 2: Implementation Shortfall Decomposition (Perold Framework)

**What:** Decompose total execution cost into components: delay cost, trading cost (market impact), spread cost, and opportunity cost.

**When to use:** Understanding WHERE execution costs come from, not just total magnitude.

**Components:**
```
IS_total = Delay Cost + Trading Cost + Opportunity Cost + Explicit Fees

Where:
- Delay Cost = shares_executed * (order_entry_price - decision_price)
- Trading Cost = actual_cost - (shares_executed * order_entry_price)
- Opportunity Cost = shares_not_executed * (closing_price - decision_price)
- Explicit Fees = commission + exchange_fees

IS_bps = (IS_total_dollars / (total_shares * decision_price)) * 10000
```

**Example:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ShortfallComponents:
    """Implementation shortfall decomposition result."""
    delay_cost_bps: float      # Cost from waiting to submit order
    trading_cost_bps: float    # Cost from market impact during execution
    spread_cost_bps: float     # Cost from bid-ask spread
    opportunity_cost_bps: float # Cost from unfilled portion
    total_bps: float           # Sum of all components

    # Dollar amounts for transparency
    delay_cost_usd: float
    trading_cost_usd: float
    spread_cost_usd: float
    opportunity_cost_usd: float
    total_usd: float


def decompose_implementation_shortfall(
    decision_price: float,       # Price when decision made (mid-price)
    order_entry_price: float,    # Price when order submitted (mid-price)
    execution_prices: list[float],  # List of fill prices
    execution_sizes: list[float],   # List of fill sizes
    total_order_size: float,     # Originally intended size
    closing_price: float,        # Price at end of period
    side: str,                   # 'buy' or 'sell'
    spread_at_entry: Optional[float] = None,  # Bid-ask spread
) -> ShortfallComponents:
    """
    Decompose implementation shortfall into components.

    Based on Perold (1988) and Wagner & Edwards (1993) framework.
    """
    executed_size = sum(execution_sizes)
    unfilled_size = total_order_size - executed_size

    # Weighted average execution price
    if executed_size > 0:
        avg_execution_price = (
            sum(p * s for p, s in zip(execution_prices, execution_sizes))
            / executed_size
        )
    else:
        avg_execution_price = decision_price

    # Delay cost: movement between decision and order entry
    delay_cost_usd = executed_size * (order_entry_price - decision_price)

    # Trading cost: movement from order entry to execution
    trading_cost_usd = executed_size * (avg_execution_price - order_entry_price)

    # Spread cost: half-spread crossed
    if spread_at_entry is not None:
        spread_cost_usd = executed_size * (spread_at_entry / 2)
    else:
        spread_cost_usd = 0.0

    # Opportunity cost: unfilled portion
    opportunity_cost_usd = unfilled_size * (closing_price - decision_price)

    # Flip signs for sells
    if side == 'sell':
        delay_cost_usd = -delay_cost_usd
        trading_cost_usd = -trading_cost_usd
        opportunity_cost_usd = -opportunity_cost_usd

    total_usd = delay_cost_usd + trading_cost_usd + spread_cost_usd + opportunity_cost_usd

    # Convert to basis points
    notional = total_order_size * decision_price

    return ShortfallComponents(
        delay_cost_bps=delay_cost_usd / notional * 10000 if notional else 0,
        trading_cost_bps=trading_cost_usd / notional * 10000 if notional else 0,
        spread_cost_bps=spread_cost_usd / notional * 10000 if notional else 0,
        opportunity_cost_bps=opportunity_cost_usd / notional * 10000 if notional else 0,
        total_bps=(total_usd / notional * 10000) if notional else 0,
        delay_cost_usd=delay_cost_usd,
        trading_cost_usd=trading_cost_usd,
        spread_cost_usd=spread_cost_usd,
        opportunity_cost_usd=opportunity_cost_usd,
        total_usd=total_usd,
    )
```

### Pattern 3: Orderbook-Based Slippage Estimation ("Walk the Book")

**What:** Simulate order execution across orderbook levels to estimate expected slippage for a given order size.

**When to use:** Cost forecasting ("what would $X cost to execute?"). This is the PRIMARY method for thin liquidity markets.

**Algorithm:**
```
1. Get current orderbook snapshot (bid_prices, bid_sizes, ask_prices, ask_sizes)
2. For buy order: iterate through asks (ascending price)
   For sell order: iterate through bids (descending price)
3. Accumulate size until order is filled
4. Calculate volume-weighted average execution price
5. Compare to mid-price for slippage estimate
```

**Example:**
```python
def estimate_slippage_from_orderbook(
    order_size: float,
    side: str,
    bid_prices: list[float],
    bid_sizes: list[float],
    ask_prices: list[float],
    ask_sizes: list[float],
) -> dict:
    """
    Estimate slippage by walking the orderbook.

    Args:
        order_size: Size to execute (in base currency)
        side: 'buy' or 'sell'
        bid_prices: Bid prices (descending order)
        bid_sizes: Sizes at each bid level
        ask_prices: Ask prices (ascending order)
        ask_sizes: Sizes at each ask level

    Returns:
        Dictionary with:
        - mid_price: Reference price
        - expected_execution_price: VWAP across levels
        - slippage_bps: Expected slippage in basis points
        - levels_consumed: Number of orderbook levels used
        - fully_filled: Whether order can be fully filled
    """
    if not bid_prices or not ask_prices:
        return {'error': 'Empty orderbook'}

    mid_price = (bid_prices[0] + ask_prices[0]) / 2

    # Select relevant side of book
    if side == 'buy':
        prices = ask_prices
        sizes = ask_sizes
    else:
        prices = bid_prices
        sizes = bid_sizes

    # Walk the book
    remaining = order_size
    total_cost = 0.0
    total_filled = 0.0
    levels_consumed = 0

    for price, size in zip(prices, sizes):
        if remaining <= 0:
            break

        fill_size = min(remaining, size)
        total_cost += fill_size * price
        total_filled += fill_size
        remaining -= fill_size
        levels_consumed += 1

    if total_filled == 0:
        return {
            'mid_price': mid_price,
            'expected_execution_price': float('nan'),
            'slippage_bps': float('nan'),
            'levels_consumed': 0,
            'fully_filled': False,
        }

    avg_price = total_cost / total_filled

    # Calculate slippage (positive = cost)
    if side == 'buy':
        slippage_bps = (avg_price - mid_price) / mid_price * 10000
    else:
        slippage_bps = (mid_price - avg_price) / mid_price * 10000

    return {
        'mid_price': mid_price,
        'expected_execution_price': avg_price,
        'slippage_bps': slippage_bps,
        'levels_consumed': levels_consumed,
        'fully_filled': remaining <= 0,
        'unfilled_size': max(0, remaining),
    }
```

### Pattern 4: Market Impact Estimation (Square-Root Model)

**What:** Estimate market impact as a function of order size and market conditions.

**When to use:** Model-based impact forecasting. Use as SECONDARY validation, not primary truth for thin markets.

**Formula (Square-Root Law):**
```
Impact_bps = sigma * sqrt(order_size / ADV) * 10000

Where:
- sigma = daily volatility (decimal, e.g., 0.02 for 2%)
- order_size = size of order
- ADV = average daily volume

For temporary vs permanent decomposition:
- Temporary Impact: Reverts after execution (concave in size)
- Permanent Impact: Persists (linear in size for arbitrage-free)
```

**Example:**
```python
import numpy as np
from dataclasses import dataclass

@dataclass
class ImpactEstimate:
    """Market impact estimation result."""
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    notes: list[str]


def estimate_market_impact(
    order_size: float,
    daily_volume: float,
    volatility: float,  # Daily volatility as decimal
    spread_bps: float,  # Current bid-ask spread in bps
    alpha: float = 0.5,  # Square-root exponent (0.5-0.6 typical)
) -> ImpactEstimate:
    """
    Estimate market impact using square-root model.

    Based on Almgren et al. (2005) empirical findings.

    WARNING: This model is calibrated for equity markets with deep liquidity.
    For thin liquidity markets (prediction markets, small-cap crypto),
    use orderbook-based estimation instead.

    Args:
        order_size: Total order size
        daily_volume: Average daily volume
        volatility: Daily volatility (decimal form)
        spread_bps: Current bid-ask spread in basis points
        alpha: Impact exponent (typically 0.5 for square-root)

    Returns:
        ImpactEstimate with temporary, permanent, and total impact
    """
    notes = []
    confidence = 'MEDIUM'

    if daily_volume <= 0:
        return ImpactEstimate(
            temporary_impact_bps=float('nan'),
            permanent_impact_bps=float('nan'),
            total_impact_bps=float('nan'),
            confidence='LOW',
            notes=['Invalid daily volume'],
        )

    participation_rate = order_size / daily_volume

    # Check for thin liquidity conditions
    if participation_rate > 0.10:
        notes.append('High participation rate (>10%): model unreliable')
        confidence = 'LOW'
    elif participation_rate > 0.01:
        notes.append('Elevated participation rate (1-10%): use with caution')

    # Square-root model for temporary impact
    # Impact = sigma * (Q/V)^alpha
    temporary_impact = volatility * (participation_rate ** alpha)
    temporary_impact_bps = temporary_impact * 10000

    # Permanent impact: typically ~1/3 to 1/2 of temporary for equities
    # For thin markets, this ratio is highly uncertain
    permanent_ratio = 0.4
    permanent_impact_bps = temporary_impact_bps * permanent_ratio

    # Add half-spread as execution cost
    spread_cost = spread_bps / 2

    total_impact_bps = temporary_impact_bps + permanent_impact_bps + spread_cost

    return ImpactEstimate(
        temporary_impact_bps=temporary_impact_bps,
        permanent_impact_bps=permanent_impact_bps,
        total_impact_bps=total_impact_bps,
        confidence=confidence,
        notes=notes,
    )
```

### Pattern 5: Calibrating Impact Parameters from Historical Data

**What:** Fit impact model parameters to actual execution data using regression.

**When to use:** When you have historical execution data and want to calibrate the model to YOUR market.

**Approach:**
```
1. Collect historical executions: (order_size, execution_price, benchmark_price)
2. Calculate realized impact: impact = (exec_price - benchmark) / benchmark
3. Calculate features: participation = order_size / daily_volume
4. Regress: log(impact) ~ alpha * log(participation) + log(sigma) + epsilon
5. Extract alpha (exponent) and implicit volatility scaling
```

**Example:**
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm


def calibrate_impact_parameters(
    executions_df: pd.DataFrame,
) -> dict:
    """
    Calibrate square-root impact model from historical executions.

    Args:
        executions_df: DataFrame with columns:
            - order_size: Size of each order
            - daily_volume: ADV at time of execution
            - realized_impact_bps: Actual impact in basis points
            - volatility: Daily volatility at time of execution

    Returns:
        Dictionary with:
        - alpha: Impact exponent (0.5 = square-root)
        - scale: Scaling factor
        - r_squared: Model fit
        - std_error: Standard error of alpha estimate
    """
    df = executions_df.copy()

    # Filter out zero/negative values
    df = df[(df['order_size'] > 0) & (df['daily_volume'] > 0)]
    df = df[df['realized_impact_bps'] > 0]  # Only adverse moves

    if len(df) < 10:
        return {'error': 'Insufficient data (need 10+ observations)'}

    # Log-transform for regression
    df['log_impact'] = np.log(df['realized_impact_bps'])
    df['log_participation'] = np.log(df['order_size'] / df['daily_volume'])
    df['log_volatility'] = np.log(df['volatility'])

    # Regression: log(impact) = c + alpha*log(participation) + beta*log(vol)
    X = df[['log_participation', 'log_volatility']]
    X = sm.add_constant(X)
    y = df['log_impact']

    model = sm.OLS(y, X).fit()

    return {
        'alpha': model.params['log_participation'],
        'volatility_sensitivity': model.params['log_volatility'],
        'intercept': model.params['const'],
        'r_squared': model.rsquared,
        'alpha_std_error': model.bse['log_participation'],
        'n_observations': len(df),
        'model_summary': str(model.summary()),
    }
```

### Anti-Patterns to Avoid

- **Using equity impact models directly on prediction markets:** Standard square-root law is calibrated for deep orderbooks. Polymarket often has thin liquidity where a single order consumes all depth. Always validate with orderbook-based estimates.

- **Ignoring participation rate thresholds:** When order_size > 10% of daily volume, ALL impact models become unreliable. Flag these cases explicitly.

- **Treating model-based and orderbook-based estimates as equivalent:** Orderbook-based is DIRECT measurement; model-based is INTERPOLATION. Prefer direct measurement.

- **Calculating slippage without side information:** A buy order paying 10bps above mid is a cost; a sell order receiving 10bps above mid is a gain. Sign conventions matter.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OLS regression | Custom least squares | `statsmodels.OLS` | Handles heteroskedasticity, provides diagnostics |
| Basis point conversion | Inline math everywhere | Utility function with side handling | Sign conventions are error-prone |
| DataFrame aggregations | Loops | pandas groupby/resample | 10-100x faster, less error-prone |
| Confidence intervals | Bootstrap by hand | statsmodels built-in | Already validated |

**Key insight:** The math is simple, but the edge cases (zero values, side conventions, empty data) are not. Centralize these in tested functions rather than duplicating logic.

## Common Pitfalls

### Pitfall 1: Applying Equity Impact Models to Thin Liquidity
**What goes wrong:** Square-root model predicts 5bps impact; actual execution is 50bps because you consumed the entire orderbook.
**Why it happens:** Equity models assume deep liquidity and continuous price function. Prediction markets have discrete, sparse orderbooks.
**How to avoid:** Always run orderbook-based estimation first. Use model-based as secondary check, not primary forecast.
**Warning signs:** High participation rate (>5%), few levels in orderbook, model estimate << orderbook estimate.

### Pitfall 2: Confusing Temporary and Permanent Impact
**What goes wrong:** Treating all impact as temporary leads to underestimating subsequent trade costs. Treating all as permanent overestimates total costs.
**Why it happens:** Temporary impact reverses after execution; permanent impact persists. The ratio varies by market.
**How to avoid:** For thin markets, assume higher permanent impact ratio (50-70%) as conservative default. Calibrate from data if possible.
**Warning signs:** Execution data shows prices not reverting after trades.

### Pitfall 3: Sign Convention Errors in Slippage
**What goes wrong:** Buy slippage reported as negative (implying gain) when execution was actually worse.
**Why it happens:** Inconsistent conventions between buy/sell sides.
**How to avoid:** Use explicit `side` parameter and document convention: positive = cost, negative = gain.
**Warning signs:** Aggregate metrics showing negative average slippage (unless strategy genuinely captures alpha).

### Pitfall 4: Implementation Shortfall with Sparse Data
**What goes wrong:** Calculating opportunity cost when no orderbook existed at decision time.
**Why it happens:** Prediction market orderbooks can disappear entirely during illiquid periods.
**How to avoid:** Check for orderbook existence at each reference time. Return NaN with explanation rather than garbage numbers.
**Warning signs:** decision_price is None or arrival_price returns None.

### Pitfall 5: Overfitting Impact Calibration
**What goes wrong:** Calibrated model has great R-squared on training data, fails on new data.
**Why it happens:** Small dataset + many parameters = overfitting. Prediction markets have few historical executions.
**How to avoid:** Use simple models (square-root, not custom polynomials). Require minimum 30+ observations. Report confidence intervals on parameters.
**Warning signs:** Alpha estimate with std_error > 0.3, R-squared < 0.3.

## Code Examples

Verified patterns from official sources:

### statsmodels OLS Regression
```python
# Source: https://www.statsmodels.org/stable/regression.html
import statsmodels.api as sm
import numpy as np

# Prepare data
X = np.column_stack([participation_rates, volatilities])
X = sm.add_constant(X)  # Add intercept term
y = realized_impacts

# Fit model with heteroskedasticity-robust standard errors
model = sm.OLS(y, X).fit(cov_type='HC3')

# Access results
print(f"Alpha: {model.params[1]:.3f} +/- {model.bse[1]:.3f}")
print(f"R-squared: {model.rsquared:.3f}")
print(f"F-statistic p-value: {model.f_pvalue:.4f}")
```

### pandas VWAP Calculation
```python
# Source: Phase 1 implementation (benchmarks.py)
def calculate_vwap(trades_df: pd.DataFrame) -> float:
    """Volume-weighted average price."""
    if trades_df.empty:
        return float('nan')
    total_volume = trades_df['size'].sum()
    if total_volume == 0:
        return float('nan')
    return (trades_df['price'] * trades_df['size']).sum() / total_volume
```

### Orderbook Slippage from Existing Code Pattern
```python
# Source: QuestDBReader.query_orderbook_snapshots() returns parsed lists
# Use this pattern for slippage estimation
snapshot = reader.query_orderbook_snapshots(
    market_id=market_id,
    start_time=order_time - timedelta(seconds=5),
    end_time=order_time + timedelta(milliseconds=1),
    token_id=token_id,
).iloc[-1]

slippage = estimate_slippage_from_orderbook(
    order_size=10000,  # $10K order
    side='buy',
    bid_prices=snapshot['bid_prices'],
    bid_sizes=snapshot['bid_sizes'],
    ask_prices=snapshot['ask_prices'],
    ask_sizes=snapshot['ask_sizes'],
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Linear impact (Kyle) | Square-root law | 2000s | Better fit for large orders |
| Fixed slippage assumption | Orderbook-based | 2010s | Much more accurate for varied sizes |
| Equity models for crypto | Crypto-specific (Talos) | 2023-2025 | Handles thin liquidity, 24/7 trading |
| Single impact number | Temporary + Permanent decomposition | 2000 (Almgren-Chriss) | Enables optimal execution |

**Deprecated/outdated:**
- **tcapy library:** Last updated 2021, FX-specific. Do not use.
- **Fixed basis point assumptions:** (e.g., "assume 10bps slippage") Meaningless for thin liquidity markets.
- **Linear Kyle lambda for large orders:** Systematically underestimates impact.

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal permanent/temporary ratio for prediction markets**
   - What we know: Equity markets show ~30-40% permanent. Crypto shows higher permanent ratio.
   - What's unclear: No published research on prediction market impact decomposition.
   - Recommendation: Start with 50% permanent as conservative default. Calibrate from execution data as it accumulates.

2. **Time decay of temporary impact**
   - What we know: Temporary impact typically decays over minutes to hours in equity markets.
   - What's unclear: Decay rate in 24/7 crypto-like prediction markets.
   - Recommendation: Track price recovery after executions. Empirically estimate decay rate from data.

3. **Handling zero-volume periods**
   - What we know: Many Polymarket markets have extended periods with zero trades.
   - What's unclear: How to estimate impact when ADV is effectively zero.
   - Recommendation: Use orderbook-based method exclusively. Return "insufficient liquidity" when orderbook too thin.

## Sources

### Primary (HIGH confidence)
- [CFA Level III Implementation Shortfall Notes](https://analystprep.com/study-notes/cfa-level-iii/measurement-and-determination-of-cost-of-trade/) - Perold framework decomposition
- [QuestDB Implementation Shortfall Glossary](https://questdb.com/glossary/implementation-shortfall-analysis/) - Component definitions
- [QuestDB Market Impact Models Glossary](https://questdb.com/glossary/market-impact-models/) - Square-root formula
- [Coin Metrics Slippage Documentation](https://gitbook-docs.coinmetrics.io/market-data/market-data-overview/liquidity/slippage) - Walk-the-book algorithm
- [statsmodels documentation](https://www.statsmodels.org/) - OLS regression API

### Secondary (MEDIUM confidence)
- [Talos Market Impact Model](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs) - Three-component crypto model
- [Amberdata Liquidity Research](https://blog.amberdata.io/beyond-the-spread-understanding-market-impact-and-execution) - Orderbook depth empirical findings
- [Quantitative Brokers IS History](https://www.quantitativebrokers.com/blog/a-brief-history-of-implementation-shortfall) - Perold framework evolution
- [Stephen Diehl Slippage Modeling](https://www.stephendiehl.com/posts/slippage/) - Four-component slippage model

### Tertiary (LOW confidence)
- [CME Reassessing Liquidity 2025](https://www.cmegroup.com/articles/2025/reassessing-liquidity-beyond-order-book-depth.html) - Temporal liquidity patterns
- [Square-Root Law Research](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact) - Theoretical background

## Metadata

**Confidence breakdown:**
- Slippage calculation: HIGH - Standard formulas, well-documented conventions
- Implementation shortfall: HIGH - Perold framework is 35+ years established
- Square-root impact model: MEDIUM - Well-established for equities, uncertain for thin markets
- Orderbook-based forecasting: HIGH - Direct measurement, no model assumptions
- Impact calibration: MEDIUM - Standard regression, but limited data in prediction markets

**Research date:** 2026-01-19
**Valid until:** 2026-02-19 (30 days - stable domain)

---
*Research completed by gsd-researcher for Phase 2: Cost Analytics*
