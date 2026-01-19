# Technology Stack: Execution Analytics

**Project:** Tributary - Execution Analytics Layer
**Researched:** 2026-01-19
**Overall Confidence:** HIGH

## Executive Summary

This document recommends the Python stack for building execution analytics on top of Tributary's existing QuestDB + pandas infrastructure. The ecosystem for execution analytics is mature but fragmented - no single library covers VWAP/TWAP, implementation shortfall, market impact modeling, and backtesting. The recommended approach is a layered architecture combining specialized libraries with custom implementations where needed.

**Key insight for eFX traders:** The execution analytics space in Python is dominated by equity-focused tools. Prediction markets and crypto have unique characteristics (24/7 trading, different liquidity patterns, binary outcomes) that require custom implementations building on general-purpose numerical libraries rather than off-the-shelf TCA solutions.

---

## Recommended Stack

### Core Numerical Foundation

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **numpy** | >=1.26 | Array operations, vectorized calculations | HIGH |
| **pandas** | >=2.0 | Time series manipulation, DataFrame operations | HIGH |
| **scipy** | >=1.11 | Optimization (SLSQP, minimize), statistical functions | HIGH |
| **numba** | >=0.59 | JIT compilation for performance-critical loops | HIGH |

**Rationale:** These form the numerical bedrock. Already partially in your `notebooks` dependencies. Numba is critical - execution analytics involves iterating over orderbook states and trade sequences; pure Python is too slow. Numba's `@njit` decorator provides 10-100x speedups on numerical loops with minimal code changes.

**Integration with existing stack:** Direct compatibility with your current pandas/numpy setup. QuestDB query results flow into pandas DataFrames via psycopg2/SQLAlchemy.

### VWAP/TWAP Calculation

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **Custom implementation** | N/A | VWAP/TWAP calculations | HIGH |
| **pandas-ta** | >=0.3.14b | Optional: 150+ technical indicators | MEDIUM |

**Rationale:** VWAP and TWAP are simple enough that custom implementation is preferred over adding dependencies. The formulas are:

```python
# VWAP: Volume-Weighted Average Price
vwap = (price * volume).cumsum() / volume.cumsum()

# TWAP: Time-Weighted Average Price
twap = price.rolling(window).mean()  # Or resampled average
```

**Why NOT dedicated VWAP libraries:**
- Most "VWAP libraries" are thin wrappers around pandas operations
- QuantConnect's Lean engine is overkill for analytics (it's a full trading system)
- Alpaca's implementations are tied to their data format

**pandas-ta consideration:** Include if you need a broader indicator library (RSI, MACD, etc.) for signal generation alongside execution analytics. The `ft-pandas-ta` fork (0.3.16) is actively maintained. However, for pure execution analytics, you don't need it.

### Implementation Shortfall

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **Custom implementation** | N/A | Implementation shortfall calculation | HIGH |
| **tcapy** | 0.11.2 | Reference implementation (FX-focused) | LOW |

**Rationale:** Implementation shortfall = Decision Price - Execution Price (in basis points). This is straightforward to implement:

```python
# Implementation shortfall in bps
shortfall_bps = ((exec_price - decision_price) / decision_price) * 10000
```

The complexity is in defining "decision price" correctly:
- Arrival price (mid at order submission)
- VWAP during execution window
- Close price
- Previous close

**tcapy assessment (Cuemacro):**
- Last updated: January 2021 (4+ years stale)
- FX-specific with heavy infrastructure requirements (Redis, Celery, Memcached)
- Overkill for your use case
- **Recommendation:** Study its benchmark calculation logic but don't use as dependency

**Confidence:** HIGH for custom implementation. The math is well-defined; the challenge is data pipeline, not algorithms.

### Market Impact Modeling (Almgren-Chriss)

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **almgren-chriss** | 1.1.0 | Almgren-Chriss optimal execution model | MEDIUM |
| **scipy.optimize** | >=1.11 | Quadratic programming for custom models | HIGH |
| **cvxpy** | >=1.4 | Convex optimization (if needed) | HIGH |

**Rationale:** The `almgren-chriss` package on PyPI provides the core functions:
- `trade_trajectory()` - optimal trade schedule
- `trade_list()` - individual trade sizes
- `cost_expectation()` - expected execution cost
- `cost_variance()` - variance of execution cost

**Limitations of almgren-chriss package:**
- Last updated May 2023
- Basic implementation (single-asset)
- No calibration tools for estimating impact parameters

**Alternative: Custom implementation with scipy.optimize**

The Almgren-Chriss model reduces to a quadratic program. For multi-asset or custom impact functions, implement directly:

```python
from scipy.optimize import minimize

def almgren_chriss_cost(trajectory, params):
    """
    params: risk_aversion, permanent_impact, temporary_impact, volatility
    """
    # ... quadratic cost function

result = minimize(almgren_chriss_cost, x0=initial_trajectory, method='SLSQP')
```

**cvxpy consideration:** Use if you need more sophisticated convex optimization (e.g., constraints on position limits, sector exposures). More readable for complex optimization problems but slower than scipy.optimize for simple cases.

**Confidence:** MEDIUM for almgren-chriss package (functional but dated). HIGH for scipy.optimize approach (well-tested, flexible).

### Impact Parameter Estimation

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **statsmodels** | >=0.14.6 | Regression for impact estimation | HIGH |
| **scikit-learn** | >=1.8 | ML-based impact prediction | HIGH |

**Rationale:** Market impact parameters (permanent/temporary impact coefficients) must be estimated from historical data. This requires:

1. **Linear regression** (statsmodels) for classic impact estimation:
   ```python
   # Estimate: price_change = alpha * sqrt(volume/ADV) + epsilon
   import statsmodels.api as sm
   model = sm.OLS(price_changes, X).fit()
   ```

2. **ML models** (scikit-learn) for non-linear impact prediction:
   - Random forests for impact regime classification
   - Gradient boosting for impact magnitude prediction

**Note:** Prediction markets have unique impact dynamics (binary outcomes, event-driven liquidity). Standard equity impact models may need significant adaptation.

### Execution Simulation / Backtesting

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **hftbacktest** | >=2.4.4 | HFT-grade orderbook simulation | HIGH |
| **vectorbt** | >=0.28.2 | Fast vectorized backtesting | MEDIUM |
| **Custom simulator** | N/A | Prediction market specific simulation | HIGH |

**Recommended: hftbacktest**

hftbacktest is the clear winner for execution simulation because it:
- Accounts for queue position in order fills
- Models both feed and order latency
- Reconstructs full orderbook from L2/L3 data
- Uses Numba for performance
- Supports tick-by-tick simulation

**Key features relevant to your use case:**
- Works with your existing orderbook snapshot data
- Realistic fill simulation (not just "fill at mid")
- Latency modeling for realistic slippage

**Installation:**
```bash
pip install hftbacktest>=2.4.4
```

**Requires:** Python 3.11+ (matches your project requirement)

**vectorbt assessment:**
- Excellent for strategy backtesting (signal-based)
- Less suitable for execution simulation (focuses on when to trade, not how)
- Useful for portfolio analytics after execution
- License: Apache 2.0 with Commons Clause (free for non-commercial, check for commercial use)

**Custom simulator rationale:**
Prediction markets have unique characteristics not handled by generic backtesting:
- Binary payoff at resolution
- Position limits (Polymarket caps)
- Resolution risk (market closes at unknown time)
- No continuous price (discrete orderbook states)

Recommend building a thin simulation layer using hftbacktest's orderbook mechanics.

### Time Series Analysis

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **statsmodels** | >=0.14.6 | ARIMA, VAR, regime detection | HIGH |
| **scipy.signal** | (part of scipy) | Filtering, spectral analysis | HIGH |

**Rationale:** For analyzing orderbook/trade time series:

1. **Volatility estimation:** GARCH models via statsmodels
2. **Regime detection:** Markov switching models
3. **Autocorrelation analysis:** ACF/PACF for order flow patterns
4. **Seasonality:** Decomposition for time-of-day effects

**Note:** QuestDB is optimized for time-series queries. Push aggregations (VWAP windows, rolling stats) to QuestDB rather than pulling raw data to Python.

### Portfolio Optimization (If Needed)

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **PyPortfolioOpt** | >=1.5.4 | Mean-variance optimization | MEDIUM |
| **skfolio** | latest | Modern portfolio optimization | MEDIUM |
| **cvxpy** | >=1.4 | Custom optimization problems | HIGH |

**Rationale:** If execution analytics expands to portfolio-level decisions:
- PyPortfolioOpt: Well-documented, includes Black-Litterman, HRP
- skfolio: Newer, scikit-learn API, better for cross-validation
- cvxpy: Build custom objectives/constraints

**Note:** These are lower priority for initial execution analytics. Add when expanding to multi-asset execution optimization.

---

## Integration with Existing Stack

### QuestDB Query Pattern

```python
# Recommended: Use questdb-connect for SQLAlchemy + pandas integration
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('questdb://localhost:8812/questdb')

# Query orderbook snapshots into pandas
df = pd.read_sql("""
    SELECT
        timestamp,
        market_id,
        mid_price,
        spread_bps,
        bid_sizes,
        ask_sizes
    FROM orderbook_snapshots
    WHERE market_id = 'market-xyz'
    AND timestamp BETWEEN '2025-01-01' AND '2025-01-15'
    ORDER BY timestamp
""", engine)
```

### Performance Pattern (Numba)

```python
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def calculate_vwap_numba(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Vectorized VWAP calculation with Numba acceleration."""
    n = len(prices)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        cum_pv += prices[i] * volumes[i]
        cum_v += volumes[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else np.nan
    return vwap
```

---

## What NOT to Use

| Technology | Why Not |
|------------|---------|
| **QuantConnect Lean** | Full trading platform, massive overhead for analytics only |
| **Zipline** | Quantopian shut down 2020, limited maintenance |
| **Backtrader** | Strategy backtesting focus, not execution simulation |
| **tcapy** | FX-specific, stale (2021), heavy infrastructure |
| **QuantLib-Python** | Derivatives pricing focus, overkill for execution analytics |
| **Alpaca SDK** | Tied to Alpaca data/broker, not general-purpose |
| **ta-lib** | Requires C compilation, pandas-ta is easier |

---

## Installation Summary

### Core Dependencies (Add to pyproject.toml)

```toml
[project.optional-dependencies]
analytics = [
    # Core numerical
    "numpy>=1.26",
    "pandas>=2.0",
    "scipy>=1.11",
    "numba>=0.59",

    # Statistical modeling
    "statsmodels>=0.14",
    "scikit-learn>=1.8",

    # Market impact
    "almgren-chriss>=1.1.0",

    # Execution simulation
    "hftbacktest>=2.4.4",

    # Optimization (optional, for advanced use)
    "cvxpy>=1.4",
]
```

### Installation Command

```bash
pip install -e ".[analytics]"
```

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Core numerical (numpy/pandas/scipy) | HIGH | Standard, well-tested ecosystem |
| VWAP/TWAP | HIGH | Simple calculations, custom implementation preferred |
| Implementation shortfall | HIGH | Well-defined metrics, straightforward |
| Almgren-Chriss (almgren-chriss pkg) | MEDIUM | Functional but dated, consider custom for flexibility |
| Execution simulation (hftbacktest) | HIGH | Best-in-class for HFT-grade simulation, actively maintained |
| Time series (statsmodels) | HIGH | Mature library, excellent documentation |
| Prediction market specifics | LOW | No off-the-shelf solutions; requires custom work |

---

## Gaps and Custom Development Needed

1. **Prediction Market Impact Models**
   - Standard impact models assume continuous trading; prediction markets are discrete
   - Need to adapt Almgren-Chriss for binary outcome markets
   - Resolution risk not captured in standard TCA

2. **QuestDB-Native Analytics**
   - Push VWAP/aggregation calculations to QuestDB SQL where possible
   - Reduces data transfer overhead

3. **Real-time Analytics**
   - Current recommendations are batch-oriented
   - For streaming analytics, consider adding `river` (online ML) or custom async pipelines

---

## Sources

### Official Documentation (HIGH confidence)
- [hftbacktest PyPI](https://pypi.org/project/hftbacktest/) - Version 2.4.4, December 2025
- [vectorbt PyPI](https://pypi.org/project/vectorbt/) - Version 0.28.2, December 2025
- [almgren-chriss PyPI](https://pypi.org/project/almgren-chriss/) - Version 1.1.0, May 2023
- [statsmodels TSA](https://www.statsmodels.org/stable/tsa.html) - Version 0.14.6
- [scipy.optimize](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) - Version 1.17.0
- [scikit-learn](https://scikit-learn.org/stable/) - Version 1.8.0, December 2025
- [QuestDB Python Docs](https://questdb.com/docs/clients/ingest-python/) - questdb client 3.0.0
- [QuestDB Pandas Integration](https://questdb.com/docs/third-party-tools/pandas/)

### Research Papers and Implementations
- [Almgren-Chriss Original Paper](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf) - "Optimal Execution of Portfolio Transactions"
- [QuestDB Almgren-Chriss Glossary](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/)
- [hftbacktest GitHub](https://github.com/nkaz001/hftbacktest) - 2,543 stars, active development

### Community Resources (MEDIUM confidence)
- [Numba Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- [CVXPY Portfolio Optimization](https://www.cvxpy.org/examples/basic/quadratic_program.html)
- [How to sell stocks wisely - Almgren-Chriss](https://quantjourney.substack.com/p/how-to-sell-stocks-wisely-the-code)

### Assessment References (LOW confidence - used for landscape survey)
- [tcapy GitHub](https://github.com/cuemacro/tcapy) - Last updated 2021, assessed for benchmark patterns
- [QuantStart Backtesting Frameworks](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/)
