# Phase 4: Simulation - Research

**Researched:** 2026-01-19
**Domain:** Event-driven backtesting, execution simulation, fill models, strategy comparison
**Confidence:** HIGH

## Summary

Phase 4 implements a backtesting engine to prove that optimized execution strategies (Almgren-Chriss, TWAP, VWAP) outperform naive approaches (market orders). The simulation must be event-driven to capture timing behavior differences between strategies, use realistic fill models that account for orderbook depth and liquidity consumption, and provide clear comparison metrics.

The project already has all the building blocks from prior phases: execution strategies generate `ExecutionTrajectory` objects (Phase 3), walk-the-book cost forecasting provides fill simulation primitives (Phase 2), and QuestDB reader enables historical data replay (Phase 1). Phase 4 connects these components through an event-driven simulation loop.

The key insight from research: event-driven architecture is essential because it processes each market event in sequence, eliminating lookahead bias and enabling realistic timing simulation. The fill model must consume liquidity from the orderbook and model partial recovery between slices to capture the real cost differences between aggressive (market order) and patient (A-C/TWAP) strategies.

**Primary recommendation:** Build a lightweight event-driven simulation engine using Python's `collections.deque` as the event queue, leveraging existing `estimate_slippage_from_orderbook()` for fill simulation, and existing `ShortfallComponents` for metric calculation. No external backtesting framework needed.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| collections.deque | builtin | Event queue (FIFO) | Optimal O(1) append/popleft |
| numpy | >=1.26 | Numeric operations, arrays | Already in project |
| pandas | >=2.0 | Time series, result DataFrames | Already in project |
| dataclasses | builtin | Event types, result containers | Project convention |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | >=3.8 | Execution charts (optional) | notebooks extra |
| plotly | >=5.0 | Interactive charts (optional) | notebooks extra |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom engine | NautilusTrader | Overkill for analytics-only, adds 100+ MB dependency |
| Custom engine | Backtrader/Zipline | Strategy-focused, not execution-focused |
| Custom engine | hftbacktest | Rust-based, complex setup, HFT-specific |
| deque | queue.Queue | Thread-safe but unnecessary overhead for single-threaded |

**Installation:**
No new dependencies required. All needed libraries already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure
```
src/tributary/analytics/
├── simulation/               # NEW: Phase 4 module
│   ├── __init__.py
│   ├── events.py            # SIM-01: Event types (MarketEvent, OrderEvent, FillEvent)
│   ├── engine.py            # SIM-01: Event-driven simulation loop
│   ├── fill_model.py        # SIM-02: Realistic fill simulation
│   ├── runner.py            # SIM-03: Multi-strategy simulation runner
│   ├── metrics.py           # SIM-04: Comparison metrics calculation
│   └── results.py           # SIM-05: Results aggregation and comparison
├── optimization/            # Existing: ExecutionTrajectory, strategies
├── cost_forecast.py         # Existing: estimate_slippage_from_orderbook()
├── shortfall.py             # Existing: ShortfallComponents
└── reader.py                # Existing: QuestDB data access
```

### Pattern 1: Event Types as Frozen Dataclasses
**What:** Define distinct event types for clean message passing
**When to use:** All events flowing through the simulation
**Example:**
```python
# Source: QuantStart event-driven architecture + project conventions
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class MarketEvent:
    """Orderbook update event from historical data replay."""
    timestamp: datetime
    market_id: str
    token_id: str
    mid_price: float
    bid_prices: tuple[float, ...]
    bid_sizes: tuple[float, ...]
    ask_prices: tuple[float, ...]
    ask_sizes: tuple[float, ...]

@dataclass(frozen=True)
class OrderEvent:
    """Order to be executed in the simulation."""
    timestamp: datetime
    strategy_name: str
    slice_index: int
    size: float
    side: str  # 'buy' or 'sell'

@dataclass(frozen=True)
class FillEvent:
    """Execution result from fill model."""
    timestamp: datetime
    strategy_name: str
    slice_index: int
    requested_size: float
    filled_size: float
    avg_price: float
    slippage_bps: float
    levels_consumed: int
```

### Pattern 2: Event-Driven Loop with deque
**What:** Central event queue processes events in FIFO order
**When to use:** Core simulation engine
**Example:**
```python
# Source: QuantStart architecture adapted for execution simulation
from collections import deque
from typing import Iterator

class SimulationEngine:
    """Event-driven execution simulation engine."""

    def __init__(self):
        self.event_queue: deque = deque()
        self.current_time: datetime = None

    def run(self, market_events: Iterator[MarketEvent]) -> list[FillEvent]:
        """Run simulation processing all events in order."""
        fills = []

        for market_event in market_events:
            # Add market event to queue
            self.event_queue.append(market_event)

            # Process all events at this timestamp
            while self.event_queue:
                event = self.event_queue.popleft()

                if isinstance(event, MarketEvent):
                    self._handle_market_event(event)
                elif isinstance(event, OrderEvent):
                    fill = self._handle_order_event(event)
                    if fill:
                        fills.append(fill)

        return fills
```

### Pattern 3: Fill Model with Liquidity Consumption
**What:** Simulate fills that consume orderbook depth with partial recovery
**When to use:** Translating OrderEvents to FillEvents
**Example:**
```python
# Source: CONTEXT.md decisions + existing cost_forecast.py
from tributary.analytics.cost_forecast import estimate_slippage_from_orderbook

class FillModel:
    """Realistic fill model using walk-the-book with impact + recovery."""

    def __init__(self, recovery_rate: float = 0.5):
        """
        Args:
            recovery_rate: Fraction of consumed liquidity that recovers
                          between slices (0.0 = no recovery, 1.0 = full recovery)
        """
        self.recovery_rate = recovery_rate
        self.consumed_liquidity: dict[str, float] = {}  # side -> depth consumed

    def execute(
        self,
        order: OrderEvent,
        orderbook: MarketEvent,
        time_since_last_slice: float,
    ) -> FillEvent:
        """Execute order against current orderbook state."""

        # Apply recovery to consumed liquidity
        self._apply_recovery(time_since_last_slice)

        # Adjust orderbook for previously consumed liquidity
        adjusted_book = self._adjust_orderbook(orderbook, order.side)

        # Walk the book
        forecast = estimate_slippage_from_orderbook(
            order_size=order.size,
            side=order.side,
            bid_prices=list(adjusted_book.bid_prices),
            bid_sizes=list(adjusted_book.bid_sizes),
            ask_prices=list(adjusted_book.ask_prices),
            ask_sizes=list(adjusted_book.ask_sizes),
        )

        # Track consumed liquidity for impact modeling
        self._track_consumption(order.side, forecast.levels_consumed)

        return FillEvent(
            timestamp=order.timestamp,
            strategy_name=order.strategy_name,
            slice_index=order.slice_index,
            requested_size=order.size,
            filled_size=order.size - forecast.unfilled_size,
            avg_price=forecast.expected_execution_price,
            slippage_bps=forecast.slippage_bps,
            levels_consumed=forecast.levels_consumed,
        )
```

### Pattern 4: Strategy Runner for Parallel Comparison
**What:** Run multiple strategies on same historical data in parallel isolation
**When to use:** SIM-03 multi-strategy comparison
**Example:**
```python
# Source: CONTEXT.md decision - parallel isolated execution
from tributary.analytics.optimization import ExecutionTrajectory

@dataclass
class StrategyRun:
    """One strategy's execution through the simulation."""
    trajectory: ExecutionTrajectory
    fills: list[FillEvent]

class StrategyRunner:
    """Run multiple strategies on same market data."""

    def run_strategies(
        self,
        strategies: list[ExecutionTrajectory],
        market_data: pd.DataFrame,
        side: str,
    ) -> list[StrategyRun]:
        """
        Run all strategies on the same historical data.
        Each strategy sees a clean orderbook (no cross-strategy impact).
        """
        results = []

        for strategy in strategies:
            # Fresh fill model for each strategy (isolated execution)
            fill_model = FillModel(recovery_rate=0.5)
            engine = SimulationEngine(fill_model=fill_model)

            fills = engine.run(
                trajectory=strategy,
                market_data=market_data,
                side=side,
            )

            results.append(StrategyRun(trajectory=strategy, fills=fills))

        return results
```

### Anti-Patterns to Avoid
- **Vectorized simulation:** Loses timing behavior differences between strategies
- **Shared orderbook state:** Strategies should not impact each other's execution
- **Perfect fills assumption:** Must model partial fills and slippage
- **Ignoring recovery:** Orderbook replenishes between slices; pure depletion is unrealistic
- **Single metric comparison:** Need multiple benchmarks (IS, VWAP slippage, risk)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Walk-the-book fill | Custom orderbook traversal | `estimate_slippage_from_orderbook()` | Already tested, handles edge cases |
| Shortfall decomposition | Custom cost breakdown | `decompose_implementation_shortfall()` | Perold framework already implemented |
| Trajectory generation | Custom slice calculation | `generate_ac_trajectory()`, `generate_twap_trajectory()` | Phase 3 already built these |
| Strategy comparison table | Custom DataFrame building | `StrategyComparison.summary_table()` | Phase 3 already built this |
| Event queue | Custom list management | `collections.deque` | O(1) operations, battle-tested |
| Time series plotting | Custom matplotlib | Existing `execution_profile_chart()` | Phase 3 provides chart data |

**Key insight:** Phase 4 connects existing components more than it creates new algorithms. The simulation engine orchestrates what's already built.

## Common Pitfalls

### Pitfall 1: Lookahead Bias
**What goes wrong:** Simulation uses future orderbook data to make decisions
**Why it happens:** Non-event-driven architecture processes all data at once
**How to avoid:**
- Strict event ordering: process events in timestamp order
- Strategy receives only orderbook state at decision time
- Never query future prices when generating orders
**Warning signs:** Strategies perform identically, impossibly good results

### Pitfall 2: No Market Impact Differentiation
**What goes wrong:** Market order and TWAP show similar costs in backtest
**Why it happens:** Fill model doesn't properly deplete liquidity
**How to avoid:**
- Market order should consume deep into the book (high slippage)
- TWAP slices should be small relative to top-of-book (low slippage)
- Validate: large market order slippage >> TWAP slippage
**Warning signs:** All strategies have nearly identical total costs

### Pitfall 3: Unrealistic Full Fills
**What goes wrong:** All orders fill completely regardless of size vs liquidity
**Why it happens:** Fill model ignores orderbook depth
**How to avoid:**
- Check available liquidity before filling
- Allow partial fills when orderbook is exhausted
- Track unfilled size and carry forward or cancel
**Warning signs:** No partial fills even with large orders on thin books

### Pitfall 4: Static Orderbook Throughout Execution
**What goes wrong:** Same orderbook used for all slices
**Why it happens:** Not replaying historical orderbook evolution
**How to avoid:**
- Each slice executes against the orderbook at that timestamp
- Replay orderbook snapshots in time order
- Model liquidity recovery between slices
**Warning signs:** Slippage doesn't vary across slices

### Pitfall 5: Metrics Without Risk Consideration
**What goes wrong:** Rank strategies by cost alone, ignoring variance
**Why it happens:** Forgetting A-C is a mean-variance framework
**How to avoid:**
- Calculate both expected cost AND cost variance
- Compute risk-adjusted metrics (Sharpe-like for execution)
- Show full risk profile: max drawdown during execution, worst-case
**Warning signs:** Market order sometimes ranked better than A-C

### Pitfall 6: Comparing Apples and Oranges
**What goes wrong:** Strategy results not comparable
**Why it happens:** Different order sizes, durations, or start times
**How to avoid:**
- All strategies execute same total order size
- All strategies run on exact same time window
- Arrival price fixed at simulation start (same benchmark)
**Warning signs:** Order sizes don't sum to same total

## Code Examples

Verified patterns from research and project context:

### SimulationResult Dataclass
```python
# Source: Project convention + CONTEXT.md metrics requirements
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SimulationResult:
    """Complete result of simulating one strategy."""
    strategy_name: str
    total_order_size: float
    side: str

    # Execution summary
    total_filled: float
    total_unfilled: float
    num_slices: int
    num_partial_fills: int

    # Cost metrics
    arrival_price: float
    avg_execution_price: float
    implementation_shortfall_bps: float
    vwap_slippage_bps: float
    total_cost_usd: float

    # Risk metrics
    cost_variance: float
    max_drawdown_bps: float
    worst_slice_slippage_bps: float

    # Detailed data
    fills: tuple  # tuple of FillEvents for frozen

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        if self.total_order_size == 0:
            return 0.0
        return self.total_filled / self.total_order_size * 100
```

### Metrics Calculator
```python
# Source: Existing shortfall.py + benchmarks.py patterns
from tributary.analytics.shortfall import decompose_implementation_shortfall
from tributary.analytics.benchmarks import calculate_vwap

def calculate_simulation_metrics(
    fills: list[FillEvent],
    arrival_price: float,
    total_order_size: float,
    side: str,
    market_vwap: float,
) -> dict:
    """
    Calculate comprehensive execution metrics.

    Returns dict with:
    - implementation_shortfall_bps: vs arrival price
    - vwap_slippage_bps: vs market VWAP
    - cost_variance: variance of slice costs
    - max_drawdown_bps: worst cumulative cost during execution
    """
    if not fills:
        return {
            "implementation_shortfall_bps": float("nan"),
            "vwap_slippage_bps": float("nan"),
            "cost_variance": float("nan"),
            "max_drawdown_bps": float("nan"),
        }

    # Extract execution data
    prices = [f.avg_price for f in fills]
    sizes = [f.filled_size for f in fills]

    # Calculate VWAP of execution
    total_value = sum(p * s for p, s in zip(prices, sizes))
    total_size = sum(sizes)
    exec_vwap = total_value / total_size if total_size > 0 else float("nan")

    # Implementation shortfall vs arrival price
    if side == "buy":
        is_bps = (exec_vwap - arrival_price) / arrival_price * 10000
        vwap_slip_bps = (exec_vwap - market_vwap) / market_vwap * 10000
    else:
        is_bps = (arrival_price - exec_vwap) / arrival_price * 10000
        vwap_slip_bps = (market_vwap - exec_vwap) / market_vwap * 10000

    # Per-slice slippage for variance calculation
    slice_slippages = [f.slippage_bps for f in fills]
    cost_variance = np.var(slice_slippages) if len(slice_slippages) > 1 else 0.0

    # Max drawdown: worst cumulative cost during execution
    cumulative_costs = np.cumsum(slice_slippages)
    max_drawdown_bps = float(np.max(cumulative_costs))

    return {
        "implementation_shortfall_bps": is_bps,
        "vwap_slippage_bps": vwap_slip_bps,
        "cost_variance": cost_variance,
        "max_drawdown_bps": max_drawdown_bps,
    }
```

### Strategy Comparison Table
```python
# Source: CONTEXT.md requirements - table + risk-adjusted ranking
def compare_simulation_results(
    results: list[SimulationResult],
    rank_by: str = "risk_adjusted",
) -> pd.DataFrame:
    """
    Compare multiple strategy simulation results.

    Args:
        results: List of SimulationResult objects
        rank_by: 'cost' (IS only), 'risk' (variance), or 'risk_adjusted'

    Returns:
        DataFrame with comparison metrics, ranked by selected criterion
    """
    rows = []
    for r in results:
        # Risk-adjusted score: cost / sqrt(variance)
        # Lower is better (less cost per unit risk)
        if r.cost_variance > 0:
            risk_adj = r.implementation_shortfall_bps / np.sqrt(r.cost_variance)
        else:
            risk_adj = r.implementation_shortfall_bps

        rows.append({
            "strategy": r.strategy_name,
            "is_bps": r.implementation_shortfall_bps,
            "vwap_slip_bps": r.vwap_slippage_bps,
            "cost_variance": r.cost_variance,
            "max_drawdown_bps": r.max_drawdown_bps,
            "fill_rate_pct": r.fill_rate,
            "risk_adjusted_score": risk_adj,
        })

    df = pd.DataFrame(rows)

    # Sort by selected criterion (lower is better)
    sort_col = {
        "cost": "is_bps",
        "risk": "cost_variance",
        "risk_adjusted": "risk_adjusted_score",
    }.get(rank_by, "risk_adjusted_score")

    return df.sort_values(sort_col).reset_index(drop=True)
```

### Execution Chart Data
```python
# Source: CONTEXT.md - execution curves (holdings over time, cost accumulation)
def execution_chart_data(
    results: list[SimulationResult],
) -> pd.DataFrame:
    """
    Generate long-format DataFrame for execution visualization.

    Returns DataFrame with columns:
    - timestamp: execution time
    - strategy: strategy name
    - holdings_pct: remaining holdings as % of order
    - cumulative_cost_bps: cost accumulated so far
    """
    rows = []

    for r in results:
        remaining = r.total_order_size
        cumulative_cost = 0.0

        # Initial state
        rows.append({
            "timestamp": r.fills[0].timestamp if r.fills else None,
            "strategy": r.strategy_name,
            "holdings_pct": 100.0,
            "cumulative_cost_bps": 0.0,
        })

        for fill in r.fills:
            remaining -= fill.filled_size
            cumulative_cost += fill.slippage_bps * (fill.filled_size / r.total_order_size)

            rows.append({
                "timestamp": fill.timestamp,
                "strategy": r.strategy_name,
                "holdings_pct": remaining / r.total_order_size * 100,
                "cumulative_cost_bps": cumulative_cost,
            })

    return pd.DataFrame(rows)
```

### Liquidity Recovery Model
```python
# Source: CONTEXT.md decision - impact + recovery model
def apply_liquidity_recovery(
    consumed: dict[str, list[float]],
    time_elapsed_ms: float,
    recovery_rate: float = 0.5,
    half_life_ms: float = 1000.0,
) -> dict[str, list[float]]:
    """
    Model partial liquidity recovery between execution slices.

    Liquidity replenishes exponentially toward original depth.

    Args:
        consumed: Dict mapping price level to consumed amount
        time_elapsed_ms: Time since last execution
        recovery_rate: Maximum fraction that can recover (0-1)
        half_life_ms: Time for half of recoverable liquidity to return

    Returns:
        Updated consumed amounts after recovery
    """
    if time_elapsed_ms <= 0:
        return consumed

    # Exponential decay of consumption
    decay = 0.5 ** (time_elapsed_ms / half_life_ms)
    recovery_factor = 1 - decay * (1 - recovery_rate)

    return {
        level: [amt * (1 - recovery_factor) for amt in amounts]
        for level, amounts in consumed.items()
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Vectorized backtest | Event-driven backtest | Standard since ~2015 | Eliminates lookahead bias |
| Perfect fill assumption | Queue position modeling | HFT research ~2018 | More realistic fills |
| Single benchmark (IS) | Multiple benchmarks | Industry standard | Better strategy comparison |
| Cost-only ranking | Risk-adjusted ranking | Almgren-Chriss framework | Aligns with theory |
| Static orderbook | Orderbook replay with impact | Research-grade simulators | Captures real dynamics |

**Deprecated/outdated:**
- Vectorized backtesting: Fast but unrealistic for execution strategies
- "Fill at mid" assumption: Ignores spread and depth entirely
- Single metric comparison: Misses risk dimension

## Open Questions

Things that couldn't be fully resolved:

1. **Liquidity recovery rate calibration**
   - What we know: Orderbooks replenish after liquidity consumption
   - What's unclear: Optimal recovery rate for prediction markets (50%? 80%?)
   - Recommendation: Start with 50%, allow configuration, validate against real execution

2. **Queue position estimation without L3 data**
   - What we know: FIFO queue affects limit order fills
   - What's unclear: How to model queue advancement with only L2 (price+size) data
   - Recommendation: Use conservative model - queue advances only on trades at price. For this project's market orders / aggressive execution, queue position is less critical than for passive strategies.

3. **Cross-asset impact correlation**
   - What we know: Trading YES may impact NO prices
   - What's unclear: Magnitude and timing of cross-token correlation
   - Recommendation: Defer to v2 (PORT-02). Phase 4 simulates single-token execution.

4. **Prediction market volume profile stability**
   - What we know: Events cause volume spikes (unlike equity markets)
   - What's unclear: Whether historical profiles predict future volume
   - Recommendation: VWAP strategy may underperform; document limitation

## Sources

### Primary (HIGH confidence)
- [QuantStart Event-Driven Backtesting Series](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/) - Architecture patterns, event types
- [HftBacktest Order Fill Documentation](https://hftbacktest.readthedocs.io/en/latest/order_fill.html) - Fill simulation models
- [HftBacktest Queue Models Tutorial](https://hftbacktest.readthedocs.io/en/latest/tutorials/Probability%20Queue%20Models.html) - Queue position estimation
- Project's existing `cost_forecast.py` - Walk-the-book implementation
- Project's existing `shortfall.py` - Perold framework implementation
- Project's `04-CONTEXT.md` - User decisions constraining implementation

### Secondary (MEDIUM confidence)
- [Moallemi & Yuan Queue Position Model](https://moallemi.com/ciamac/papers/queue-value-2016.pdf) - Academic foundation for queue modeling
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) - Production-grade architecture reference
- [arXiv Limit Order Book Simulations Review](https://arxiv.org/html/2402.17359v1) - Survey of simulation techniques

### Tertiary (LOW confidence)
- Medium articles on event-driven backtesters - Implementation examples (needs validation)
- QuantReplay documentation - Multi-asset simulation concepts

## Metadata

**Confidence breakdown:**
- Event-driven architecture: HIGH - Well-established pattern, QuantStart series verified
- Fill model (walk-the-book): HIGH - Already implemented in project, just need to call it
- Liquidity recovery: MEDIUM - Conceptually sound, specific rate needs calibration
- Metrics calculation: HIGH - Builds on existing shortfall.py
- Queue position modeling: LOW - Complex topic, simplified for v1

**Research date:** 2026-01-19
**Valid until:** 60 days (stable patterns, project-specific constraints won't change)
