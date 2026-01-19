# Architecture Patterns: Execution Analytics Layer

**Domain:** Execution analytics for market data collection framework
**Researched:** 2026-01-19
**Confidence:** HIGH (established patterns, verified against official sources and industry implementations)

## Recommended Architecture

The execution analytics layer sits as a **peer layer** to the existing storage layer, consuming data from QuestDB rather than intercepting the collection pipeline. This follows the Query Layer pattern used by tcapy and similar TCA systems.

```
                         EXISTING                           NEW LAYER
                    +----------------+
                    |    CLI Layer   |
                    +----------------+
                           |
                    +----------------+
                    |   Scheduler    |
                    +----------------+
                           |
                    +----------------+
                    |   Collectors   |
                    +----------------+
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
+----------------+                    +----------------+
| Storage Layer  |                    | Analytics Layer|  <-- NEW
| (QuestDBWriter)|                    | (QuestDBReader)|
+----------------+                    +----------------+
        |                                     |
        +------------------+------------------+
                           |
                           v
                    +----------------+
                    |    QuestDB     |
                    +----------------+
```

**Key principle:** Analytics is a consumer of collected data, not part of the collection pipeline. This ensures:
1. Collection and analytics can evolve independently
2. Analytics failures don't impact data collection
3. Analytics can query historical data without affecting live collection

### Component Architecture

```
src/tributary/analytics/
|-- __init__.py
|-- reader.py              # QuestDB query interface (reads what QuestDBWriter writes)
|-- core/
|   |-- __init__.py
|   |-- models.py          # Analytics-specific Pydantic models
|   |-- exceptions.py      # Analytics exceptions
|-- cost/                  # Cost Analytics Module
|   |-- __init__.py
|   |-- benchmarks.py      # VWAP, TWAP, arrival price calculators
|   |-- slippage.py        # Slippage analysis
|   |-- impact.py          # Market impact estimation
|   |-- shortfall.py       # Implementation shortfall
|-- optimization/          # Optimal Execution Module
|   |-- __init__.py
|   |-- almgren_chriss.py  # Almgren-Chriss model implementation
|   |-- parameters.py      # Parameter calibration from historical data
|   |-- strategies.py      # Execution strategy definitions (TWAP, VWAP, AC)
|-- simulation/            # Execution Simulation Engine
|   |-- __init__.py
|   |-- engine.py          # Core simulation loop
|   |-- events.py          # Event definitions (MarketEvent, OrderEvent, FillEvent)
|   |-- fill_model.py      # Order fill simulation with slippage
|   |-- metrics.py         # Performance measurement
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `reader.py` | Query historical data from QuestDB | QuestDB via PostgreSQL wire protocol |
| `cost/` | Calculate execution cost metrics | reader (data source) |
| `optimization/` | Generate optimal execution schedules | cost/ (for parameter calibration), reader |
| `simulation/` | Backtest execution strategies | reader (market data), optimization/ (strategies) |

## Data Flow

### Flow 1: Cost Analysis Query

```
User Request
    |
    v
analytics.cost.benchmarks.calculate_vwap(market_id, start, end)
    |
    v
reader.query_trades(market_id, start, end)  -- SQL via PostgreSQL protocol
    |
    v
QuestDB: SELECT timestamp, price, size FROM trades WHERE market_id = ? ...
    |
    v
List[Trade] returned to benchmarks module
    |
    v
VWAP = sum(price * size) / sum(size)
    |
    v
CostAnalysisResult returned to user
```

### Flow 2: Almgren-Chriss Parameter Calibration

```
calibration_request(market_id, lookback_days)
    |
    +-- reader.query_trades(market_id, ...) --> List[Trade]
    |
    +-- reader.query_orderbooks(market_id, ...) --> List[OrderBookSnapshot]
    |
    v
parameters.calibrate_from_data()
    |
    +-- Estimate sigma (volatility) from trade price changes
    |
    +-- Estimate gamma (permanent impact) from price reversion analysis
    |
    +-- Estimate eta (temporary impact) from orderbook depth analysis
    |
    v
AlmgrenChrissParameters(sigma, gamma, eta, epsilon)
```

### Flow 3: Execution Simulation

```
simulation_request(strategy, market_id, order_size, horizon)
    |
    v
simulation.engine.SimulationEngine
    |
    +-- Load historical data via reader
    |   |
    |   +-- trades: for price evolution
    |   +-- orderbooks: for liquidity/spread
    |
    +-- Initialize strategy (TWAP, VWAP, AlmgrenChriss)
    |
    v
Event Loop (for each time step):
    |
    +-- MarketEvent: new price/orderbook data
    |       |
    |       v
    +-- Strategy.generate_order(market_state) --> OrderEvent
    |       |
    |       v
    +-- FillModel.simulate_fill(order, orderbook) --> FillEvent
    |       |
    |       v
    +-- Portfolio.update(fill)
    |       |
    |       v
    +-- Metrics.record(fill, benchmark_price)
    |
    v
SimulationResult(trades, metrics, pnl_curve)
```

## Patterns to Follow

### Pattern 1: Request Object Pattern (from tcapy)

Define explicit request objects for all analytics operations. This allows:
- Clear interface contracts
- Easy serialization for CLI/API
- Validation at the boundary

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class CostAnalysisRequest(BaseModel):
    """Request for cost analysis computation."""
    market_id: str
    start_time: datetime
    end_time: datetime
    benchmark: str = "vwap"  # vwap, twap, arrival
    include_market_impact: bool = True

class CostAnalysisResult(BaseModel):
    """Result of cost analysis."""
    market_id: str
    benchmark_price: float
    slippage_bps: float
    market_impact_bps: Optional[float]
    total_cost_bps: float
    trade_count: int
    total_volume: float
```

### Pattern 2: Strategy Abstract Base Class

All execution strategies implement a common interface, enabling the simulation engine to be strategy-agnostic.

```python
from abc import ABC, abstractmethod
from typing import List
from datetime import datetime

class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""

    @abstractmethod
    def generate_schedule(
        self,
        total_shares: float,
        start_time: datetime,
        end_time: datetime,
        intervals: int,
    ) -> List[ScheduledTrade]:
        """Generate the full execution schedule upfront."""
        pass

    @abstractmethod
    def get_next_trade(
        self,
        current_time: datetime,
        remaining_shares: float,
        market_state: MarketState,
    ) -> Optional[TradeOrder]:
        """Get next trade based on current market conditions (adaptive)."""
        pass

class TWAPStrategy(ExecutionStrategy):
    """Time-Weighted Average Price - equal slices over time."""
    ...

class VWAPStrategy(ExecutionStrategy):
    """Volume-Weighted Average Price - follows historical volume profile."""
    ...

class AlmgrenChrissStrategy(ExecutionStrategy):
    """Optimal execution minimizing impact + risk."""
    ...
```

### Pattern 3: Separate Calibration from Execution

Parameter calibration is a distinct operation from strategy execution. Keep them separate:

```python
# Calibration: runs periodically or on-demand
params = calibrate_almgren_chriss_params(market_id, lookback_days=30)

# Execution: uses pre-calibrated parameters
strategy = AlmgrenChrissStrategy(params)
schedule = strategy.generate_schedule(shares, start, end, intervals)
```

This separation allows:
- Caching calibrated parameters
- A/B testing different calibration methods
- Auditing parameter history

### Pattern 4: Event-Driven Simulation (from QSTrader)

The simulation engine uses an event queue to decouple components:

```python
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Any

class EventType(Enum):
    MARKET = "market"
    ORDER = "order"
    FILL = "fill"

@dataclass
class Event:
    """Base event for simulation engine."""
    event_type: EventType
    timestamp: datetime
    data: Any

class SimulationEngine:
    """Event-driven execution simulation engine."""

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: ExecutionStrategy,
        fill_model: FillModel,
        metrics: MetricsCollector,
    ):
        self.event_queue: Queue[Event] = Queue()
        self.data_handler = data_handler
        self.strategy = strategy
        self.fill_model = fill_model
        self.metrics = metrics

    def run(self) -> SimulationResult:
        """Run simulation to completion."""
        while self.data_handler.has_next():
            # Generate market event
            market_event = self.data_handler.next()
            self.event_queue.put(market_event)

            # Process events
            while not self.event_queue.empty():
                event = self.event_queue.get()
                self._process_event(event)

        return self.metrics.get_results()
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Tight Coupling to Collection Pipeline

**What:** Adding analytics hooks directly into the collector or scheduler.

**Why bad:**
- Analytics failures could disrupt data collection
- Analytics typically needs historical data, not just live stream
- Different performance characteristics (collection: high throughput writes; analytics: complex reads)

**Instead:** Analytics layer reads from QuestDB independently.

### Anti-Pattern 2: Monolithic "Analytics" Class

**What:** Single class handling benchmarks, impact estimation, optimization, and simulation.

**Why bad:**
- Hard to test components in isolation
- Difficult to extend or replace individual components
- Cognitive overload

**Instead:** Separate modules for cost/, optimization/, simulation/ with clear interfaces.

### Anti-Pattern 3: Hardcoding Market-Specific Logic

**What:** Embedding Polymarket-specific assumptions (e.g., binary outcomes, 0-1 price bounds) into core analytics.

**Why bad:**
- PROJECT.md explicitly requires extensibility to other asset classes
- Prediction markets have different characteristics than crypto spot

**Instead:**
- Analytics core operates on generic Trade/OrderBookSnapshot models
- Market-specific adaptations via configuration or subclasses
- Validate assumptions at boundaries

### Anti-Pattern 4: Simulation Without Realistic Fill Model

**What:** Assuming orders fill at mid-price or instantly.

**Why bad:**
- Overstates strategy performance
- Prediction markets have thin liquidity - fill quality matters

**Instead:**
- Fill model consumes orderbook depth
- Simulates walking the book for market orders
- Models partial fills for limit orders

## QuestDB Query Patterns

### Reader Interface Design

```python
from typing import List, Optional
from datetime import datetime
import pandas as pd

class QuestDBReader:
    """Read interface for analytics queries against QuestDB."""

    def __init__(self, host: str, port: int = 8812):
        """Initialize with PostgreSQL wire protocol connection."""
        self.connection_string = f"postgresql://admin:quest@{host}:{port}/qdb"

    def query_trades(
        self,
        market_id: str,
        start_time: datetime,
        end_time: datetime,
        token_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query trades for analysis.

        Uses QuestDB's time-based partitioning for efficient range queries.
        """
        query = """
        SELECT timestamp, market_id, token_id, price, size, side, value
        FROM trades
        WHERE market_id = $1
          AND timestamp >= $2
          AND timestamp < $3
        ORDER BY timestamp
        """
        # Execute via psycopg or similar
        ...

    def query_vwap(
        self,
        market_id: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Calculate VWAP using QuestDB's SAMPLE BY.

        Leverages database-side aggregation for efficiency.
        """
        query = """
        SELECT
            timestamp,
            sum(price * size) / sum(size) as vwap,
            sum(size) as volume,
            count() as trade_count
        FROM trades
        WHERE market_id = $1
          AND timestamp >= $2
          AND timestamp < $3
        SAMPLE BY $4
        """
        ...

    def query_orderbook_depth(
        self,
        market_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Query orderbook snapshots for liquidity analysis.

        Used for market impact estimation and fill simulation.
        """
        query = """
        SELECT
            timestamp,
            token_id,
            best_bid,
            best_ask,
            spread_bps,
            total_bid_volume,
            total_ask_volume,
            bid_prices,
            bid_sizes,
            ask_prices,
            ask_sizes
        FROM orderbook_snapshots
        WHERE market_id = $1
          AND timestamp >= $2
          AND timestamp < $3
        ORDER BY timestamp
        """
        ...
```

### Key QuestDB Features for Analytics

| Feature | Use Case | Example |
|---------|----------|---------|
| `SAMPLE BY` | VWAP/TWAP calculation | `SAMPLE BY 15m` for 15-minute buckets |
| `first()/last()` | OHLC bars | `first(price) as open, last(price) as close` |
| `stddev_samp()` | Volatility estimation | Price return standard deviation |
| Time partitioning | Efficient range queries | Daily partitions on trades table |
| PostgreSQL wire | Python connectivity | psycopg2/psycopg for queries |

## Almgren-Chriss Implementation Structure

### Parameters Model

```python
from pydantic import BaseModel, Field

class AlmgrenChrissParameters(BaseModel):
    """
    Almgren-Chriss model parameters.

    Calibrated from historical market data.
    """
    sigma: float = Field(..., description="Price volatility (annualized)")
    gamma: float = Field(..., description="Permanent impact coefficient")
    eta: float = Field(..., description="Temporary impact coefficient")
    epsilon: float = Field(0.0, description="Fixed cost per trade")
    lambda_: float = Field(..., description="Risk aversion parameter")

    # Metadata for auditability
    market_id: str
    calibration_date: datetime
    lookback_days: int
    trade_count: int  # How many trades were used for calibration
```

### Calibration Logic

```python
class AlmgrenChrissCalibrator:
    """
    Calibrate Almgren-Chriss parameters from historical data.

    Approach:
    1. sigma: Estimate from log returns of trade prices
    2. gamma: Estimate from permanent price impact (requires signed trades)
    3. eta: Estimate from temporary impact via orderbook depth
    """

    def __init__(self, reader: QuestDBReader):
        self.reader = reader

    def calibrate(
        self,
        market_id: str,
        lookback_days: int = 30,
        lambda_: float = 1e-6,
    ) -> AlmgrenChrissParameters:
        """
        Calibrate parameters for a specific market.

        Args:
            market_id: Market to calibrate for
            lookback_days: Historical data window
            lambda_: Risk aversion (user-specified, not calibrated)

        Returns:
            Calibrated parameters
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)

        # Fetch data
        trades = self.reader.query_trades(market_id, start_time, end_time)
        orderbooks = self.reader.query_orderbook_depth(market_id, start_time, end_time)

        # Estimate sigma from trade price volatility
        sigma = self._estimate_volatility(trades)

        # Estimate gamma from permanent impact
        gamma = self._estimate_permanent_impact(trades)

        # Estimate eta from orderbook depth
        eta = self._estimate_temporary_impact(orderbooks, trades)

        return AlmgrenChrissParameters(
            sigma=sigma,
            gamma=gamma,
            eta=eta,
            epsilon=0.0,  # TODO: estimate from fee structure
            lambda_=lambda_,
            market_id=market_id,
            calibration_date=datetime.utcnow(),
            lookback_days=lookback_days,
            trade_count=len(trades),
        )

    def _estimate_volatility(self, trades: pd.DataFrame) -> float:
        """
        Estimate price volatility from trade data.

        Uses log returns sampled at regular intervals to avoid
        microstructure noise from tick-by-tick returns.
        """
        # Resample to 1-hour intervals
        hourly = trades.set_index('timestamp').resample('1h')['price'].last().dropna()
        log_returns = np.log(hourly / hourly.shift(1)).dropna()

        # Annualize (assuming ~8760 hours/year)
        return log_returns.std() * np.sqrt(8760)

    def _estimate_permanent_impact(self, trades: pd.DataFrame) -> float:
        """
        Estimate permanent impact coefficient.

        This is challenging without metaorder data. Use regression of
        price changes on signed volume.
        """
        # Simplified: regress price change on signed trade size
        trades['signed_size'] = trades.apply(
            lambda x: x['size'] if x['side'] == 'buy' else -x['size'],
            axis=1
        )
        trades['price_change'] = trades['price'].diff()

        # gamma = cov(delta_P, signed_volume) / var(signed_volume)
        # This is a rough approximation
        ...

    def _estimate_temporary_impact(
        self,
        orderbooks: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> float:
        """
        Estimate temporary impact from orderbook depth.

        Approach: Average cost to execute typical trade size against
        orderbook depth.
        """
        avg_trade_size = trades['size'].mean()

        # Calculate average spread and depth
        avg_spread_bps = orderbooks['spread_bps'].mean()
        avg_depth = (orderbooks['total_bid_volume'] + orderbooks['total_ask_volume']).mean() / 2

        # Simplified: eta proportional to spread / depth
        # Real implementation should simulate walking the book
        ...
```

### Strategy Implementation

```python
class AlmgrenChrissStrategy(ExecutionStrategy):
    """
    Optimal execution strategy using Almgren-Chriss framework.

    Generates a trading schedule that minimizes expected cost + risk penalty.
    """

    def __init__(self, params: AlmgrenChrissParameters):
        self.params = params

    def generate_schedule(
        self,
        total_shares: float,
        start_time: datetime,
        end_time: datetime,
        intervals: int,
    ) -> List[ScheduledTrade]:
        """
        Generate optimal trading trajectory.

        The Almgren-Chriss solution is "front-loaded" - most trading
        happens early to reduce exposure to price risk.
        """
        tau = (end_time - start_time).total_seconds() / intervals

        # Calculate decay rate (kappa)
        kappa = self._calculate_decay_rate(tau)

        # Generate trajectory
        trajectory = []
        remaining = total_shares
        current_time = start_time

        for i in range(intervals):
            # Optimal trade size at each step
            trade_size = self._optimal_trade_size(
                remaining, i, intervals, kappa, tau
            )

            trajectory.append(ScheduledTrade(
                time=current_time,
                size=trade_size,
                remaining_after=remaining - trade_size,
            ))

            remaining -= trade_size
            current_time += timedelta(seconds=tau)

        return trajectory

    def _calculate_decay_rate(self, tau: float) -> float:
        """
        Calculate the characteristic decay rate kappa.

        kappa = sqrt(lambda * sigma^2 / eta)
        """
        return np.sqrt(
            self.params.lambda_ * self.params.sigma**2 / self.params.eta
        )

    def _optimal_trade_size(
        self,
        remaining: float,
        step: int,
        total_steps: int,
        kappa: float,
        tau: float,
    ) -> float:
        """
        Calculate optimal trade size at given step.

        Based on the Almgren-Chriss closed-form solution.
        """
        # Simplified: exponential decay
        decay = np.exp(-kappa * tau * step)
        normalization = sum(np.exp(-kappa * tau * j) for j in range(total_steps))
        return remaining * decay / normalization
```

## Scalability Considerations

| Concern | Current Scale (10 markets) | Medium Scale (100 markets) | Large Scale (1000+ markets) |
|---------|---------------------------|----------------------------|----------------------------|
| Query latency | Direct psycopg queries | Connection pooling | Consider read replicas |
| Parameter caching | Recalculate on demand | Cache in memory | Cache in Redis with TTL |
| Simulation speed | Sequential processing | Parallel per-market | Distributed via Celery |
| Data volume | Full orderbook history | Sample older data | Aggregate older snapshots |

## Suggested Build Order

Based on component dependencies:

```
Phase 1: Foundation
+-- reader.py (QuestDB query interface)
+-- cost/benchmarks.py (VWAP, TWAP)

Phase 2: Cost Analytics
+-- cost/slippage.py (requires benchmarks)
+-- cost/impact.py (requires reader, orderbook queries)
+-- cost/shortfall.py (requires benchmarks, slippage)

Phase 3: Optimization Core
+-- optimization/parameters.py (calibration, requires reader)
+-- optimization/almgren_chriss.py (requires parameters)
+-- optimization/strategies.py (TWAP, VWAP, AC strategies)

Phase 4: Simulation Engine
+-- simulation/events.py (event definitions)
+-- simulation/fill_model.py (requires orderbook data)
+-- simulation/engine.py (requires all above)
+-- simulation/metrics.py (compare strategies)
```

**Rationale:**
1. Reader first - everything depends on querying data
2. Benchmarks second - needed to measure anything
3. Cost analytics builds on benchmarks
4. Optimization needs calibration from historical analysis
5. Simulation ties everything together for backtesting

## Integration with Existing Architecture

### CLI Extension

Add analytics commands to existing CLI structure:

```python
# In tributary/cli/commands.py

@main.group()
def analytics():
    """Execution analytics commands."""
    pass

@analytics.command()
@click.option("--market", required=True)
@click.option("--days", default=7)
def cost_report(market: str, days: int):
    """Generate cost analysis report for a market."""
    ...

@analytics.command()
@click.option("--market", required=True)
@click.option("--lookback", default=30)
def calibrate(market: str, lookback: int):
    """Calibrate Almgren-Chriss parameters."""
    ...

@analytics.command()
@click.option("--market", required=True)
@click.option("--strategy", type=click.Choice(["twap", "vwap", "ac"]))
@click.option("--shares", type=float, required=True)
@click.option("--horizon", default="1h")
def simulate(market: str, strategy: str, shares: float, horizon: str):
    """Run execution simulation."""
    ...
```

### Configuration Extension

Add analytics configuration to existing YAML structure:

```yaml
# config/settings.yaml
analytics:
  enabled: true

  calibration:
    default_lookback_days: 30
    recalibrate_interval_hours: 24
    min_trades_required: 100

  simulation:
    default_fill_model: "orderbook"  # or "midpoint", "spread"
    slippage_multiplier: 1.0  # Scale simulated slippage
```

## Sources

- [Almgren-Chriss PyPI Package](https://pypi.org/project/almgren-chriss/) - Reference implementation with API
- [tcapy GitHub Repository](https://github.com/cuemacro/tcapy) - TCA architecture patterns
- [QuestDB Aggregate Functions](https://questdb.com/docs/query/functions/aggregation/) - Query patterns for time-series
- [QuantStart Event-Driven Backtesting](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/) - Simulation engine patterns
- [QSTrader GitHub](https://github.com/mhallsmoore/qstrader) - Modular backtesting architecture
- [Talos Market Impact Model](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs) - Crypto-specific impact modeling
