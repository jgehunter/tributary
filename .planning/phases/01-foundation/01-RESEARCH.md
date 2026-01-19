# Phase 1: Foundation - Research

**Researched:** 2026-01-19
**Domain:** QuestDB reader interface, VWAP/TWAP benchmarks, arrival price
**Confidence:** HIGH

## Summary

Phase 1 establishes the analytics layer foundation by creating a QuestDB reader (peer to the existing writer) and implementing core benchmark calculations (VWAP, TWAP, arrival price). The existing project research from `.planning/research/ARCHITECTURE.md` and `.planning/research/STACK.md` has already established the architecture patterns and technology choices.

The QuestDB reader should use the PostgreSQL wire protocol (port 8812) with either psycopg2 (already in dependencies) or psycopg3 for better performance. VWAP and TWAP calculations should leverage QuestDB's `SAMPLE BY` for time-based aggregations where possible, falling back to pandas for complex cases. Arrival price is simply the mid-price from an orderbook snapshot at order submission time.

**Primary recommendation:** Build a thin `QuestDBReader` class that mirrors the existing `QuestDBWriter` pattern, returning pandas DataFrames. Implement VWAP/TWAP as both SQL-side (using `SAMPLE BY`) and Python-side calculations to support different use cases.

## Standard Stack

The established libraries/tools for this domain:

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `psycopg2-binary` | >=2.9 | PostgreSQL wire protocol queries | Already in pyproject.toml |
| `pandas` | >=2.0 | DataFrame operations, time series | Already in notebooks optional deps |
| `numpy` | >=1.26 | Numerical calculations | Already in notebooks optional deps |

### New Dependencies
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `psycopg` (v3) | >=3.1 | Faster queries with binary protocol | Optional upgrade for performance |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| psycopg2 | psycopg3/asyncpg | psycopg3 is 2-3x faster, asyncpg best for large results; but psycopg2 already works |
| pandas VWAP | QuestDB SAMPLE BY | SAMPLE BY more efficient for simple aggregations; pandas needed for complex rolling calculations |
| SQLAlchemy ORM | Raw psycopg2 | ORM adds overhead; raw queries more efficient for analytics |

**Installation:**
```bash
# Already available via existing deps
pip install -e ".[notebooks]"
```

## Architecture Patterns

### Recommended Module Structure
```
src/tributary/analytics/
|-- __init__.py
|-- reader.py              # QuestDBReader - peer to storage/questdb.py
|-- benchmarks.py          # VWAP, TWAP, arrival price calculations
|-- models.py              # Analytics-specific result models (optional)
```

**Note:** The existing architecture research (`.planning/research/ARCHITECTURE.md`) recommends a deeper structure with `cost/`, `optimization/`, `simulation/` submodules. For Phase 1, keep it flat with just `reader.py` and `benchmarks.py`. The deeper structure can be added in Phase 2 when cost analytics expands.

### Pattern 1: Reader as Writer's Peer

**What:** QuestDBReader mirrors QuestDBWriter's connection management pattern
**When to use:** All QuestDB read operations for analytics
**Example:**
```python
# Source: Existing project pattern from src/tributary/storage/questdb.py
# Adapted for read operations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import psycopg2

@dataclass
class QuestDBReader:
    """Read interface for analytics queries against QuestDB."""

    host: str = "localhost"
    port: int = 8812  # PostgreSQL wire protocol port
    user: str = "admin"
    password: str = "quest"
    database: str = "qdb"

    _conn: Optional[psycopg2.connection] = None

    def connect(self) -> None:
        """Establish connection to QuestDB."""
        self._conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
        )
        self._conn.autocommit = True  # QuestDB requires autocommit

    def close(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def is_connected(self) -> bool:
        """Check if reader is connected."""
        return self._conn is not None and not self._conn.closed
```

### Pattern 2: DataFrame Return Type

**What:** All query methods return pandas DataFrames
**When to use:** Analytics queries that will be processed further
**Example:**
```python
# Source: QuestDB pandas integration pattern
# https://questdb.com/docs/query/pgwire/python/

def query_trades(
    self,
    market_id: str,
    start_time: datetime,
    end_time: datetime,
    token_id: Optional[str] = None,
) -> pd.DataFrame:
    """Query trades for a market and time range."""
    query = """
        SELECT timestamp, market_id, token_id, price, size, side, value
        FROM trades
        WHERE market_id = %s
          AND timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp
    """
    params = [market_id, start_time, end_time]

    if token_id:
        query = query.replace(
            "AND timestamp < %s",
            "AND timestamp < %s AND token_id = %s"
        )
        params.append(token_id)

    with self._conn.cursor() as cur:
        cur.execute(query, params)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    return pd.DataFrame(rows, columns=columns)
```

### Pattern 3: SAMPLE BY for Time Aggregations

**What:** Use QuestDB's SAMPLE BY for time-bucketed calculations
**When to use:** VWAP/TWAP over fixed intervals, aggregated metrics
**Example:**
```python
# Source: QuestDB SAMPLE BY documentation
# https://questdb.com/docs/reference/sql/sample-by/

def query_vwap_sampled(
    self,
    market_id: str,
    start_time: datetime,
    end_time: datetime,
    interval: str = "1h",  # e.g., "15m", "1h", "1d"
) -> pd.DataFrame:
    """
    Calculate VWAP using QuestDB SAMPLE BY.

    More efficient than fetching all trades and calculating in Python.
    """
    query = """
        SELECT
            timestamp,
            sum(price * size) / sum(size) as vwap,
            sum(size) as volume,
            count() as trade_count
        FROM trades
        WHERE market_id = %s
          AND timestamp >= %s
          AND timestamp < %s
        SAMPLE BY %s
        ALIGN TO CALENDAR
    """
    # Note: interval must be interpolated, not parameterized
    # QuestDB doesn't support bind variables for SAMPLE BY interval
    query = query.replace("SAMPLE BY %s", f"SAMPLE BY {interval}")

    with self._conn.cursor() as cur:
        cur.execute(query, [market_id, start_time, end_time])
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    return pd.DataFrame(rows, columns=columns)
```

### Anti-Patterns to Avoid

- **Fetching all data for simple aggregations:** Use SAMPLE BY instead of pulling raw trades to Python
- **Connection per query:** Reuse connections; QuestDB performs best with persistent connections
- **Using ORM for analytics:** Raw SQL is more efficient and clearer for complex queries
- **Ignoring autocommit:** QuestDB requires autocommit=True; transactions are not supported

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Time-bucketed aggregations | Custom groupby logic | QuestDB `SAMPLE BY` | Database-side is faster, handles timestamps correctly |
| VWAP calculation | Complex loop | `sum(price*size)/sum(size)` | One-liner in pandas or SQL |
| Cumulative VWAP | Manual iteration | `cumsum()` | Vectorized pandas operation |
| Timezone handling | Manual conversion | `ALIGN TO CALENDAR TIME ZONE` | QuestDB handles DST transitions |

**Key insight:** QuestDB's SAMPLE BY does most of the heavy lifting for time-series aggregations. Only fall back to Python-side pandas calculations when you need rolling windows or custom logic that SAMPLE BY doesn't support.

## Common Pitfalls

### Pitfall 1: SAMPLE BY Interval as Bind Variable
**What goes wrong:** `SAMPLE BY %s` fails - QuestDB doesn't support parameterized intervals
**Why it happens:** Developers expect all query parts to be parameterizable
**How to avoid:** String-interpolate the interval (e.g., `f"SAMPLE BY {interval}"`) after validating it
**Warning signs:** Query fails with syntax error on SAMPLE BY clause

### Pitfall 2: Missing autocommit
**What goes wrong:** Queries hang or fail silently
**Why it happens:** QuestDB doesn't support transactions; autocommit must be enabled
**How to avoid:** Set `conn.autocommit = True` immediately after connecting
**Warning signs:** First query works, subsequent queries timeout

### Pitfall 3: Cumulative vs Window VWAP
**What goes wrong:** Wrong VWAP calculation - getting per-interval instead of cumulative
**Why it happens:** Confusion between "VWAP for this hour" vs "VWAP from market open to now"
**How to avoid:**
- **Per-interval:** Use SAMPLE BY with `sum(price*size)/sum(size)`
- **Cumulative:** Fetch trades, use pandas `cumsum()` pattern
**Warning signs:** VWAP resets each interval instead of being cumulative

### Pitfall 4: Empty Results from Time Range Queries
**What goes wrong:** Query returns no rows even when data exists
**Why it happens:** Timestamp format mismatch or timezone issues
**How to avoid:**
- Use Python datetime objects, not strings
- Ensure timestamps are UTC (QuestDB stores in UTC)
- Check that designated timestamp column matches filter column
**Warning signs:** Same query works in QuestDB console but not from Python

### Pitfall 5: JSON Columns (bid_prices, ask_prices)
**What goes wrong:** orderbook_snapshots queries return JSON strings instead of lists
**Why it happens:** The writer stores lists as JSON strings via `json.dumps()`
**How to avoid:** Parse JSON after fetching: `df['bid_prices'] = df['bid_prices'].apply(json.loads)`
**Warning signs:** Getting string `"[0.64, 0.63, 0.62]"` instead of list `[0.64, 0.63, 0.62]`

## Code Examples

Verified patterns from official sources and existing codebase:

### VWAP Calculation (Python-side)
```python
# Source: Standard VWAP formula
# https://databento.com/blog/vwap-python

def calculate_vwap(trades_df: pd.DataFrame) -> float:
    """
    Calculate VWAP for a set of trades.

    Args:
        trades_df: DataFrame with 'price' and 'size' columns

    Returns:
        Volume-weighted average price
    """
    if trades_df.empty or trades_df['size'].sum() == 0:
        return float('nan')

    return (trades_df['price'] * trades_df['size']).sum() / trades_df['size'].sum()


def calculate_cumulative_vwap(trades_df: pd.DataFrame) -> pd.Series:
    """
    Calculate cumulative VWAP over time.

    Args:
        trades_df: DataFrame with 'price', 'size', 'timestamp' columns

    Returns:
        Series with cumulative VWAP at each trade
    """
    df = trades_df.sort_values('timestamp')
    cumulative_value = (df['price'] * df['size']).cumsum()
    cumulative_volume = df['size'].cumsum()

    return cumulative_value / cumulative_volume
```

### TWAP Calculation
```python
# Source: Standard TWAP formula

def calculate_twap(trades_df: pd.DataFrame, interval: str = "1min") -> float:
    """
    Calculate Time-Weighted Average Price.

    TWAP = average of prices sampled at regular intervals.

    Args:
        trades_df: DataFrame with 'price', 'timestamp' columns
        interval: Resampling interval (e.g., "1min", "5min", "1h")

    Returns:
        Time-weighted average price
    """
    if trades_df.empty:
        return float('nan')

    # Resample to regular intervals, take last price in each interval
    df = trades_df.set_index('timestamp')
    resampled = df['price'].resample(interval).last().dropna()

    return resampled.mean()


def calculate_twap_from_orderbooks(
    orderbooks_df: pd.DataFrame,
    interval: str = "1min"
) -> float:
    """
    Calculate TWAP from orderbook mid-prices.

    More accurate than trade-based TWAP for illiquid markets.

    Args:
        orderbooks_df: DataFrame with 'mid_price', 'timestamp' columns
        interval: Resampling interval

    Returns:
        Time-weighted average mid-price
    """
    if orderbooks_df.empty:
        return float('nan')

    df = orderbooks_df.set_index('timestamp')
    resampled = df['mid_price'].resample(interval).last().dropna()

    return resampled.mean()
```

### Arrival Price Lookup
```python
# Source: TCA benchmark definition
# https://www.talos.com/insights/execution-insights-through-transaction-cost-analysis-tca-benchmarks-and-slippage

def get_arrival_price(
    reader: "QuestDBReader",
    market_id: str,
    token_id: str,
    order_time: datetime,
    lookback_seconds: int = 5,
) -> Optional[float]:
    """
    Get arrival price (mid-price at order submission time).

    Arrival price is the benchmark for measuring implementation shortfall.
    Uses the closest orderbook snapshot before or at order_time.

    Args:
        reader: QuestDB reader instance
        market_id: Market identifier
        token_id: Token/outcome identifier
        order_time: Order submission timestamp
        lookback_seconds: How far back to search for snapshot

    Returns:
        Mid-price at arrival, or None if no snapshot found
    """
    start_time = order_time - timedelta(seconds=lookback_seconds)

    # Get the last orderbook snapshot at or before order_time
    query = """
        SELECT mid_price, timestamp
        FROM orderbook_snapshots
        WHERE market_id = %s
          AND token_id = %s
          AND timestamp >= %s
          AND timestamp <= %s
        ORDER BY timestamp DESC
        LIMIT 1
    """

    df = reader.execute_query(query, [market_id, token_id, start_time, order_time])

    if df.empty:
        return None

    return float(df['mid_price'].iloc[0])
```

### QuestDB SAMPLE BY for VWAP
```python
# Source: QuestDB documentation
# https://questdb.com/docs/reference/sql/sample-by/

def query_vwap_hourly(
    reader: "QuestDBReader",
    market_id: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """
    Query hourly VWAP using QuestDB SAMPLE BY.

    Returns DataFrame with columns: timestamp, vwap, volume, trade_count
    """
    query = """
        SELECT
            timestamp,
            sum(price * size) / sum(size) as vwap,
            sum(size) as volume,
            count() as trade_count
        FROM trades
        WHERE market_id = %s
          AND timestamp >= %s
          AND timestamp < %s
        SAMPLE BY 1h
        ALIGN TO CALENDAR
    """

    return reader.execute_query(query, [market_id, start_time, end_time])
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ALIGN TO FIRST OBSERVATION | ALIGN TO CALENDAR (default) | QuestDB 7.4.0 | Time buckets align to calendar hours/days by default |
| psycopg2 only | psycopg3 + asyncpg options | 2024+ | 2-3x faster queries with binary protocol |
| Manual VWAP loops | Vectorized pandas + SAMPLE BY | - | 10-100x faster for large datasets |

**Deprecated/outdated:**
- QuestDB before 7.4: Required explicit `ALIGN TO CALENDAR`; now default
- questdb-connect package: Convenience wrapper; raw psycopg2/psycopg3 is simpler

## Open Questions

Things that couldn't be fully resolved:

1. **psycopg2 vs psycopg3 for this project**
   - What we know: psycopg3 is faster (binary protocol), psycopg2 already works
   - What's unclear: Whether performance matters enough to add new dependency
   - Recommendation: Start with psycopg2 (already available), add psycopg3 if needed

2. **Handling missing intervals in SAMPLE BY**
   - What we know: FILL(NULL) or FILL(PREV) can fill gaps
   - What's unclear: Which fill strategy is best for VWAP/TWAP calculations
   - Recommendation: Use FILL(NULL), handle NaN in Python for analytics clarity

3. **Token-level vs Market-level queries**
   - What we know: trades/orderbooks have both market_id and token_id
   - What's unclear: When to aggregate across tokens vs query per-token
   - Recommendation: Default to token_id queries; market_id aggregation as option

## Sources

### Primary (HIGH confidence)
- [QuestDB SAMPLE BY Documentation](https://questdb.com/docs/reference/sql/sample-by/) - Time-based aggregation syntax
- [QuestDB Python PGWire Guide](https://questdb.com/docs/query/pgwire/python/) - Connection patterns and examples
- [QuestDB Aggregation Functions](https://questdb.com/docs/query/functions/aggregation/) - Available aggregate functions
- Existing codebase: `src/tributary/storage/questdb.py` - Writer patterns to mirror
- Existing codebase: `src/tributary/core/models.py` - Trade and OrderBookSnapshot models

### Secondary (MEDIUM confidence)
- [Databento VWAP Python Guide](https://databento.com/blog/vwap-python) - VWAP calculation patterns
- [Talos TCA Benchmarks](https://www.talos.com/insights/execution-insights-through-transaction-cost-analysis-tca-benchmarks-and-slippage) - Arrival price and benchmark definitions
- [Wikipedia Transaction Cost Analysis](https://en.wikipedia.org/wiki/Transaction_cost_analysis) - Implementation shortfall definition
- `.planning/research/ARCHITECTURE.md` - Project-specific architecture patterns
- `.planning/research/STACK.md` - Technology stack recommendations

### Tertiary (LOW confidence)
- WebSearch results on VWAP formulas - Verified against primary sources
- Medium articles on TCA - Cross-referenced with official documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses existing dependencies, verified QuestDB docs
- Architecture: HIGH - Mirrors existing writer pattern, follows project conventions
- VWAP/TWAP formulas: HIGH - Standard financial calculations, verified
- Arrival price: HIGH - Well-defined TCA benchmark
- QuestDB query patterns: HIGH - Verified against official documentation
- Pitfalls: MEDIUM - Some based on general QuestDB knowledge

**Research date:** 2026-01-19
**Valid until:** 2026-02-19 (30 days - stable domain)
