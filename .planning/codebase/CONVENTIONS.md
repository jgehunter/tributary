# Coding Conventions

**Analysis Date:** 2026-01-19

## Naming Patterns

**Files:**
- Use `snake_case.py` for all Python modules
- Module names match their primary class (e.g., `collector.py` contains `PolymarketCollector`)
- Private helper functions prefix with underscore (e.g., `_build_config`, `_parse_json_field`)

**Functions:**
- Use `snake_case` for all functions and methods
- Async methods prefix with `async` keyword (e.g., `async def fetch_markets`)
- Private methods prefix with underscore (e.g., `_collect_orderbooks`, `_health_check`)
- Getter methods use property decorator without `get_` prefix

**Variables:**
- Use `snake_case` for all variables
- Private instance variables prefix with underscore (e.g., `self._running`, `self._session`)
- Constants use `UPPER_SNAKE_CASE` (e.g., `_COLLECTORS`, `_max_seen_ids`)

**Classes:**
- Use `PascalCase` for all classes
- Suffix base classes with `Base` or use abstract pattern (e.g., `MarketDataCollector`)
- Suffix configs with `Config` (e.g., `QuestDBConfig`, `PolymarketConfig`)
- Suffix errors with `Error` (e.g., `TributaryError`, `CollectionError`)

**Types (Enums):**
- Use `PascalCase` for enum classes (e.g., `AssetType`, `Exchange`, `Side`)
- Use `UPPER_SNAKE_CASE` for enum values (e.g., `PREDICTION_MARKET`, `POLYMARKET`)

## Code Style

**Formatting:**
- Tool: `black`
- Line length: 100 characters
- Target: Python 3.11
- Config in `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ["py311"]
```

**Linting:**
- Tool: `ruff`
- Line length: 100 characters
- Target: Python 3.11
- Config in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py311"
```

**Type Hints:**
- Use type hints for all function parameters and return values
- Use `Optional[T]` for nullable parameters
- Use `List`, `Dict`, `Tuple` from `typing` module
- Use `AsyncIterator` for async generators

## Import Organization

**Order:**
1. Standard library imports (alphabetically)
2. Third-party imports (alphabetically)
3. Local application imports (alphabetically)

**Example from `src/tributary/collectors/polymarket/collector.py`:**
```python
import logging
from datetime import datetime
from typing import List, Optional, AsyncIterator

import aiohttp

from tributary.core.base_collector import MarketDataCollector, CollectionResult
from tributary.core.models import Market, Trade, OrderBookSnapshot, AssetType, Exchange
from tributary.core.config import PolymarketConfig
from tributary.core.exceptions import CollectionError, AuthenticationError

from .auth import PolymarketAuth
from .rate_limiter import RateLimiterRegistry
from .transformers import (
    transform_market_response,
    transform_trade_response,
    transform_orderbook_response,
)
```

**Path Aliases:**
- No path aliases used - full relative imports from package root
- Use relative imports (`.`) within same package (e.g., `from .auth import PolymarketAuth`)
- Use absolute imports for cross-package imports (e.g., `from tributary.core.models import ...`)

## Error Handling

**Exception Hierarchy:**
- Base: `TributaryError` (all custom exceptions inherit from this)
- Domain-specific: `ConfigurationError`, `CollectionError`, `AuthenticationError`, `RateLimitError`, `StorageError`, `ValidationError`
- Defined in: `src/tributary/core/exceptions.py`

**Patterns:**
```python
# Raise specific exceptions with descriptive messages
raise CollectionError(f"Failed to fetch markets: {response.status} - {text}")

# Catch and log with context
except Exception as e:
    logger.error(f"Failed to fetch market {slug}: {e}")
    continue

# Catch specific exceptions when needed
except IngressError as e:
    logger.error(f"Failed to write trade {trade.trade_id}: {e}")

# Use warnings for non-fatal issues
logger.warning(f"Failed to authenticate: {e}. Using public endpoints only.")
```

**Error Recovery:**
- Continue processing after non-critical errors (e.g., single failed market fetch)
- Log errors with context (market slug, token ID, etc.)
- Track error counts in `CollectionResult` and metrics
- Use `ValidationResult` with separate `errors` and `warnings` lists

## Logging

**Framework:** Python standard `logging` module with `rich` handler for CLI

**Setup (from `src/tributary/cli/commands.py`):**
```python
import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
```

**Logger Creation:**
```python
logger = logging.getLogger(__name__)
```

**Log Levels:**
- `DEBUG`: Detailed operational info (batch counts, pagination details)
- `INFO`: Important state changes (connection established, markets initialized)
- `WARNING`: Non-fatal issues (failed single fetch, missing token IDs)
- `ERROR`: Failures that affect operation (connection failures, write errors)

**Patterns:**
```python
# Info for significant events
logger.info(f"Authenticated as {self._auth.address}")
logger.info(f"Successfully initialized {total} of {len(enabled_markets)} markets")

# Debug for operational details
logger.debug(f"Fetched page {page + 1}: {len(markets)} markets (total: {len(all_markets)})")
logger.debug(f"Orderbook collection: {total_snapshots} snapshots, {errors} errors, {duration:.0f}ms")

# Warning for recoverable issues
logger.warning(f"Failed to collect orderbook for {slug}/{token_id[:20]}...: {e}")

# Error with context and guidance
logger.error(f"Failed to connect to QuestDB: {e}")
logger.error("Start QuestDB with: docker compose -f docker/docker-compose.yml up -d")
```

## Comments

**Module Docstrings:**
- Every module has a one-line docstring describing its purpose
- Example: `"""Polymarket CLOB data collector."""`

**Class Docstrings:**
- Classes have docstrings explaining purpose and key functionality
- Use multi-line format for complex classes
```python
class PolymarketCollector(MarketDataCollector):
    """
    Polymarket CLOB data collector.

    Uses three APIs:
    - CLOB API (clob.polymarket.com): Orderbooks, real-time data
    - Gamma API (gamma-api.polymarket.com): Market metadata
    - Data API (data-api.polymarket.com): Historical trades
    """
```

**Method Docstrings:**
- Abstract methods have docstrings explaining contract
- Complex methods document Args and Returns
```python
async def fetch_trades(
    self,
    market_id: str,
    since: Optional[datetime] = None,
    limit: int = 1000,
) -> List[Trade]:
    """
    Fetch recent trades from Data API.

    Args:
        market_id: Market condition ID (Polymarket uses conditionId for trade queries)
        since: Only fetch trades after this timestamp
        limit: Maximum number of trades to fetch

    Returns:
        List of trades (each with market_id=conditionId and token_id=asset_id)
    """
```

**Inline Comments:**
- Use for non-obvious implementation details
- Explain API-specific quirks and data transformations
```python
# API returns as JSON-encoded string: "[\"id1\", \"id2\"]"
clob_token_ids = _parse_json_field(raw_token_ids, [])

# Unix timestamp in seconds or milliseconds
if ts_value > 1e12:
    timestamp = datetime.fromtimestamp(ts_value / 1000, tz=timezone.utc)
```

## Function Design

**Size:**
- Functions focus on single responsibility
- Complex operations split into private helper methods
- Example: `fetch_market_by_slug` delegates to `_fetch_market_by_slug_exact` and `_fetch_market_by_slug_contains`

**Parameters:**
- Use typed parameters with defaults where appropriate
- Use Optional for nullable parameters
- Use keyword-only args for clarity in complex functions
```python
def __init__(
    self,
    min_price: float = 0.001,
    max_price: float = 0.999,
    max_age_hours: int = 24,
):
```

**Return Values:**
- Use explicit return types
- Use dataclasses for complex return values (e.g., `CollectionResult`, `ValidationResult`)
- Use `Optional[T]` when None is valid return
- Return tuples for multiple values with type hints

## Module Design

**Exports:**
- Define `__all__` in `__init__.py` files to control public API
- Example from `src/tributary/collectors/__init__.py`:
```python
from .registry import get_collector, register_collector

__all__ = ["get_collector", "register_collector"]
```

**Barrel Files:**
- Package `__init__.py` files re-export key symbols
- Enables clean imports: `from tributary.collectors import get_collector`

**Lazy Loading:**
- Use lazy imports to avoid circular dependencies
- Example in registry:
```python
def get_collector(name: str, config: Any) -> MarketDataCollector:
    if name not in _COLLECTORS:
        # Lazy import to avoid circular imports
        if name == "polymarket":
            from .polymarket.collector import PolymarketCollector
            register_collector("polymarket", PolymarketCollector)
```

## Data Classes

**Configuration:**
- Use `@dataclass` for configuration classes
- Use `field(default_factory=...)` for mutable defaults
```python
@dataclass
class PolymarketConfig:
    enabled: bool = True
    clob_url: str = "https://clob.polymarket.com"
    rate_limits: Dict[str, int] = field(default_factory=dict)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
```

**Domain Models:**
- Use Pydantic `BaseModel` for data validation
- Use `@computed_field` for derived properties
```python
class Trade(BaseModel):
    market_id: str
    price: float
    size: float

    @computed_field
    @property
    def value(self) -> float:
        return self.price * self.size
```

## Async Patterns

**Session Management:**
- Use context managers for session lifecycle
- Store session in instance variable for reuse
```python
async def connect(self) -> None:
    self._session = aiohttp.ClientSession()

async def disconnect(self) -> None:
    if self._session:
        await self._session.close()
        self._session = None
```

**Rate Limiting:**
- Use async-aware rate limiters with locks
- Acquire before each API call
```python
await self._rate_limiters.acquire("gamma_markets")
async with self._session.get(url, params=params) as response:
    ...
```

**Async Iterators:**
- Use `AsyncIterator` for paginated/streaming data
- Yield batches for memory efficiency
```python
async def fetch_historical_trades(...) -> AsyncIterator[List[Trade]]:
    while batch_count < max_batches:
        ...
        yield trades
```

---

*Convention analysis: 2026-01-19*
