# Architecture

**Analysis Date:** 2026-01-19

## Pattern Overview

**Overall:** Layered Service Architecture with Plugin-Based Collectors

**Key Characteristics:**
- Abstract base class pattern for collectors (strategy pattern for multi-exchange support)
- Registry/factory pattern for collector instantiation
- Async-first design using Python asyncio throughout
- Configuration-driven operation with YAML configs and environment variable substitution
- Time-series optimized storage layer using QuestDB ILP protocol

## Layers

**CLI Layer:**
- Purpose: User interface for running the system
- Location: `src/tributary/cli/`
- Contains: Click-based command handlers, logging setup
- Depends on: Core (config), Scheduler
- Used by: End users via `tributary` command

**Scheduler Layer:**
- Purpose: Orchestrates periodic data collection tasks
- Location: `src/tributary/scheduler/`
- Contains: APScheduler-based job management, market initialization, collection coordination
- Depends on: Core (config, models), Collectors, Storage
- Used by: CLI layer

**Collectors Layer:**
- Purpose: Fetches market data from external exchanges
- Location: `src/tributary/collectors/`
- Contains: Abstract collector interface, exchange-specific implementations
- Depends on: Core (models, config, exceptions)
- Used by: Scheduler layer

**Storage Layer:**
- Purpose: Persists collected data to QuestDB
- Location: `src/tributary/storage/`
- Contains: QuestDB writer using ILP HTTP protocol
- Depends on: Core (models, config, exceptions)
- Used by: Scheduler layer, CLI backfill command

**Validation Layer:**
- Purpose: Validates incoming market data for quality/consistency
- Location: `src/tributary/validation/`
- Contains: Trade validator, OrderBook validator, stateful duplicate detection
- Depends on: Core (models)
- Used by: Collectors (optional pre-storage validation)

**Monitoring Layer:**
- Purpose: Tracks collection metrics and detects data gaps
- Location: `src/tributary/monitoring/`
- Contains: Metrics aggregation, gap detection with alerting
- Depends on: Core (models)
- Used by: Scheduler (future integration)

**Core Layer:**
- Purpose: Shared domain models, configuration, and exceptions
- Location: `src/tributary/core/`
- Contains: Pydantic models (Trade, OrderBookSnapshot, Market), dataclass configs, custom exceptions
- Depends on: External libraries (pydantic, yaml)
- Used by: All other layers

## Data Flow

**Live Collection Flow:**

1. CLI `start` command loads configuration from `config/` directory
2. `TributaryScheduler` initializes collectors via registry (`get_collector`)
3. Scheduler resolves market slugs to condition IDs and token IDs via Gamma API
4. APScheduler triggers periodic jobs:
   - Orderbook collection: Every 10s, fetches per token_id from CLOB API
   - Trade collection: Every 5s, fetches per condition_id from Data API
5. `QuestDBWriter` persists data via ILP HTTP protocol with auto-flush

**Backfill Flow:**

1. CLI `backfill` command or `scripts/backfill.py` invoked with market slug
2. Collector resolves market metadata (condition_id, token_ids)
3. Collector iterates through historical trades using cursor-based pagination
4. Callback function writes batches to QuestDB as they arrive
5. Writer flushes data in batches for throughput

**State Management:**
- Market tracking: `TributaryScheduler._market_data` dict maps slug to {condition_id, token_ids}
- Last trade timestamps: `MarketDataCollector._last_trade_timestamps` for incremental fetching
- Rate limiting: Per-endpoint sliding window rate limiters track request counts in deques
- Duplicate detection: `TradeValidator._seen_trade_ids` set (bounded to 100k entries)

## Key Abstractions

**MarketDataCollector:**
- Purpose: Abstract interface for fetching market data from any exchange
- Examples: `src/tributary/core/base_collector.py`, `src/tributary/collectors/polymarket/collector.py`
- Pattern: Template Method - base class provides `collect_all_trades()`, `collect_all_orderbooks()`, `backfill()` orchestration; subclasses implement `fetch_trades()`, `fetch_orderbook()`, `fetch_historical_trades()`

**Domain Models (Pydantic):**
- Purpose: Type-safe, validated data representations with computed fields
- Examples: `src/tributary/core/models.py` - `Trade`, `OrderBookSnapshot`, `Market`
- Pattern: Value Objects with computed properties (e.g., `Trade.value`, `OrderBookSnapshot.spread_bps`)

**Configuration Dataclasses:**
- Purpose: Strongly-typed hierarchical configuration
- Examples: `src/tributary/core/config.py` - `AppConfig`, `PolymarketConfig`, `QuestDBConfig`
- Pattern: Composition with nested dataclasses, built from YAML with env var substitution

**CollectionResult:**
- Purpose: Standardized return type for collection operations
- Examples: `src/tributary/core/base_collector.py`
- Pattern: Result object containing success flag, collected data, errors, and timing metrics

## Entry Points

**CLI Entry Point:**
- Location: `src/tributary/cli/commands.py:main()`
- Triggers: `tributary` CLI command (defined in pyproject.toml `[project.scripts]`)
- Responsibilities: Parse args, load config, dispatch to subcommands (start, backfill, validate-config, status)

**Module Entry Point:**
- Location: `src/tributary/__main__.py`
- Triggers: `python -m tributary`
- Responsibilities: Delegates to CLI main()

**Backfill Script:**
- Location: `scripts/backfill.py`
- Triggers: Direct Python execution
- Responsibilities: Standalone historical data backfill with progress logging

## Error Handling

**Strategy:** Hierarchical custom exceptions with graceful degradation

**Patterns:**
- All custom exceptions inherit from `TributaryError` (`src/tributary/core/exceptions.py`)
- Specific exceptions: `ConfigurationError`, `CollectionError`, `AuthenticationError`, `RateLimitError`, `StorageError`, `ValidationError`
- Storage failures logged but don't crash collector (collector continues without persistence)
- Collection errors per-market tracked in `CollectionResult.errors`, don't halt other markets
- Rate limiters block async tasks rather than raise exceptions

## Cross-Cutting Concerns

**Logging:**
- Python `logging` module with `rich.logging.RichHandler` for CLI
- Per-module loggers via `logging.getLogger(__name__)`
- Log level configurable via config and CLI flags

**Validation:**
- Pydantic models provide schema validation on construction
- `TradeValidator` and `OrderBookValidator` provide business rule validation
- Validation results contain both errors (invalid) and warnings (valid but suspicious)

**Authentication:**
- Polymarket L2 auth via `py-clob-client` library
- Private key from config/env, derives API credentials on connect
- Falls back to public endpoints if auth fails

**Rate Limiting:**
- `SlidingWindowRateLimiter` with configurable requests/window
- `RateLimiterRegistry` manages named limiters per endpoint type
- Async `acquire()` blocks until slot available

---

*Architecture analysis: 2026-01-19*
