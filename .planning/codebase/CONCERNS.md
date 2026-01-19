# Codebase Concerns

**Analysis Date:** 2026-01-19

## Tech Debt

**Incomplete Status Command:**
- Issue: CLI `status` command is a stub with "Status check not implemented yet."
- Files: `src/tributary/cli/commands.py` (line 148)
- Impact: Users cannot check collector health or metrics without external tooling
- Fix approach: Implement status command that queries QuestDB for recent metrics or connects to running scheduler

**Health Check is a No-Op:**
- Issue: `_health_check` method only logs "Health check: OK" with no actual checks
- Files: `src/tributary/scheduler/scheduler.py` (lines 323-326)
- Impact: No alerting when data collection stalls, QuestDB becomes unreachable, or collectors fail
- Fix approach: Implement actual health checks for DB connectivity, data freshness, and collector state

**Validators Not Integrated:**
- Issue: TradeValidator and OrderBookValidator exist but are not used in data collection pipeline
- Files: `src/tributary/validation/validators.py`, `src/tributary/scheduler/scheduler.py`
- Impact: Invalid data (bad prices, crossed books) written to QuestDB without filtering
- Fix approach: Integrate validators in `_collect_trades` and `_collect_orderbooks` before writing to storage

**Gap Detection Not Integrated:**
- Issue: GapDetector class exists but is never instantiated or used in scheduler
- Files: `src/tributary/monitoring/alerting.py`, `src/tributary/scheduler/scheduler.py`
- Impact: Data gaps go undetected and unreported
- Fix approach: Initialize GapDetector in scheduler and call `record_data` after each collection cycle

**CollectionMetrics Class Unused:**
- Issue: Detailed metrics tracking class exists but is not integrated with scheduler
- Files: `src/tributary/monitoring/metrics.py`, `src/tributary/scheduler/scheduler.py`
- Impact: No in-memory metrics aggregation; only raw metrics written to QuestDB
- Fix approach: Instantiate CollectionMetrics in scheduler and record collection results

**Backfill Uses Token ID Instead of Condition ID:**
- Issue: Backfill script calls `collector.backfill(token_id)` but `fetch_historical_trades` expects `condition_id`
- Files: `scripts/backfill.py` (line 74), `src/tributary/collectors/polymarket/collector.py` (line 302)
- Impact: Historical backfill may fetch incorrect data or fail silently
- Fix approach: Pass condition_id (market_id) to backfill and historical trades methods consistently

**Abstract Methods with Just `pass`:**
- Issue: Base collector abstract methods use `pass` instead of `raise NotImplementedError`
- Files: `src/tributary/core/base_collector.py` (lines 42, 48, 53, 58, 63, 73, 78, 92)
- Impact: Missing implementations won't raise errors at runtime, causing silent failures
- Fix approach: Replace `pass` with `raise NotImplementedError("Subclass must implement")`

## Known Bugs

**Timezone Mixing:**
- Symptoms: `datetime.utcnow()` used throughout but some API timestamps are timezone-aware
- Files: `src/tributary/validation/validators.py` (line 58), `src/tributary/core/base_collector.py` (line 96, 125), `src/tributary/monitoring/metrics.py` (line 26)
- Trigger: Comparing naive datetime with timezone-aware datetime from API
- Workaround: Currently works because transformers normalize to timezone-aware UTC

**Duplicate Trade ID Tracking Memory Leak:**
- Symptoms: Set of seen trade IDs can grow unbounded between pruning
- Files: `src/tributary/validation/validators.py` (lines 39-40, 72-76)
- Trigger: High-frequency collection with unique trade IDs
- Workaround: Current code prunes to half when exceeding 100k entries, but pruning is arbitrary (not oldest)

**Rate Limiter Uses datetime.utcnow() Under Lock:**
- Symptoms: Potential timing inconsistency when lock acquisition delayed
- Files: `src/tributary/collectors/polymarket/rate_limiter.py` (line 26, 47)
- Trigger: High concurrency or slow acquisition
- Workaround: None; impact is minor (slight rate limit overshoot)

## Security Considerations

**Private Key in Configuration:**
- Risk: Private key stored in YAML config files or environment variables
- Files: `src/tributary/core/config.py` (line 60), config files
- Current mitigation: Uses environment variable substitution with `${VAR:default}` pattern
- Recommendations: Ensure `.env` is in `.gitignore`; consider secrets manager for production

**No Input Validation on Market Slugs:**
- Risk: Malformed slugs passed to API endpoints without sanitization
- Files: `src/tributary/collectors/polymarket/collector.py` (line 168, 196)
- Current mitigation: None; relies on API to reject invalid input
- Recommendations: Validate slug format before making API requests

**PostgreSQL Default Credentials:**
- Risk: Default QuestDB credentials (admin/quest) used in config defaults
- Files: `src/tributary/core/config.py` (lines 26-27)
- Current mitigation: Can be overridden via config or environment
- Recommendations: Require explicit credential configuration in production mode

## Performance Bottlenecks

**Sequential Orderbook Collection:**
- Problem: Orderbooks fetched one-by-one for each token_id
- Files: `src/tributary/scheduler/scheduler.py` (lines 250-261)
- Cause: Single `for` loop with `await` per request
- Improvement path: Use `asyncio.gather` with rate-limited concurrency (e.g., semaphore with 5-10 concurrent requests)

**Sequential Trade Collection Per Market:**
- Problem: Trades collected sequentially for each market
- Files: `src/tributary/scheduler/scheduler.py` (lines 288-305)
- Cause: Single `for` loop with `await` per market
- Improvement path: Parallelize with `asyncio.gather` respecting rate limits

**Rate Limiter Lock Contention:**
- Problem: Single asyncio lock per endpoint, blocking all concurrent requests
- Files: `src/tributary/collectors/polymarket/rate_limiter.py` (line 21)
- Cause: Lock held during entire rate check and potential sleep
- Improvement path: Use asyncio.Semaphore or atomic counter with retry logic

**Full Orderbook Depth Serialized as JSON:**
- Problem: Entire bid/ask arrays serialized as JSON strings in QuestDB
- Files: `src/tributary/storage/questdb.py` (lines 175-178)
- Cause: QuestDB ILP protocol doesn't support native arrays
- Improvement path: Consider storing only top N levels, or separate depth tables

## Fragile Areas

**Polymarket API Response Parsing:**
- Files: `src/tributary/collectors/polymarket/transformers.py`
- Why fragile: Multiple try/except blocks silently swallow parse errors and return defaults
- Safe modification: Add logging inside except blocks before returning defaults
- Test coverage: No tests for transformer functions

**Market Slug Resolution Fallback:**
- Files: `src/tributary/collectors/polymarket/collector.py` (lines 150-163)
- Why fragile: Multiple fallback attempts (exact slug, slug_contains) without clear failure modes
- Safe modification: Test each fallback path individually
- Test coverage: No integration tests for market lookup

**Signal Handler in CLI:**
- Files: `src/tributary/cli/commands.py` (lines 68-73)
- Why fragile: Creates task in potentially different event loop context
- Safe modification: Use proper asyncio signal handling or KeyboardInterrupt pattern
- Test coverage: No tests for graceful shutdown

**Configuration Deep Merge:**
- Files: `src/tributary/core/config.py` (lines 129-137)
- Why fragile: Recursive merge can produce unexpected results with mixed types
- Safe modification: Add type checking before merge
- Test coverage: No tests for config merging

## Scaling Limits

**In-Memory Market Data Tracking:**
- Current capacity: All monitored markets stored in `_market_data` dict
- Limit: Scales with number of markets; hundreds should be fine
- Scaling path: Currently acceptable; consider LRU cache if tracking thousands

**QuestDB Write Buffering:**
- Current capacity: Auto-flush at 1000 rows or 1 second
- Limit: High-volume collection may cause write latency
- Scaling path: Tune `auto_flush_rows` based on throughput; consider separate writer instances

**Single Collector Instance:**
- Current capacity: One Polymarket collector per scheduler
- Limit: Cannot distribute load across processes/machines
- Scaling path: Implement distributed task queue (Redis, RabbitMQ) for horizontal scaling

## Dependencies at Risk

**py-clob-client:**
- Risk: Unofficial Polymarket SDK; could break with API changes
- Impact: Authentication and credential derivation would fail
- Migration plan: Implement direct signing using eth-account if SDK becomes unmaintained

**APScheduler 3.x:**
- Risk: Version 4.0 in development with breaking changes
- Impact: Scheduler initialization would need updates
- Migration plan: Pin to 3.x series; evaluate migration when 4.0 stable

**aiohttp Session Management:**
- Risk: Session created but not used in context manager pattern everywhere
- Impact: Potential resource leaks on exception paths
- Migration plan: Refactor to use `async with` pattern consistently

## Missing Critical Features

**No Data Deduplication at Storage Layer:**
- Problem: Same trade written multiple times if collected in overlapping windows
- Blocks: Accurate analytics; requires post-hoc deduplication queries

**No Retry Logic for Failed API Requests:**
- Problem: Single failure results in data gap
- Blocks: Reliable data collection in presence of transient API errors

**No Graceful Degradation:**
- Problem: If QuestDB unavailable at startup, collector continues without persistence
- Blocks: Data loss awareness; users may not notice missing data

**No Schema Migration for QuestDB:**
- Problem: Table schema changes require manual intervention
- Blocks: Safe upgrades when adding new fields or tables

## Test Coverage Gaps

**No Tests for Collector Classes:**
- What's not tested: PolymarketCollector methods (fetch_markets, fetch_trades, fetch_orderbook)
- Files: `src/tributary/collectors/polymarket/collector.py`
- Risk: API response parsing changes could break collection silently
- Priority: High

**No Tests for Scheduler:**
- What's not tested: TributaryScheduler lifecycle, market initialization, collection cycles
- Files: `src/tributary/scheduler/scheduler.py`
- Risk: Integration bugs between components undetected
- Priority: High

**No Tests for Storage Layer:**
- What's not tested: QuestDBWriter connection, write methods, flush behavior
- Files: `src/tributary/storage/questdb.py`
- Risk: Data corruption or loss undetected
- Priority: High

**No Tests for Transformers:**
- What's not tested: API response to model transformations
- Files: `src/tributary/collectors/polymarket/transformers.py`
- Risk: Parser regressions when API response format changes
- Priority: High

**No Tests for Config Loading:**
- What's not tested: YAML loading, env substitution, deep merge
- Files: `src/tributary/core/config.py`
- Risk: Config changes could break in unexpected ways
- Priority: Medium

**No Tests for Rate Limiter:**
- What's not tested: Window sliding, concurrent acquisition, timing behavior
- Files: `src/tributary/collectors/polymarket/rate_limiter.py`
- Risk: Rate limit violations leading to API bans
- Priority: Medium

**No Integration Tests:**
- What's not tested: End-to-end collection flow with real or mocked APIs
- Files: `tests/integration/` (empty `__init__.py` only)
- Risk: Component interactions fail in production
- Priority: High

---

*Concerns audit: 2026-01-19*
