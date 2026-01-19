# Tributary

## What This Is

A multi-asset data collection and execution analytics framework. Collects orderbook snapshots and trade history from prediction markets (Polymarket) and crypto exchanges, then provides execution cost analysis and optimal execution strategies. Built by an eFX trader to learn and apply quantitative execution concepts.

## Core Value

Demonstrably reduce execution costs vs naive approaches (market orders, simple TWAP) — proven through backtesting against historical data.

## Requirements

### Validated

- ✓ Multi-exchange collector architecture (abstract base class + registry pattern) — existing
- ✓ Polymarket orderbook snapshot collection via CLOB API — existing
- ✓ Polymarket trade history collection via Data API — existing
- ✓ Historical trade backfill with cursor-based pagination — existing
- ✓ QuestDB storage via ILP protocol with auto-flush — existing
- ✓ Per-endpoint rate limiting (sliding window) — existing
- ✓ Data validation (trades, orderbooks) with duplicate detection — existing
- ✓ YAML configuration with environment variable substitution — existing
- ✓ CLI interface (start, backfill, validate-config, status) — existing

### Active

**Phase 2: Execution Intelligence**

Cost Analytics:
- [ ] VWAP/TWAP benchmark calculation
- [ ] Implementation shortfall measurement
- [ ] Market impact estimation from historical data
- [ ] Slippage analysis

Optimal Execution:
- [ ] Almgren-Chriss framework implementation
- [ ] Trade scheduling optimizer
- [ ] Adaptive execution strategies
- [ ] Strategy comparison framework
- [ ] Execution simulation engine for backtesting

### Out of Scope

- Real-time execution/order routing — analytics only, not trading
- FX market integration — learning exercise, not production FX system
- Sub-second latency optimization — not HFT

## Context

**Professional background:** User is an eFX trader building this to develop quantitative execution skills with carryover to day job.

**Data available:** Orderbook snapshots and trade history from Polymarket, stored in QuestDB. Designed to expand to crypto exchanges.

**Market characteristics:** Prediction markets and crypto share thin liquidity characteristics vs traditional markets. Almgren-Chriss parameters need calibration from collected data.

**Success measurement:** Backtesting framework that compares optimal strategies vs naive baselines on historical data, updated as new data collected.

## Constraints

- **Tech stack**: Python 3.11+, QuestDB, existing async architecture — maintain consistency with Phase 1
- **Data source**: Must work with orderbook snapshots + trade history (what's being collected)
- **Extensibility**: Design analytics to work across asset classes (not Polymarket-specific)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Cost analytics before optimization | Need to measure costs before optimizing them | — Pending |
| Almgren-Chriss as optimization framework | Well-established, adaptable to different markets | — Pending |
| Backtest-based validation | Historical data available, no live trading risk | — Pending |

---
*Last updated: 2026-01-19 after initialization*
