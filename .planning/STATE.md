# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-19)

**Core value:** Demonstrably reduce execution costs vs naive approaches
**Current focus:** Phase 4 - Simulation (backtesting engine)

## Current Position

Phase: 4 of 4 (Simulation)
Plan: 2 of 3 in current phase
Status: In progress
Last activity: 2026-01-19 - Completed 04-02-PLAN.md (engine and runner)

Progress: [#########-] 90%

## Performance Metrics

**Velocity:**
- Total plans completed: 10
- Average duration: 4.7m
- Total execution time: 47m

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 2/2 | 7m | 3.5m |
| 2. Cost Analytics | 3/3 | 13m | 4.3m |
| 3. Optimization | 3/3 | 18m | 6.0m |
| 4. Simulation | 2/3 | 9m | 4.5m |

**Recent Trend:**
- Last 5 plans: 03-01 (5m), 03-02 (8m), 03-03 (5m), 04-01 (4m), 04-02 (5m)
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 4-phase structure follows dependency chain: Reader -> Benchmarks -> Cost Analytics -> Impact -> Optimization -> Simulation
- [Roadmap]: Research suggests empirical-first approach - measure actual market behavior before building models
- [01-01]: Synchronous psycopg2 for reader (async not needed for analytics workloads)
- [01-01]: Regex validation for SAMPLE BY intervals (prevents SQL injection)
- [01-02]: Use pandas resample() for TWAP interval buckets (native time-series handling)
- [01-02]: Return float('nan') for empty/invalid inputs (consistent with numpy/pandas conventions)
- [02-01]: Positive slippage = cost (unfavorable) for both buy and sell sides
- [02-01]: Perold framework for shortfall decomposition (delay, trading, spread, opportunity)
- [02-02]: Walk-the-book as PRIMARY cost forecasting method (direct orderbook measurement, not model-based)
- [02-03]: Square-root model as SECONDARY validation (calibrated for equity markets, not prediction markets)
- [02-03]: Participation rate thresholds: HIGH (<1%), MEDIUM (1-10%), LOW (>10%)
- [02-03]: Permanent impact = 40% of temporary (conservative default for thin markets)
- [03-01]: Risk-neutral (lambda=0) produces TWAP automatically via A-C math
- [03-01]: eta_tilde constraint violation falls back to TWAP with warning
- [03-01]: A-C heuristics: 1% ADV = full spread (temp), 10% ADV = full spread (perm)
- [03-02]: TWAP randomization default 10% for avoiding detection
- [03-02]: VWAP falls back to TWAP when volume profile sums to zero
- [03-02]: risk_aversion=inf for market order (infinite timing risk aversion)
- [03-03]: Participation rate constraint determines minimum intervals for uniform execution
- [03-03]: Risk-averse A-C trajectories may slightly exceed participation limit due to front-loading
- [03-03]: At least 2 strategies required for comparison
- [03-03]: Long-format DataFrame for chart data (strategy x period rows)
- [04-01]: Tuples for orderbook levels in frozen dataclasses (immutability)
- [04-01]: Exponential recovery model with configurable half-life for liquidity
- [04-01]: Track consumed liquidity per level index, not price
- [04-02]: Orders execute against most recent market state at or before order time (no lookahead)
- [04-02]: Fresh FillModel per strategy for isolated execution in StrategyRunner
- [04-02]: StrategyRun dataclass provides convenience aggregation properties

### Pending Todos

None yet.

### Blockers/Concerns

- Pre-existing test failure in test_models.py::TestOrderBookSnapshot::test_spread_computed - test expects 2000 bps but spread_bps formula returns 1000 for 10% spread. Should be fixed.

## Session Continuity

Last session: 2026-01-19 19:05 UTC
Stopped at: Completed 04-02-PLAN.md (engine and runner)
Resume file: None
