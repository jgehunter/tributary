# Roadmap: Tributary Execution Intelligence

## Overview

Phase 2 (Execution Intelligence) transforms Tributary from a data collection framework into an execution analytics platform. The journey moves from data access (querying QuestDB) through cost measurement (benchmarks, slippage, impact) to optimization (Almgren-Chriss, strategies) and finally validation (backtesting that proves better execution). Each phase builds on the previous, with simulation being the payoff that demonstrates the core value: reduced execution costs vs naive approaches.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - QuestDB reader and core benchmarks (VWAP, TWAP, arrival price)
- [x] **Phase 2: Cost Analytics** - Slippage measurement and market impact estimation
- [x] **Phase 3: Optimization** - Almgren-Chriss framework and execution strategies
- [ ] **Phase 4: Simulation** - Backtesting engine that proves better execution

## Phase Details

### Phase 1: Foundation
**Goal**: Analytics can efficiently query historical data and calculate core benchmarks
**Depends on**: Nothing (first phase); builds on existing QuestDB storage from Phase 1 data collection
**Requirements**: DATA-01, DATA-02, DATA-03, COST-01, COST-02, COST-03
**Success Criteria** (what must be TRUE):
  1. User can query orderbook snapshots for any market and time range
  2. User can query trade history for any market and time range
  3. User can calculate VWAP for any time window and market
  4. User can calculate TWAP for any time window and market
  5. User can calculate arrival price (mid-price at order submission time)
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md - QuestDBReader with data queries (DATA-01, DATA-02, DATA-03)
- [x] 01-02-PLAN.md - Benchmark calculations: VWAP, TWAP, arrival price (COST-01, COST-02, COST-03)

### Phase 2: Cost Analytics
**Goal**: Analytics can measure execution costs and estimate market impact
**Depends on**: Phase 1 (requires data reader and benchmarks)
**Requirements**: COST-04, COST-05, COST-06, COST-07, COST-08
**Success Criteria** (what must be TRUE):
  1. User can calculate slippage in basis points for any executed order
  2. User can decompose implementation shortfall into timing, impact, and spread components
  3. User can estimate temporary and permanent market impact from historical data
  4. User can forecast execution cost for a given order size ("what would $X cost?")
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md - Slippage calculation and implementation shortfall decomposition (COST-04, COST-05)
- [x] 02-02-PLAN.md - Orderbook-based cost forecasting / walk-the-book (COST-08)
- [x] 02-03-PLAN.md - Market impact estimation and parameter calibration (COST-06, COST-07)

### Phase 3: Optimization
**Goal**: Generate optimal execution trajectories and compare strategies
**Depends on**: Phase 2 (requires impact estimation for Almgren-Chriss calibration)
**Requirements**: OPT-01, OPT-02, OPT-03, OPT-04, OPT-05, OPT-06
**Success Criteria** (what must be TRUE):
  1. User can calibrate Almgren-Chriss parameters from collected historical data
  2. User can generate optimal execution trajectories for a given order and risk aversion
  3. User can run TWAP, VWAP, and market order baseline strategies
  4. User can compare strategy outputs (timing, sizing) before simulation
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md - Almgren-Chriss parameter calibration and trajectory generation (OPT-01, OPT-02)
- [x] 03-02-PLAN.md - Baseline strategies: TWAP, VWAP, market order (OPT-03, OPT-04, OPT-05)
- [x] 03-03-PLAN.md - Trade scheduler optimizer and strategy comparison (OPT-06)

### Phase 4: Simulation
**Goal**: Prove better execution vs naive approaches through backtesting
**Depends on**: Phase 3 (requires strategies to simulate)
**Requirements**: SIM-01, SIM-02, SIM-03, SIM-04, SIM-05
**Success Criteria** (what must be TRUE):
  1. User can run event-driven execution simulations on historical data
  2. Simulations use realistic fill models based on actual orderbook depth
  3. User can compare multiple strategies on the same historical period
  4. User can see clear metrics (cost, risk, shortfall) for each strategy
  5. Backtests demonstrate better execution with optimized strategies vs naive approaches
**Plans**: 3 plans

Plans:
- [ ] 04-01-PLAN.md - Event types and fill model with liquidity consumption (SIM-01, SIM-02)
- [ ] 04-02-PLAN.md - Simulation engine and multi-strategy runner (SIM-01, SIM-03)
- [ ] 04-03-PLAN.md - Metrics calculation and strategy comparison (SIM-04, SIM-05)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete | 2026-01-19 |
| 2. Cost Analytics | 3/3 | Complete | 2026-01-19 |
| 3. Optimization | 3/3 | Complete | 2026-01-19 |
| 4. Simulation | 0/3 | Planned | - |

---
*Roadmap created: 2026-01-19*
*Phase 1 planned: 2026-01-19*
*Phase 1 complete: 2026-01-19*
*Phase 2 planned: 2026-01-19*
*Phase 2 complete: 2026-01-19*
*Phase 3 planned: 2026-01-19*
*Phase 3 complete: 2026-01-19*
*Phase 4 planned: 2026-01-19*
