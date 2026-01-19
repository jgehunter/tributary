# Requirements: Tributary

**Defined:** 2026-01-19
**Core Value:** Demonstrably reduce execution costs vs naive approaches

## v1 Requirements

Requirements for Phase 2: Execution Intelligence. Each maps to roadmap phases.

### Data Access

- [x] **DATA-01**: Analytics can query orderbook snapshots from QuestDB efficiently
- [x] **DATA-02**: Analytics can query trade history from QuestDB efficiently
- [x] **DATA-03**: Reader leverages QuestDB SAMPLE BY for time-based aggregations

### Cost Benchmarks

- [x] **COST-01**: Calculate VWAP for any time window and market
- [x] **COST-02**: Calculate TWAP for any time window and market
- [x] **COST-03**: Calculate arrival price (mid-price at order submission time)
- [x] **COST-04**: Calculate slippage in basis points (expected vs actual execution)

### Cost Analytics

- [x] **COST-05**: Decompose implementation shortfall into timing, impact, and spread components
- [x] **COST-06**: Estimate temporary market impact from orderbook/trade data
- [x] **COST-07**: Estimate permanent market impact from orderbook/trade data
- [x] **COST-08**: Forecast execution cost for a given order size ("what would $X cost?")

### Optimization

- [x] **OPT-01**: Calibrate Almgren-Chriss parameters from historical data
- [x] **OPT-02**: Generate optimal execution trajectories using Almgren-Chriss framework
- [x] **OPT-03**: Implement TWAP execution strategy
- [x] **OPT-04**: Implement VWAP execution strategy
- [x] **OPT-05**: Implement market order baseline strategy
- [x] **OPT-06**: Trade scheduling optimizer (optimal slice timing and sizing)

### Simulation

- [x] **SIM-01**: Event-driven execution simulation engine
- [x] **SIM-02**: Realistic fill models using actual orderbook depth
- [x] **SIM-03**: Run multiple strategies on same historical data
- [x] **SIM-04**: Compare strategy results with clear metrics (cost, risk, shortfall)
- [x] **SIM-05**: Prove better execution vs naive approaches through backtesting

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Simulation

- **SIM-06**: Queue position modeling for maker orders
- **SIM-07**: Market replay with tick-level reconstruction

### ML Enhancement

- **ML-01**: ML-based slippage prediction
- **ML-02**: Regime detection for adaptive parameter selection

### Real-time

- **RT-01**: Real-time cost monitoring during execution
- **RT-02**: Live execution recommendations

### Multi-asset

- **PORT-01**: Multi-asset portfolio execution optimization
- **PORT-02**: Cross-asset impact correlation

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real-time order routing | Analytics only, not trading system |
| FX market integration | Learning exercise; FX has different infrastructure |
| Sub-millisecond latency | Not HFT; analytics can be computed on demand |
| Production trading signals | Research/analytics tool, not production trading |
| Mobile/web dashboard | CLI-first; UI is v2+ if at all |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| COST-01 | Phase 1 | Complete |
| COST-02 | Phase 1 | Complete |
| COST-03 | Phase 1 | Complete |
| COST-04 | Phase 2 | Complete |
| COST-05 | Phase 2 | Complete |
| COST-06 | Phase 2 | Complete |
| COST-07 | Phase 2 | Complete |
| COST-08 | Phase 2 | Complete |
| OPT-01 | Phase 3 | Complete |
| OPT-02 | Phase 3 | Complete |
| OPT-03 | Phase 3 | Complete |
| OPT-04 | Phase 3 | Complete |
| OPT-05 | Phase 3 | Complete |
| OPT-06 | Phase 3 | Complete |
| SIM-01 | Phase 4 | Complete |
| SIM-02 | Phase 4 | Complete |
| SIM-03 | Phase 4 | Complete |
| SIM-04 | Phase 4 | Complete |
| SIM-05 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-01-19*
*Last updated: 2026-01-19 after Phase 4 completion*
