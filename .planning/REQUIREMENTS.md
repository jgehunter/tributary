# Requirements: Tributary

**Defined:** 2026-01-19
**Core Value:** Demonstrably reduce execution costs vs naive approaches

## v1 Requirements

Requirements for Phase 2: Execution Intelligence. Each maps to roadmap phases.

### Data Access

- [ ] **DATA-01**: Analytics can query orderbook snapshots from QuestDB efficiently
- [ ] **DATA-02**: Analytics can query trade history from QuestDB efficiently
- [ ] **DATA-03**: Reader leverages QuestDB SAMPLE BY for time-based aggregations

### Cost Benchmarks

- [ ] **COST-01**: Calculate VWAP for any time window and market
- [ ] **COST-02**: Calculate TWAP for any time window and market
- [ ] **COST-03**: Calculate arrival price (mid-price at order submission time)
- [ ] **COST-04**: Calculate slippage in basis points (expected vs actual execution)

### Cost Analytics

- [ ] **COST-05**: Decompose implementation shortfall into timing, impact, and spread components
- [ ] **COST-06**: Estimate temporary market impact from orderbook/trade data
- [ ] **COST-07**: Estimate permanent market impact from orderbook/trade data
- [ ] **COST-08**: Forecast execution cost for a given order size ("what would $X cost?")

### Optimization

- [ ] **OPT-01**: Calibrate Almgren-Chriss parameters from historical data
- [ ] **OPT-02**: Generate optimal execution trajectories using Almgren-Chriss framework
- [ ] **OPT-03**: Implement TWAP execution strategy
- [ ] **OPT-04**: Implement VWAP execution strategy
- [ ] **OPT-05**: Implement market order baseline strategy
- [ ] **OPT-06**: Trade scheduling optimizer (optimal slice timing and sizing)

### Simulation

- [ ] **SIM-01**: Event-driven execution simulation engine
- [ ] **SIM-02**: Realistic fill models using actual orderbook depth
- [ ] **SIM-03**: Run multiple strategies on same historical data
- [ ] **SIM-04**: Compare strategy results with clear metrics (cost, risk, shortfall)
- [ ] **SIM-05**: Prove better execution vs naive approaches through backtesting

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
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| COST-01 | Phase 1 | Pending |
| COST-02 | Phase 1 | Pending |
| COST-03 | Phase 1 | Pending |
| COST-04 | Phase 2 | Pending |
| COST-05 | Phase 2 | Pending |
| COST-06 | Phase 2 | Pending |
| COST-07 | Phase 2 | Pending |
| COST-08 | Phase 2 | Pending |
| OPT-01 | Phase 3 | Pending |
| OPT-02 | Phase 3 | Pending |
| OPT-03 | Phase 3 | Pending |
| OPT-04 | Phase 3 | Pending |
| OPT-05 | Phase 3 | Pending |
| OPT-06 | Phase 3 | Pending |
| SIM-01 | Phase 4 | Pending |
| SIM-02 | Phase 4 | Pending |
| SIM-03 | Phase 4 | Pending |
| SIM-04 | Phase 4 | Pending |
| SIM-05 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-19*
*Last updated: 2026-01-19 after initial definition*
