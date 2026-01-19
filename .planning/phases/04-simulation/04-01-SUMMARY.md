---
phase: 04-simulation
plan: 01
subsystem: simulation-engine
tags: [events, fill-model, market-impact, liquidity]

dependency-graph:
  requires: [03-optimization]
  provides: [event-types, fill-model, liquidity-consumption]
  affects: [04-02-strategy-integration, 04-03-backtest-runner]

tech-stack:
  added: []
  patterns: [frozen-dataclasses, liquidity-tracking, exponential-recovery]

key-files:
  created:
    - src/tributary/analytics/simulation/__init__.py
    - src/tributary/analytics/simulation/events.py
    - src/tributary/analytics/simulation/fill_model.py
    - tests/unit/test_simulation_events.py
    - tests/unit/test_fill_model.py
  modified: []

decisions:
  - id: "SIM-01"
    choice: "Tuples for orderbook levels in frozen dataclasses"
    rationale: "Lists are mutable and can't be used in frozen dataclasses; tuples ensure immutability"
  - id: "SIM-02"
    choice: "Exponential recovery model with configurable half-life"
    rationale: "Captures realistic liquidity dynamics - fast initial recovery that slows over time"
  - id: "SIM-03"
    choice: "Track consumed liquidity per level index, not price"
    rationale: "Level index mapping allows same data structure as orderbook arrays; price-keyed would require float comparisons"

metrics:
  duration: 4m
  completed: 2026-01-19
---

# Phase 4 Plan 1: Events and Fill Model Summary

Event types and realistic fill model with liquidity consumption for execution simulation.

## One-liner

Frozen event dataclasses (MarketEvent, OrderEvent, FillEvent) and FillModel with per-level liquidity consumption showing 2.2x slippage difference between aggressive and patient execution.

## What Was Built

### Event Type Dataclasses (`events.py`)

Three frozen dataclasses forming the foundation of the event-driven simulation:

1. **MarketEvent** - Orderbook snapshot at a point in time
   - Timestamp, market_id, token_id
   - mid_price for slippage calculation
   - bid/ask prices and sizes as tuples (frozen-compatible)

2. **OrderEvent** - Trading instruction from a strategy
   - Timestamp, strategy_name, slice_index
   - size and side ('buy' or 'sell')

3. **FillEvent** - Execution result with full details
   - Requested vs filled size (partial fills)
   - Average execution price and slippage_bps
   - Levels consumed and mid_price_at_fill

### Fill Model (`fill_model.py`)

Realistic order execution simulation with:

- **Orderbook Walking**: Consumes liquidity level by level
- **Liquidity Tracking**: Remembers consumed amounts per side per level
- **Recovery Modeling**: Exponential decay with configurable half-life
- **Strategy Differentiation**: Large orders show higher slippage than equivalent TWAP slices

Key behavioral verification:
```
Large order (5000 units): 440 bps slippage
TWAP (5x1000 with reset): 200 bps slippage
Ratio: 2.2x more costly for aggressive execution
```

## Test Coverage

| Test File | Lines | Tests | Coverage Focus |
|-----------|-------|-------|----------------|
| test_simulation_events.py | 257 | 13 | Immutability, instantiation, hashability |
| test_fill_model.py | 468 | 21 | Slippage scaling, liquidity consumption, recovery, strategy differentiation |

Key test scenarios:
- Basic buy/sell fills
- Multi-level orderbook walking
- Consecutive orders consuming liquidity
- Partial and zero fills
- Recovery over time (0, partial, full)
- TWAP vs market order slippage comparison

## Key Implementation Details

### Liquidity Consumption

```python
# Track consumed liquidity per side per level
_consumed: dict[str, dict[int, float]] = {"buy": {}, "sell": {}}

# On execution, subtract from available sizes
adjusted = max(0.0, original_size - consumed[side].get(level, 0.0))
```

### Liquidity Recovery

```python
# Exponential decay model
decay_factor = 0.5 ** (elapsed_ms / half_life_ms)
recovery_factor = recovery_rate * (1 - decay_factor)
remaining_consumed = consumed * (1 - recovery_factor)
```

Default parameters:
- `recovery_rate = 0.5` (max 50% recovers)
- `half_life_ms = 1000.0` (1 second)

## Deviations from Plan

None - plan executed exactly as written.

## Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| Event types defined as frozen dataclasses | PASS |
| FillModel consumes liquidity from orderbook | PASS |
| FillModel models partial liquidity recovery | PASS |
| Large orders have higher slippage than small slices | PASS (2.2x) |
| All tests pass, code lint-clean | PASS (34 tests, ruff clean) |
| Module exports all public types | PASS |

## Next Phase Readiness

Ready for 04-02 (Strategy Integration):
- Event types available for strategy adapters
- FillModel ready for SimulationEngine integration
- All exports in `__init__.py`
