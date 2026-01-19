---
phase: 02-cost-analytics
plan: 01
subsystem: analytics
tags: [slippage, implementation-shortfall, perold-framework, execution-quality]

dependency-graph:
  requires:
    - 01-02: VWAP/TWAP/arrival price benchmarks
  provides:
    - calculate_slippage_bps function
    - ShortfallComponents dataclass
    - decompose_implementation_shortfall function
  affects:
    - 02-02: Cost forecasting (will use slippage calculations)
    - 02-03: Market impact estimation

tech-stack:
  added: []
  patterns:
    - Perold framework for implementation shortfall decomposition
    - Consistent sign convention (positive = cost)

key-files:
  created:
    - src/tributary/analytics/slippage.py
    - src/tributary/analytics/shortfall.py
    - tests/unit/test_slippage.py
    - tests/unit/test_shortfall.py
  modified:
    - src/tributary/analytics/__init__.py

decisions:
  - id: COST-SIGN-CONVENTION
    choice: Positive value = cost (unfavorable) for both buy and sell
    rationale: Consistent interpretation across all analytics functions
  - id: SHORTFALL-USD-BPS
    choice: Return both USD and basis point values
    rationale: USD for transparency, bps for comparison across different notional sizes

metrics:
  duration: 5m
  completed: 2026-01-19
---

# Phase 2 Plan 1: Slippage and Shortfall Summary

Slippage calculation and Perold implementation shortfall decomposition with consistent sign convention.

## One-liner

Slippage in bps and four-component shortfall decomposition (delay, trading, spread, opportunity cost) using Perold framework.

## What Was Built

### Slippage Calculation (slippage.py)
- `calculate_slippage_bps(execution_price, benchmark_price, side)` function
- Formula: `(execution_price - benchmark_price) / benchmark_price * 10000`
- Sign flipping for sell side (receiving less is a cost)
- Returns NaN for zero benchmark price
- Raises ValueError for invalid side

### Implementation Shortfall (shortfall.py)
- `ShortfallComponents` dataclass with 10 fields (5 bps + 5 USD)
- `decompose_implementation_shortfall()` function implementing Perold framework
- Components:
  - Delay cost: price movement from decision to order entry
  - Trading cost: market impact during execution
  - Spread cost: half-spread crossed (optional)
  - Opportunity cost: unfilled portion

### Module Exports
- Updated `__init__.py` to export all new functions and classes
- Clean imports: `from tributary.analytics import calculate_slippage_bps, ShortfallComponents, decompose_implementation_shortfall`

## Verification Results

```
tests/unit/test_slippage.py: 11 passed
tests/unit/test_shortfall.py: 15 passed
Total: 26 tests passing
```

Sign convention verified:
- Buy: paid 102 vs 100 = +200 bps (cost)
- Sell: received 98 vs 100 = +200 bps (cost)

Lint: `ruff check src/tributary/analytics/` - All checks passed

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sign convention | Positive = cost | Intuitive interpretation: positive is bad |
| Dual return values | Both USD and bps | USD for absolute cost, bps for comparison |
| Spread handling | Optional parameter | Not all executions have spread data |
| Zero notional | Return 0.0 bps | Can't compute meaningful bps with zero notional |

## Deviations from Plan

None - plan executed exactly as written.

## Requirements Satisfied

- **COST-04**: Calculate slippage in basis points - SATISFIED
- **COST-05**: Decompose implementation shortfall - SATISFIED

## Commits

| Hash | Message |
|------|---------|
| d609666 | feat(02-01): add slippage calculation in basis points |
| 3a529d0 | feat(02-01): add implementation shortfall decomposition |

## Next Phase Readiness

**Blockers:** None

**Ready for:** 02-02 (Cost forecasting) can proceed with orderbook-based slippage estimation.
