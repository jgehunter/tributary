# Phase 4: Simulation - Context

**Gathered:** 2026-01-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove better execution vs naive approaches through backtesting. Run execution simulations on historical data, compare strategies (Almgren-Chriss, TWAP, VWAP, market order) with realistic fill models, and demonstrate that optimized strategies reduce execution costs.

</domain>

<decisions>
## Implementation Decisions

### Fill Model Realism
- FIFO queue position approximation: estimate queue depth from orderbook, fill proportionally as volume trades through
- Partial fills allowed: slices can partially fill based on available liquidity, remainder carries forward
- Impact + recovery model: consuming liquidity depletes the book, but model partial recovery between slices based on time elapsed
- Market orders walk-the-book: simulate actual execution through orderbook levels until order filled

### Simulation Granularity
- Event-driven architecture: process each orderbook update/trade as an event (highest fidelity)
- Book depth: Claude's discretion based on order size relative to available depth
- Parallel isolated execution: run strategies in parallel but each sees clean orderbook (no cross-strategy impact)
- Millisecond time resolution: match QuestDB timestamp precision

### Comparison Metrics
- Multiple benchmarks: calculate both implementation shortfall AND VWAP slippage (and others), let user choose which to optimize
- Full risk profile: cost, variance, max drawdown during execution, worst-case scenarios
- Table + execution charts: summary table plus visual execution curves (holdings over time, cost accumulation)
- Risk-adjusted ranking: rank strategies by risk-adjusted performance, not just raw cost

### Claude's Discretion
- Orderbook depth selection based on order size
- Liquidity recovery rate calibration
- Chart visualization implementation details
- Edge case handling for thin/gappy data

</decisions>

<specifics>
## Specific Ideas

- The fill model should be realistic enough to capture the difference between strategies — if market orders and TWAP look similar in backtest, the model isn't capturing impact properly
- Event-driven simulation is essential for capturing the timing behavior of A-C vs TWAP (front-loaded vs uniform)
- Risk-adjusted ranking aligns with Almgren-Chriss philosophy: mean-variance optimization trades off cost vs risk

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-simulation*
*Context gathered: 2026-01-19*
