# Project Research Summary

**Project:** Tributary - Execution Analytics Layer
**Domain:** Execution Analytics / Transaction Cost Analysis for Thin Liquidity Markets
**Researched:** 2026-01-19
**Confidence:** HIGH

## Executive Summary

Tributary is building an execution analytics layer for prediction markets (Polymarket) and crypto, where thin liquidity fundamentally breaks assumptions from traditional equity TCA tools. The recommended approach combines Python's mature numerical ecosystem (numpy, pandas, scipy, numba) with custom implementations rather than off-the-shelf TCA libraries. The ecosystem is fragmented - no single library covers VWAP/TWAP, implementation shortfall, market impact modeling, and backtesting - but the building blocks are well-documented and can be composed into a modular analytics layer that sits alongside the existing QuestDB storage.

The recommended architecture treats analytics as a peer layer consuming data from QuestDB rather than intercepting the collection pipeline. This separation ensures collection reliability while allowing analytics to query historical data independently. The core pattern is: QuestDB Reader -> Cost Analytics -> Market Impact Calibration -> Almgren-Chriss Optimization -> Simulation Engine. Each component has clear boundaries and can be tested independently.

The primary risk is applying equity market assumptions to thin liquidity markets. Standard square-root impact models are calibrated for deep order books; Polymarket exhibits extreme liquidity concentration (63% of short-term markets have zero 24-hour volume). Every model must be validated empirically against actual execution data before trusting its predictions. Secondary risks include look-ahead bias in backtests and slippage calibration mismatches that can invalidate results entirely.

## Key Findings

### Recommended Stack

The stack leverages Python's mature numerical ecosystem with targeted additions for execution simulation. No single TCA library fits the prediction market use case - the recommendation is custom implementations built on established numerical foundations.

**Core technologies:**
- **numpy/pandas/scipy**: Foundation for array operations, time series, and optimization - already in project dependencies
- **numba**: JIT compilation for performance-critical loops (10-100x speedups) - essential for orderbook iteration
- **statsmodels**: Volatility estimation (GARCH), regime detection, impact parameter regression
- **hftbacktest**: HFT-grade orderbook simulation with queue position modeling - best-in-class for realistic fill simulation
- **almgren-chriss**: PyPI package for optimal execution model - use as reference, consider custom for flexibility

**What NOT to use:**
- tcapy (stale since 2021, FX-specific, heavy infrastructure)
- QuantConnect Lean (full trading platform, overkill for analytics)
- Zipline (Quantopian shut down 2020)
- QuantLib-Python (derivatives focus, wrong domain)

### Expected Features

**Must have (table stakes):**
- VWAP/TWAP benchmark calculations - industry standard, straightforward implementation
- Arrival price benchmark - mid-price at order submission
- Slippage calculation - expected vs actual execution in basis points
- Implementation shortfall - total cost decomposition
- Historical execution simulation - prove "better execution" claims
- Strategy vs benchmark comparison - minimum viable validation

**Should have (competitive):**
- Almgren-Chriss trajectory generation - optimal execution with risk/cost tradeoff
- Market impact estimation (temporary + permanent) - enables cost forecasting
- Multi-strategy comparison dashboard - AC vs TWAP vs VWAP vs Market Order
- Parameter calibration from historical data - data-driven optimization

**Defer (v2+):**
- Queue position modeling (HIGH complexity, unclear benefit for prediction markets)
- Market replay with tick-level reconstruction (research-grade complexity)
- ML-based slippage prediction (adds training overhead without clear win)
- Multi-asset portfolio optimization (scope expansion)
- Real-time algorithmic execution (requires production-grade reliability)

### Architecture Approach

Analytics layer sits as a peer to storage, consuming from QuestDB via PostgreSQL wire protocol. This isolates analytics failures from data collection and allows querying historical data without affecting live collection. The architecture follows the Query Layer pattern used by tcapy and similar TCA systems.

**Major components:**
1. **reader.py** - QuestDB query interface; leverages SAMPLE BY for VWAP/aggregations
2. **cost/** - Benchmarks (VWAP, TWAP), slippage, impact estimation, implementation shortfall
3. **optimization/** - Almgren-Chriss model, parameter calibration, strategy definitions
4. **simulation/** - Event-driven execution simulation with realistic fill models

**Key patterns:**
- Request/Result objects (Pydantic) for clear interfaces and validation
- Strategy abstract base class for polymorphic execution algorithms
- Separate calibration from execution (enables caching, A/B testing)
- Event-driven simulation loop (decouple market data, strategy, fills, metrics)

### Critical Pitfalls

1. **Applying equity market impact models to thin liquidity** - Square-root law assumes deep order books; Polymarket can have single orders consuming all liquidity. Start empirical, measure before modeling.

2. **Look-ahead bias in execution backtests** - Using VWAP calculated from future trades, or orderbook states that reflect information after your simulated order. Use event-driven simulation processing data sequentially.

3. **Permanent vs temporary impact mis-attribution** - Underestimating permanent impact means subsequent tranches execute worse than expected. Use dynamic estimation and validate with actual execution data.

4. **Slippage calibration mismatch** - Fixed 0.1% slippage is meaningless when prediction markets can show 1-3% for moderate orders. Use orderbook-based models with conservative estimates.

5. **VWAP/TWAP failures in thin markets** - When your order IS the volume, you benchmark against yourself. Use arrival price or implementation shortfall as primary benchmarks.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Foundation - QuestDB Reader and Cost Benchmarks
**Rationale:** Everything depends on querying data efficiently. Benchmarks are needed to measure anything. This must come first per architecture dependencies.
**Delivers:** QuestDB read interface leveraging SAMPLE BY; VWAP, TWAP, arrival price calculations
**Addresses:** Table stakes cost analytics, liquidity metrics from existing orderbook data
**Avoids:** Pitfall 2 (look-ahead bias) by establishing correct timestamp handling from the start

### Phase 2: Cost Analytics and Impact Estimation
**Rationale:** Cost analytics builds on benchmarks; impact estimation enables cost forecasting (answers "what would it cost to trade $X?")
**Delivers:** Slippage calculation, implementation shortfall decomposition, market impact estimation (square-root model adapted for thin markets)
**Uses:** statsmodels for regression, scipy for optimization
**Implements:** cost/ module components
**Avoids:** Pitfall 1 (equity models) by starting with empirical measurement; Pitfall 3 (impact mis-attribution) by separate estimation approaches

### Phase 3: Almgren-Chriss Optimization
**Rationale:** Requires calibrated impact parameters from Phase 2; answers "what's the optimal way to execute?"
**Delivers:** Parameter calibration from historical data, optimal trajectory generation, TWAP/VWAP/AC strategy implementations
**Uses:** almgren-chriss package as reference, scipy.optimize for custom implementation
**Implements:** optimization/ module
**Avoids:** Pitfall 5 (time inconsistency) by using appropriate objective function; Pitfall 6 (information leakage) by validating vs TWAP baseline

### Phase 4: Simulation and Backtesting
**Rationale:** Ties everything together; requires all components to prove "better execution"
**Delivers:** Event-driven simulation engine, realistic fill models using orderbook depth, multi-strategy comparison
**Uses:** hftbacktest for HFT-grade orderbook simulation, numba for performance
**Implements:** simulation/ module
**Avoids:** Pitfall 7 (VWAP/TWAP failures) by liquidity-aware scheduling; Pitfall 8 (slippage mismatch) by conservative orderbook-based models

### Phase 5: CLI Integration and Reporting
**Rationale:** User interface layer after core analytics are validated
**Delivers:** CLI commands for cost reports, calibration, simulation; configuration extension
**Implements:** CLI analytics command group; YAML configuration for calibration/simulation parameters

### Phase Ordering Rationale

- **Dependency chain:** Reader -> Benchmarks -> Cost Analytics -> Impact Estimation -> Optimization -> Simulation
- **Early pitfall mitigation:** Correct timestamp handling in Phase 1 prevents look-ahead bias from contaminating later phases
- **Empirical-first approach:** Phase 2 measures actual market behavior before Phase 3 builds models on assumptions
- **Validation before complexity:** Phase 4 proves strategies work before adding user-facing features in Phase 5

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Market impact calibration for thin liquidity is under-documented; will need experimentation with different functional forms
- **Phase 3:** Almgren-Chriss parameter interpretation varies across sources (lambda direction); needs careful validation

Phases with standard patterns (skip research-phase):
- **Phase 1:** VWAP/TWAP calculations are well-documented; QuestDB SAMPLE BY is standard
- **Phase 4:** Event-driven simulation has established patterns (QSTrader, hftbacktest examples)
- **Phase 5:** CLI extension follows existing Tributary patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Mature Python ecosystem; versions verified on PyPI; hftbacktest actively maintained |
| Features | HIGH | TCA literature consensus on core metrics; Almgren-Chriss well-documented |
| Architecture | HIGH | Follows established patterns from tcapy, QSTrader; clean separation of concerns |
| Pitfalls | HIGH | Multiple sources confirm thin liquidity issues; look-ahead bias well-documented |

**Overall confidence:** HIGH

### Gaps to Address

- **Prediction market impact models:** No off-the-shelf solution exists. Will require custom development with empirical validation. Plan for iteration in Phase 2.

- **Binary outcome semantics:** Implementation shortfall interpretation differs for binary positions. Document methodology explicitly during Phase 2.

- **24/7 liquidity patterns:** Need to profile actual Polymarket liquidity by hour/day before assuming patterns. Address in Phase 1 data exploration.

- **Parameter stability:** Impact parameters may vary significantly across calibration windows. Add stability checks in Phase 3.

## Sources

### Primary (HIGH confidence)
- [hftbacktest PyPI](https://pypi.org/project/hftbacktest/) - v2.4.4, December 2025; execution simulation
- [QuestDB Almgren-Chriss Glossary](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/) - model parameters
- [QuestDB Aggregate Functions](https://questdb.com/docs/query/functions/aggregation/) - SAMPLE BY patterns
- [statsmodels TSA](https://www.statsmodels.org/stable/tsa.html) - v0.14.6; time series analysis
- [scipy.optimize](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) - v1.17.0; optimization

### Secondary (MEDIUM confidence)
- [Talos Market Impact Model](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs) - crypto-specific impact
- [Anboto Labs TCA](https://medium.com/@anboto_labs/slippage-benchmarks-and-beyond-transaction-cost-analysis-tca-in-crypto-trading-2f0b0186980e) - crypto TCA patterns
- [QSTrader GitHub](https://github.com/mhallsmoore/qstrader) - modular backtesting architecture
- [tcapy GitHub](https://github.com/cuemacro/tcapy) - TCA architecture patterns (reference only; stale)

### Tertiary (LOW confidence)
- [PANews Polymarket Liquidity Analysis](https://www.panewslab.com/en/articles/d886495b-90ba-40bc-90a8-49419a956701) - liquidity concentration stats
- [Crypto.com Prediction Markets Research](https://crypto.com/en/research/prediction-markets-oct-2025) - market characteristics

---
*Research completed: 2026-01-19*
*Ready for roadmap: yes*
