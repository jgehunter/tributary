# Feature Landscape: Execution Analytics

**Domain:** Execution Analytics / Transaction Cost Analysis
**Researched:** 2026-01-19
**Context:** Building for prediction markets (Polymarket) and crypto with thin liquidity; user wants to answer "If I need to trade $X, what would be the cost?" and "What's the optimal way to execute this?"
**Success metric:** Demonstrably better execution costs vs naive approaches (market orders, simple TWAP)

---

## Table Stakes

Features users expect from credible execution analytics. Missing = product feels incomplete or unusable.

### Cost Analytics (Core Metrics)

| Feature | Why Expected | Complexity | Implementation Notes |
|---------|--------------|------------|---------------------|
| **VWAP Benchmark Calculation** | Industry-standard execution benchmark; everyone compares to VWAP | Low | Straightforward calculation from trades table: sum(price * volume) / sum(volume) over time interval |
| **TWAP Benchmark Calculation** | Essential for thin-liquidity markets where volume prediction is unreliable | Low | Time-weighted average from orderbook snapshots or trade prices |
| **Arrival Price Benchmark** | Standard for systematic trading; compares execution to decision price | Low | Mid-price at order submission time from orderbook snapshot |
| **Slippage Calculation** | Core metric - difference between expected and actual execution price | Low | Formula: (Executed Price - Expected Price) / Expected Price * 10000 bps |
| **Implementation Shortfall** | Total cost measurement including all components (market impact, spread, delay) | Medium | IS = Paper Return - Actual Return; decompose into timing, spread, and impact components |
| **Spread Cost Attribution** | Isolates cost of crossing the bid-ask spread | Low | Half-spread cost: 0.5 * spread for each side crossed |

### Simulation and Backtesting (Foundation)

| Feature | Why Expected | Complexity | Implementation Notes |
|---------|--------------|------------|---------------------|
| **Historical Execution Simulation** | Can't claim "better execution" without proving it on historical data | Medium | Replay historical orderbook snapshots, simulate order fills with realistic assumptions |
| **Strategy vs Benchmark Comparison** | Core proof that optimization works; minimum viable validation | Low | Compare strategy returns vs VWAP/TWAP/Arrival across same historical periods |
| **Fill Simulation with Spread** | Realistic cost accounting - market orders pay full spread | Low | Buy fills at ask, sell fills at bid; limit orders fill at specified price if reachable |
| **Trade Breakdown Reporting** | Show execution quality per trade with contributing factors | Low | Report per-trade: slippage, spread paid, market conditions at execution |

### Data Requirements (Infrastructure)

| Feature | Why Expected | Complexity | Implementation Notes |
|---------|--------------|------------|---------------------|
| **Orderbook Snapshot Retrieval** | Foundation for all cost estimation and impact modeling | Low | Already have - 10s snapshots in QuestDB |
| **Trade History Access** | Foundation for calibration and backtesting | Low | Already have - trades table in QuestDB |
| **Market Liquidity Metrics** | Context for cost estimates; liquidity affects everything | Low | Compute from orderbook: depth at N bps, total bid/ask volume, imbalance |

---

## Differentiators

Features that set the system apart. Not expected by default, but provide competitive advantage for the stated goal.

### Market Impact Modeling

| Feature | Value Proposition | Complexity | Implementation Notes |
|---------|-------------------|------------|---------------------|
| **Temporary Impact Estimation** | Predicts immediate price movement from trading - enables cost forecasting | High | Calibrate from historical data: regress price impact vs order size, normalized by ADV/volatility |
| **Permanent Impact Estimation** | Predicts lasting price change - critical for large position management | High | Measure price level before/after trades at various lags; fit impact decay function |
| **Square-Root Impact Model** | Industry-standard functional form: Impact ~ sigma * sqrt(Q/V) | Medium | Parameters: sigma (volatility), Q (order size), V (daily volume); calibrate coefficients from data |
| **Orderbook Depth Impact** | Predicts impact based on actual liquidity available | Medium | Walk the orderbook to calculate fill price at various sizes; more accurate than statistical models |

### Optimal Execution (Almgren-Chriss Framework)

| Feature | Value Proposition | Complexity | Implementation Notes |
|---------|-------------------|------------|---------------------|
| **Almgren-Chriss Trajectory Generation** | Optimal trading schedule that minimizes cost + risk tradeoff | High | Requires: volatility (sigma), temp impact (eta), perm impact (gamma), risk aversion (lambda); outputs trade schedule |
| **Risk Aversion Parameter Calibration** | Personalize execution aggressiveness for user preferences | Medium | Allow user to set urgency/risk tradeoff; map to lambda parameter |
| **Front-Loaded vs Uniform Trajectories** | Show how different risk profiles translate to execution patterns | Medium | Visualize: aggressive (front-load) vs passive (uniform) schedules and cost implications |
| **Execution Horizon Optimization** | Determine optimal time to complete order given size and conditions | Medium | Optimize over horizon length; shorter = less timing risk, more impact; find sweet spot |

### Advanced Simulation

| Feature | Value Proposition | Complexity | Implementation Notes |
|---------|-------------------|------------|---------------------|
| **Queue Position Modeling** | Realistic limit order fills considering position in queue | High | Track orderbook changes; estimate fill probability based on depth ahead of order |
| **Market Replay Backtesting** | True out-of-sample testing with tick-level orderbook reconstruction | High | Replay orderbook updates in sequence; simulate strategy decisions and fills in real time |
| **Latency Simulation** | Account for execution delays in strategy evaluation | Medium | Add configurable delay between signal and execution; measure impact on performance |
| **Child Order Simulation** | Model splitting parent orders into children for execution | Medium | Simulate TWAP/VWAP slicing; track individual child fills and aggregate |

### Strategy Comparison Framework

| Feature | Value Proposition | Complexity | Implementation Notes |
|---------|-------------------|------------|---------------------|
| **Multi-Strategy Comparison Dashboard** | Compare Almgren-Chriss vs TWAP vs VWAP vs Market Order across scenarios | Medium | Run same historical scenarios through each strategy; tabulate costs, risks, consistency |
| **Parameter Sensitivity Analysis** | Show how results change with different assumptions | Medium | Vary: risk aversion, impact parameters, volatility; plot cost curves |
| **Confidence Intervals on Cost Estimates** | Quantify uncertainty in predictions | Medium | Bootstrap historical data or use analytical bounds from model |
| **Performance Attribution** | Decompose why one strategy beat another | Medium | Break down: timing, spread capture, impact avoidance, luck/noise |

---

## Anti-Features

Features to explicitly NOT build. Common in execution analytics but wrong for this scope.

### Real-Time Execution Infrastructure

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Live Order Routing** | Scope creep - analytics project, not execution system; regulatory/security complexity | Focus on simulation/backtesting that informs manual execution decisions |
| **Smart Order Router (SOR)** | Requires integration with exchange APIs, order management; major engineering effort | Simulate routing decisions against historical data instead |
| **Sub-Millisecond Latency Optimization** | Not HFT; prediction markets have seconds-scale dynamics | Optimize for correctness and insight, not speed |
| **Real-Time Algorithmic Execution** | Requires production-grade reliability, monitoring, failover | Output optimal schedules for manual or external execution |

### Enterprise TCA Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **80+ KPI Dashboard** | Complexity for enterprise sales, not learning; most metrics won't be used | Focus on 5-10 core metrics that answer the key questions |
| **MiFID II / Regulatory Reporting** | Regulatory compliance overhead; not needed for personal analytics | Skip compliance features entirely |
| **Multi-Asset Portfolio Optimization** | Adds correlation complexity; single-asset execution is hard enough | Start with single-asset execution; portfolio is future milestone |
| **Broker/Venue Comparison** | Requires data from multiple venues; Polymarket is single venue | Optimize execution on single venue first |

### Over-Engineering

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **ML-Based Slippage Prediction** | Adds training complexity; simpler models work for initial version | Use analytical models (square-root, Almgren-Chriss) calibrated from data |
| **Reinforcement Learning Execution** | Research-grade complexity; unclear if better than analytical solutions | Implement analytical optimal execution first; ML is future extension |
| **News/Sentiment Integration** | Additional data source complexity; unclear alpha for execution | Focus on orderbook and trade data which are already collected |
| **Real-Time Volatility Forecasting** | Requires streaming infrastructure; historical volatility sufficient for backtesting | Use realized volatility from historical data for calibration |

---

## Feature Dependencies

```
                      ┌─────────────────────────────────────┐
                      │        DATA FOUNDATION              │
                      │  (Already have: orderbook + trades) │
                      └───────────────┬─────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              v                       v                       v
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ COST ANALYTICS  │    │  LIQUIDITY      │    │   VOLATILITY    │
    │ VWAP, TWAP,     │    │  METRICS        │    │   ESTIMATION    │
    │ Arrival, IS     │    │  Depth, spread  │    │   From trades   │
    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
             │                      │                      │
             │         ┌────────────┴────────────┐         │
             │         │                         │         │
             v         v                         v         v
    ┌────────────────────────┐       ┌────────────────────────┐
    │   MARKET IMPACT        │       │   BASIC SIMULATION     │
    │   Temp + Perm impact   │<──────│   Historical replay    │
    │   calibration          │       │   with spread costs    │
    └───────────┬────────────┘       └───────────┬────────────┘
                │                                │
                └────────────┬───────────────────┘
                             │
                             v
                ┌────────────────────────┐
                │   ALMGREN-CHRISS       │
                │   Optimal trajectories │
                │   Risk/cost tradeoff   │
                └───────────┬────────────┘
                            │
                            v
                ┌────────────────────────┐
                │   STRATEGY COMPARISON  │
                │   AC vs TWAP vs VWAP   │
                │   Performance proof    │
                └────────────────────────┘
```

### Critical Path for MVP

1. **Cost Analytics** (Week 1) - VWAP, TWAP, Arrival, Slippage calculation
2. **Liquidity Metrics** (Week 1) - Depth analysis, spread metrics from existing orderbook data
3. **Volatility Estimation** (Week 2) - Historical volatility from trade data
4. **Basic Simulation** (Week 2-3) - Simulate execution against historical orderbooks
5. **Market Impact Calibration** (Week 3-4) - Fit impact models from historical data
6. **Almgren-Chriss Implementation** (Week 4-5) - Optimal trajectory generation
7. **Strategy Comparison** (Week 5-6) - Prove AC beats naive approaches

---

## MVP Recommendation

For MVP, prioritize features that directly answer the user's questions:

**"If I need to trade $X, what would be the cost?"**
1. **Slippage calculation** from orderbook depth walk - immediate, accurate cost estimate
2. **Spread cost** - guaranteed minimum cost for market orders
3. **Impact estimation** (square-root model) - predicted additional cost for order size
4. **Total cost estimate** - spread + estimated impact

**"What's the optimal way to execute this?"**
1. **TWAP baseline** - simplest "optimal" that's hard to beat in thin markets
2. **Almgren-Chriss trajectory** - parameterized optimal given risk/cost tradeoff
3. **Historical comparison** - show that AC/TWAP beats market orders

**MVP Scope:**
- Table stakes cost metrics (VWAP, TWAP, IS, slippage)
- Basic market impact model (calibrated from data)
- Almgren-Chriss trajectory generation
- Historical simulation comparing strategies
- Simple reporting dashboard

**Defer to Post-MVP:**
- Queue position modeling (HIGH complexity, unclear benefit for prediction markets)
- Market replay with tick-level reconstruction (HIGH complexity)
- ML-based enhancements (research-grade, not needed for proof)
- Multi-asset portfolio optimization (scope expansion)

---

## Confidence Assessment

| Category | Confidence | Reasoning |
|----------|------------|-----------|
| Table Stakes Features | HIGH | Well-documented in TCA literature; industry consensus on core metrics |
| Almgren-Chriss Parameters | HIGH | Verified via QuestDB glossary (authoritative); academic papers consistent |
| Market Impact Models | MEDIUM | Standard approaches documented; calibration for thin markets needs validation |
| Simulation Complexity | MEDIUM | General approaches known; prediction market specifics need experimentation |
| MVP Scope | MEDIUM | Based on user goals; may need iteration based on actual data characteristics |

---

## Sources

**Transaction Cost Analysis:**
- [Talos: Execution Insights Through TCA](https://www.talos.com/insights/execution-insights-through-transaction-cost-analysis-tca-benchmarks-and-slippage)
- [Anboto Labs: Slippage, Benchmarks and Beyond](https://medium.com/@anboto_labs/slippage-benchmarks-and-beyond-transaction-cost-analysis-tca-in-crypto-trading-2f0b0186980e)
- [LSEG: How to Build End-to-End TCA Framework](https://developers.lseg.com/en/article-catalog/article/build-end-to-end-transaction-cost-analysis-framework)

**Almgren-Chriss Model:**
- [QuestDB: Almgren-Chriss Model Glossary](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/)
- [QuestDB: Execution Slippage Measurement](https://questdb.com/glossary/execution-slippage-measurement/)
- [Almgren-Chriss Original Paper](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [Dean Markwick: Solving the Almgren-Chriss Model](https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html)
- [GitHub: Almgren-Chriss Implementation](https://github.com/joshuapjacob/almgren-chriss-optimal-execution)

**Market Impact:**
- [Talos: Market Impact Model for Crypto](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs)
- [QuestDB: Market Impact Models](https://questdb.com/glossary/market-impact-models/)
- [Lillo: Market Impact Models Lecture](http://market-microstructure.institutlouisbachelier.org/uploads/91_1%20LILLO%20paris2014Lillomerge.pdf)

**Slippage Analysis:**
- [QuantJourney: Slippage Comprehensive Analysis](https://quantjourney.substack.com/p/slippage-a-comprehensive-analysis)
- [Stephen Diehl: Slippage Modelling](https://www.stephendiehl.com/posts/slippage/)
- [LuxAlgo: Trading Slippage](https://www.luxalgo.com/blog/trading-slippage-minimize-hidden-costs/)

**Backtesting:**
- [HFTBacktest GitHub](https://github.com/nkaz001/hftbacktest)
- [Limit Order Book Simulations Review](https://arxiv.org/html/2402.17359v1)
- [CoinAPI: Order Book Replay Guide](https://www.coinapi.io/blog/crypto-order-book-replay)

**Crypto/Prediction Market Context:**
- [Amberdata: Temporal Patterns in Market Depth](https://blog.amberdata.io/the-rhythm-of-liquidity-temporal-patterns-in-market-depth)
- [Crypto.com: Prediction Markets Research](https://crypto.com/en/research/prediction-markets-oct-2025)

---

*Research conducted: 2026-01-19*
