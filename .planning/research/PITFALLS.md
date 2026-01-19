# Execution Analytics Pitfalls

**Domain:** Execution analytics for thin liquidity markets (prediction markets, crypto)
**Researched:** 2026-01-19
**Project Context:** eFX trader building execution analytics for Polymarket (prediction market), expanding to crypto

---

## Critical Pitfalls

Mistakes that cause fundamentally wrong results or require complete rewrites.

---

### Pitfall 1: Applying Equity Market Impact Models to Thin Liquidity Markets

**What goes wrong:**
Standard market impact models (square-root law, Almgren-Chriss) are calibrated on liquid equity markets with deep order books and continuous trading. Applying these directly to prediction markets or illiquid crypto pairs produces wildly inaccurate cost estimates.

The square-root law assumes:
- Constant liquidity
- Deep order books where meta-orders can be split without information leakage
- Participation rates where the law holds (typically 0.5%-15% of daily volume)

Prediction markets like Polymarket exhibit:
- Extreme liquidity concentration (63% of short-term markets have zero 24-hour volume)
- Order books where a single market order can consume all liquidity at multiple price levels
- Participation rates that routinely exceed model assumptions
- Liquidity that appears/disappears based on event proximity (U.S. politics markets average $450k depth; sports markets often <$100)

**Why it happens:**
Academic literature focuses on equity markets. The models are elegant, well-documented, and easy to implement. Developers default to these without questioning applicability.

**Consequences:**
- Cost estimates off by 10x-100x
- Optimization produces "optimal" trajectories that are actually worse than naive execution
- Backtest results don't translate to live performance
- False confidence in execution quality

**Warning signs:**
- Model predicts <1% slippage but actual execution shows 5%+
- "Optimal" execution consistently underperforms TWAP in live trading
- Model parameters (eta, lambda) require extreme values to fit historical data
- Residuals show systematic patterns rather than volatility-explained noise

**Prevention:**
1. **Start empirical, not theoretical.** Build market impact estimates from your actual execution data first, not from academic models
2. **Measure order book depth distributions** in your target markets before choosing a model
3. **Use participation rate limits** much lower than equity markets (cap at 1-2% of available liquidity, not 5-10%)
4. **Accept linear models** for very thin markets where square-root law crossover to linear regime dominates
5. **Build liquidity regime detection** - different models for different market states

**Phase recommendation:** Address in Phase 1 (Market Impact Modeling). Start with empirical measurement before any model implementation.

**Confidence:** HIGH - Multiple sources confirm square-root law limitations in thin markets. Talos specifically developed separate crypto models because equity models don't transfer.

**Sources:**
- [Bouchaud on Square-Root Law](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact)
- [Talos Crypto Market Impact Model](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs)
- [PANews Polymarket Liquidity Analysis](https://www.panewslab.com/en/articles/d886495b-90ba-40bc-90a8-49419a956701)

---

### Pitfall 2: Look-Ahead Bias in Execution Backtests

**What goes wrong:**
Backtesting execution strategies requires using only information available at decision time. Look-ahead bias sneaks in through:

1. **Using final day's VWAP to evaluate intraday decisions** - The VWAP benchmark includes trades that hadn't happened yet when you made the decision
2. **Order book state reconstruction** - Using order book snapshots that reflect information arriving after your simulated order would have been placed
3. **Slippage models using full-day statistics** - Volatility, spread, and volume statistics computed from data not yet available
4. **Fill assumption at mid-price** - Assuming fills at mid when your order would have moved the market

**Why it happens:**
- Easier to code with full historical data than event-driven simulation
- Subtle timestamp misalignments (off-by-one errors in data alignment)
- Using aggregate statistics (daily volatility) instead of running estimates
- Financial statement data releases not properly time-stamped

**Consequences:**
"Look-ahead is the worst type of bias because the results are wrong" - unlike other biases revealed by regime change, look-ahead bias is immediately revealed in real-world execution. Your backtest shows 15% annual improvement; live trading shows 0% or negative.

**Warning signs:**
- Very smooth equity curve (straight line in log chart)
- Sharpe ratio > 1.5 for execution alpha
- MAR ratio > 1
- Strategy always "knows" optimal execution timing
- Results degrade immediately when going live

**Prevention:**
1. **Use event-driven backtesting** that processes data sequentially, never accessing future data
2. **Bitemporal data handling** - Track both event time and knowledge time
3. **Walk-forward validation** - Define strategy rules in-sample, validate out-of-sample
4. **Arrival price benchmarks** - Use price at order arrival, not price at execution completion
5. **Code review specifically for look-ahead** - Audit every data access for timestamp correctness

**Phase recommendation:** Address in Phase 3 (Backtesting Framework). This is architectural - get it right before building analytics on top.

**Confidence:** HIGH - Well-documented bias with clear detection methods.

**Sources:**
- [Michael Harris on Look-Ahead Bias Detection](https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879)
- [LuxAlgo Backtesting Traps](https://www.luxalgo.com/blog/backtesting-traps-common-errors-to-avoid/)
- [Portfolio Optimization Book - Backtesting](https://bookdown.org/palomar/portfoliooptimizationbook/8.2-seven-sins.html)

---

### Pitfall 3: Permanent vs Temporary Impact Mis-Attribution

**What goes wrong:**
Market impact has components:
- **Temporary impact:** Cost that decays after your trade (liquidity consumption)
- **Permanent impact:** Price change that persists (information revelation)

Mis-attributing between them causes:
1. **Underestimating permanent impact** - You think prices will revert; they don't. Subsequent tranches execute worse than expected.
2. **Overestimating permanent impact** - You trade too fast to "beat" information leakage that isn't happening
3. **Assuming linear permanent impact** - Academic theory requires linear permanent impact to avoid arbitrage, but empirical data shows non-linearity

In thin markets, this is especially dangerous because:
- A single order can be large enough to reveal information (permanent) rather than just consume liquidity (temporary)
- Recovery times are longer and more variable
- Permanent impact is harder to separate from random price movements

**Why it happens:**
- Regression of price increments on volumes conflates permanent and temporary
- Standard 5-minute aggregation periods lose microstructure information
- Static estimation over long periods misses regime changes
- Non-linear permanent impact biases instantaneous impact estimation

**Consequences:**
- Almgren-Chriss optimization produces wrong trajectories (trades too fast or too slow)
- Implementation shortfall attribution is wrong (blaming market vs. execution decisions)
- Cost predictions systematically biased

**Warning signs:**
- Price doesn't revert as your model predicts after completing an order
- Estimated permanent impact parameter varies wildly across calibration windows
- Residuals show systematic patterns
- Model requires different parameters for buys vs. sells

**Prevention:**
1. **Use transient impact models** that allow non-linear permanent impact (e.g., propagator models)
2. **Dynamic estimation** - Update impact parameters intraday, not static daily estimates
3. **Separate calibration** - Estimate permanent and temporary components separately with appropriate methodologies
4. **Validate with actual execution data** - Compare predicted vs realized price paths post-execution

**Phase recommendation:** Address in Phase 1 (Market Impact Modeling). Fundamental to getting cost estimates right.

**Confidence:** MEDIUM - Academic literature is clear on theory, but practical separation remains challenging. Multiple estimation approaches exist.

**Sources:**
- [Imperial College - Market Impact Models](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/cfm-imperial-institute-of-quantitative-finance/events/Lillo-Imperial-Lecture3.pdf)
- [Arxiv - Calibration of Optimal Execution](https://arxiv.org/pdf/1206.0682)
- [UPenn - Direct Estimation of Equity Market Impact](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf)

---

### Pitfall 4: Implementation Shortfall Calculation Errors

**What goes wrong:**
Implementation Shortfall (IS) measures the cost of executing vs. a "paper" portfolio. Errors arise from:

1. **Benchmark timestamp errors** - Using wrong "decision price" (arrival time of order to desk vs. manager decision time)
2. **Incomplete cost attribution** - Not separating delay costs, spread costs, impact costs, timing costs
3. **Benchmark selection mismatch** - Using VWAP benchmark when IS is the goal, or vice versa
4. **Partial fill handling** - Not accounting for opportunity cost of unfilled orders
5. **Multi-day order handling** - How to attribute overnight gaps

For prediction markets specifically:
- "Decision price" may not exist if market was illiquid when decision was made
- Binary outcome markets have different IS semantics (you either have the position or you don't)
- Market makers may withdraw liquidity upon seeing your interest

**Why it happens:**
- "There are many different ways to calculate Implementation Shortfall" - no single standard
- Communication gaps between portfolio manager and trading desk about order timing
- Systems not designed for thin markets where fills are lumpy, not smooth

**Consequences:**
- Execution quality appears good/bad when it's actually the opposite
- Wrong attribution leads to wrong improvement priorities
- Comparison across strategies becomes meaningless

**Warning signs:**
- IS varies wildly depending on calculation method chosen
- Large unexplained residual in cost decomposition
- Delay cost dominates when it shouldn't (or vice versa)
- Trader performance metrics don't match intuition

**Prevention:**
1. **Document and version your IS methodology** - Be explicit about every choice
2. **Separate cost components clearly** - Delay, spread, impact, timing
3. **Use arrival price as primary benchmark** for thin markets (it's unambiguous)
4. **Handle partial fills explicitly** - Include opportunity cost of unfilled quantity
5. **Validate against known trades** - Manual verification on sample executions

**Phase recommendation:** Address in Phase 2 (Cost Analysis). Define methodology before building automated analysis.

**Confidence:** HIGH - Well-documented calculation variations with clear methodological choices.

**Sources:**
- [LSEG Transaction Cost Analysis Framework](https://developers.lseg.com/en/article-catalog/article/build-end-to-end-transaction-cost-analysis-framework)
- [Wikipedia - Transaction Cost Analysis](https://en.wikipedia.org/wiki/Transaction_cost_analysis)
- [KDB TCA Documentation](https://code.kx.com/q/wp/transaction-cost/)

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or degraded accuracy.

---

### Pitfall 5: Almgren-Chriss Time Inconsistency

**What goes wrong:**
The original Almgren-Chriss model uses mean-variance optimization, which is time-inconsistent. An "optimal" trajectory computed at t=0 is no longer optimal if recomputed at t=1 with updated information.

This means:
- Your optimizer says "trade X shares now" at 9:00 AM
- At 9:30 AM, with new information, the optimizer says you should have traded Y shares at 9:00 AM
- Following the original plan is suboptimal; replanning continuously leads to instability

**Why it happens:**
Mean-variance framework is tractable and well-understood, but has known time-consistency issues. Many implementations use it anyway for simplicity.

**Consequences:**
- Execution trajectories that "don't feel right"
- Difficulty explaining why strategy changed mid-execution
- Instability when adapting to new information

**Prevention:**
1. **Use mean-quadratic-variation objective** instead of mean-variance (addresses time inconsistency)
2. **Accept that trajectories are guidelines** - Don't recompute too frequently
3. **Use adaptive algorithms** that are designed for online updating (e.g., reinforcement learning approaches)

**Phase recommendation:** Address in Phase 2 (Almgren-Chriss implementation) when choosing objective function.

**Confidence:** MEDIUM - Academic literature documents the issue, but practical impact in thin markets may be dominated by other errors.

**Sources:**
- [Arxiv - Almgren-Chriss with Geometric Brownian Motion](https://arxiv.org/pdf/2006.11426)
- [Dean Markwick - Solving Almgren-Chriss](https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html)

---

### Pitfall 6: Information Leakage from Front-Loaded Execution

**What goes wrong:**
Classic Almgren-Chriss says: trade fastest at the beginning to minimize timing risk. In thin markets, this:
- Reveals your intent to other participants
- Allows front-running of subsequent tranches
- Causes more permanent impact than predicted

The model assumes other participants don't adapt to your behavior. In thin markets with few participants, this assumption is very wrong.

**Why it happens:**
Original model didn't consider strategic interaction with other traders. In equity markets with many participants, your individual trades are small signals. In thin markets, you may be the dominant signal.

**Consequences:**
- Actual permanent impact exceeds model prediction
- "Optimal" front-loaded strategy underperforms more patient approaches
- Training data from patient execution may not transfer to aggressive execution

**Prevention:**
1. **Test against naive TWAP baseline** - If your "optimal" strategy doesn't beat TWAP, something's wrong
2. **Randomize execution** - Add noise to prevent predictability
3. **Consider information-aware models** that account for price response to your revealed intent
4. **Start with conservative (slower) execution** until you have data on actual market response

**Phase recommendation:** Address in Phase 2 when implementing execution strategies. Validate empirically before trusting model output.

**Confidence:** MEDIUM - Documented in academic literature, but magnitude of effect in prediction markets is uncertain.

**Sources:**
- [Academia - Optimal Execution with Information Leakage](https://www.academia.edu/22938911/Optimal_Execution_of_Portfolio_Transactions_the_Effect_of_Information_Leakage)
- [Anboto Labs - Almgren-Chriss Framework](https://medium.com/@anboto_labs/deep-dive-into-is-the-almgren-chriss-framework-be45a1bde831)

---

### Pitfall 7: VWAP/TWAP Algorithm Failures in Thin Markets

**What goes wrong:**

**VWAP failures:**
- Relies on volume prediction; prediction errors are high in thin/volatile markets
- Large orders relative to market volume break the algorithm
- In thin markets, your order IS the volume - you're benchmarking against yourself

**TWAP failures:**
- Fixed time slicing ignores liquidity availability
- In thin markets, liquidity can completely disappear for intervals
- Predictable execution patterns allow exploitation

**Specific prediction market issues:**
- Liquidity spikes around events (news, market opens)
- 22.9% of Polymarket markets have cycles under 1 day - VWAP over a day is meaningless
- Liquidity fragments across related markets (YES/NO shares for same event)

**Why it happens:**
VWAP/TWAP are designed for liquid equity markets with predictable intraday volume patterns. These assumptions don't hold in thin markets.

**Consequences:**
- Benchmark becomes meaningless (comparing to your own trades)
- Execution during illiquid periods causes excessive slippage
- Strategy appears optimal in backtest, fails live

**Prevention:**
1. **Use arrival price or implementation shortfall as primary benchmark** for thin markets
2. **Adaptive slicing** - modify order sizes based on real-time liquidity
3. **Randomization** - avoid predictable execution patterns
4. **Liquidity-aware scheduling** - don't trade during known illiquid periods
5. **Volume participation caps** - limit to fraction of available liquidity, not target volume

**Phase recommendation:** Address in Phase 2-3. Build liquidity-aware execution before attempting VWAP optimization.

**Confidence:** HIGH - Well-documented limitations, especially in crypto and DeFi contexts.

**Sources:**
- [Arxiv - Deep Learning for VWAP in Crypto](https://arxiv.org/html/2502.13722v1)
- [Chainlink - TWAP vs VWAP](https://chain.link/education-hub/twap-vs-vwap)
- [Amberdata - Global VWAP vs TWAP](https://blog.amberdata.io/comparing-global-vwap-and-twap-for-better-trade-execution)

---

### Pitfall 8: Slippage Model Calibration Mismatch

**What goes wrong:**
Slippage in backtests is often:
- A fixed percentage (e.g., 0.1%)
- Based on equity market benchmarks
- Static across all market conditions

Reality in thin markets:
- Slippage varies 10x-100x depending on liquidity conditions
- Can exceed 1-3% for moderate orders in low-volume environments
- Changes dramatically around events

A backtest showing 15% annual returns can collapse to near-zero after realistic slippage.

**Why it happens:**
- Easier to use fixed slippage than model it
- Equity market rules-of-thumb (0.02-0.10 per share) don't apply
- Order book data not available for accurate modeling

**Consequences:**
- Strategies profitable in backtest are unprofitable live
- Under-estimated trading frequency costs
- Over-optimized for historical conditions

**Warning signs:**
- Backtest equity curve is suspiciously smooth
- Win rate degrades significantly when going live
- Actual fill prices consistently worse than expected

**Prevention:**
1. **Conservative slippage estimates** - Assume worse-than-average execution quality
2. **Order-book-based slippage models** - Use actual depth data
3. **Regime-specific slippage** - Different estimates for different market conditions
4. **Participation rate caps** - Don't assume you can trade more than a fraction of available liquidity

Specific numbers for reference:
- Liquid crypto (BTC/ETH): <1.5 basis points spread
- Less liquid crypto: >2.5 basis points spread
- Thin prediction markets: can exceed 1-3% slippage for institutional sizes

**Phase recommendation:** Address in Phase 3 (Backtesting Framework). Build conservative slippage models before trusting any backtest.

**Confidence:** HIGH - Universal issue with clear solutions.

**Sources:**
- [LuxAlgo - Slippage and Liquidity](https://www.luxalgo.com/blog/backtesting-limitations-slippage-and-liquidity-explained/)
- [FasterCapital - Slippage in Backtesting](https://www.fastercapital.com/content/Slippage--The-Hidden-Costs--Accounting-for-Slippage-in-Backtesting.html)
- [QuantConnect - Slippage Modeling](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts)

---

### Pitfall 9: Parameter Overfitting in Execution Models

**What goes wrong:**
Execution cost models have parameters (impact coefficients, decay rates, volatility scaling). Calibrating these to historical data leads to:
- Parameters that fit past data but don't generalize
- Overly complex models that capture noise as signal
- Models that work in backtest but fail live

"Whatever optimal parameters one found are likely to suffer from data snooping bias"

**Why it happens:**
- Not enough historical trades for statistical significance
- High dimensionality with correlated regressors
- Multiple parameters allow fitting noise
- No clear in-sample/out-of-sample split for execution data

**Consequences:**
- False confidence in cost predictions
- "Optimal" strategies that are actually random
- Inability to distinguish skill from luck

**Prevention:**
1. **Simple models first** - Start with 1-2 parameters, add complexity only when justified
2. **Walk-forward validation** - Always validate on unseen data
3. **Parameter stability checks** - Good parameters should be stable across calibration windows
4. **Synthetic data validation** - Generate simulated price series to test for overfitting
5. **Cross-validation** - Use multiple train/test splits
6. **Regularization** - Penalize model complexity

**Phase recommendation:** Address throughout, but especially Phase 1-2 when building models.

**Confidence:** HIGH - Universal statistical issue with well-established solutions.

**Sources:**
- [Predictnow - Optimizing Without Overfitting](https://predictnow-ai.medium.com/optimizing-trading-strategies-without-overfitting-365fbf1dc5fe)
- [Portfolio Optimization Book - Backtesting](https://portfoliooptimizationbook.com/slides/slides-backtesting.pdf)
- [SSRN - Detecting Overfitting](https://papers.ssrn.com/sol3/Delivery.cfm/5331456.pdf?abstractid=5331456&mirid=1)

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

---

### Pitfall 10: 24/7 Market Microstructure Ignorance

**What goes wrong:**
Crypto and prediction markets trade 24/7. Traditional equity microstructure patterns don't apply:
- No market open/close auctions
- No lunch hour patterns
- Liquidity cycles follow global time zones, not exchange hours

But liquidity DOES have patterns:
- 1.42x peak/trough liquidity relationship in crypto
- Liquidity correlates with attention (event proximity in prediction markets)
- Weekend vs. weekday patterns

**Consequences:**
- Execution timing based on equity market intuition fails
- Miss optimal execution windows
- Underestimate execution costs during low-liquidity periods

**Prevention:**
1. **Profile liquidity by hour/day** for your specific markets
2. **Build event-aware scheduling** for prediction markets
3. **Don't assume 24/7 means uniform liquidity**

**Phase recommendation:** Address in Phase 1 when building market data collection.

**Confidence:** HIGH - Well-documented in crypto microstructure research.

**Sources:**
- [Amberdata - Temporal Patterns in Market Depth](https://blog.amberdata.io/the-rhythm-of-liquidity-temporal-patterns-in-market-depth)
- [Easley et al. - Microstructure in Crypto Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4814346)

---

### Pitfall 11: Survivorship Bias in Market Selection

**What goes wrong:**
Analyzing execution quality across "markets that exist today" excludes:
- Markets that resolved and closed
- Markets that failed due to low liquidity
- Markets that were manipulated

This creates false confidence - you're studying the survivors.

**Consequences:**
- Overestimate average market quality
- Miss patterns that predict market failure
- Models trained on survivors don't generalize

**Prevention:**
1. **Include resolved/closed markets** in historical analysis
2. **Track market lifecycle** - creation, active trading, resolution
3. **Weight analysis by market-time, not current market count**

**Phase recommendation:** Address in Phase 3 (Backtesting). Ensure historical data includes closed markets.

**Confidence:** MEDIUM - Standard bias, specific impact in prediction markets uncertain.

**Sources:**
- [Portfolio Optimization Book - Seven Sins](https://bookdown.org/palomar/portfoliooptimizationbook/8.2-seven-sins.html)

---

### Pitfall 12: Risk Aversion Parameter (Lambda) Misinterpretation

**What goes wrong:**
Almgren-Chriss has a risk aversion parameter lambda. Common misunderstandings:
- Higher lambda = more risk averse = slower trading (WRONG - it's the opposite)
- Lambda = 0 means risk neutral = TWAP
- Lambda is a fixed personality trait (it should adapt to market conditions)

Some sources contradict each other on interpretation, causing implementation confusion.

**Consequences:**
- Optimization produces opposite of intended behavior
- Parameters set based on wrong intuition
- Difficulty tuning for desired behavior

**Prevention:**
1. **Test empirically** - Verify parameter changes have expected directional effect
2. **Document your convention** - Which way does lambda go in YOUR implementation?
3. **Start with lambda = 0 (TWAP)** and adjust from known baseline

**Phase recommendation:** Address in Phase 2 (Almgren-Chriss implementation).

**Confidence:** MEDIUM - Multiple sources show inconsistent conventions.

**Sources:**
- [QuestDB - Almgren-Chriss Model](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/)
- [SimTrade - Understanding Almgren-Chriss](https://www.simtrade.fr/blog_simtrade/understanding-almgren-chriss-model-for-optimal-trade-execution/)

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|---------------|------------|
| Market Impact Modeling | Applying equity models to thin markets | Start empirical; measure before modeling |
| Market Impact Modeling | Permanent/temporary mis-attribution | Use transient models; dynamic estimation |
| Cost Analysis | IS calculation methodology errors | Document methodology; manual validation |
| Almgren-Chriss Implementation | Time inconsistency; information leakage | Use appropriate objective; validate vs TWAP |
| Almgren-Chriss Implementation | Lambda parameter confusion | Test directionally; document convention |
| Backtesting Framework | Look-ahead bias | Event-driven architecture; code audit |
| Backtesting Framework | Slippage calibration mismatch | Conservative estimates; order-book-based |
| All phases | Parameter overfitting | Walk-forward validation; simple models first |

---

## Prediction Market Specific Warnings

These issues are unique or amplified in prediction markets:

| Issue | Why It's Different | Mitigation |
|-------|-------------------|------------|
| Extreme liquidity concentration | 63% of short-term markets have zero volume | Filter markets by liquidity before analysis |
| Binary outcome semantics | IS interpretation differs for binary positions | Define methodology for binary markets |
| Event-driven liquidity | Liquidity spikes/crashes around events | Event-aware execution timing |
| Market lifecycle | Markets resolve and close; not ongoing | Include closed markets in analysis |
| Related market fragmentation | YES/NO shares; multiple markets on same event | Aggregate related liquidity |
| Manipulation risk | Low-float markets vulnerable to spoofing | Detect anomalous order patterns |

---

## Pre-Flight Checklist

Before trusting any execution analytics result:

- [ ] Is the market impact model appropriate for this liquidity regime?
- [ ] Are all timestamps strictly "past" relative to decision points?
- [ ] Is slippage modeled conservatively for the actual market?
- [ ] Have parameters been validated out-of-sample?
- [ ] Does the "optimal" strategy beat naive baselines (TWAP) in live testing?
- [ ] Is implementation shortfall methodology documented and consistent?
- [ ] Are liquidity patterns (24/7, event-driven) accounted for?
- [ ] Are closed/resolved markets included in historical analysis?

---

## Summary: Top 5 Pitfalls for This Project

Given the project context (eFX trader, Polymarket, learning-focused, expanding to crypto):

1. **Applying equity market impact models to thin liquidity markets** (Critical - most likely to cause fundamental errors)

2. **Look-ahead bias in execution backtests** (Critical - will invalidate all backtest results)

3. **Slippage calibration mismatch** (Moderate - will make backtest results misleading)

4. **Permanent vs temporary impact mis-attribution** (Moderate - will cause wrong optimization)

5. **VWAP/TWAP failures in thin markets** (Moderate - standard benchmarks may be meaningless)

The overarching theme: **Standard execution analytics tools are designed for liquid equity markets. Every assumption must be questioned for thin liquidity markets like Polymarket.**

Start with empirical measurement. Build intuition from your own data. Use academic models as guides, not ground truth.
