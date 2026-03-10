# Pairs Trading: A Learning Journal
### by Manu Prabakaran

This is a record of how I built a pairs trading strategy from scratch, what I tried, what broke, what I learned from the parts that broke, and what I did next. The notebooks are kept in the order I built them, including the versions that did not work.

---

## What is pairs trading?

Two stocks that are economically linked (same industry, same customers, same cost structure) tend to move together over time. But on any given day, one might run ahead of the other. When the gap between them gets unusually wide, there is a bet to make: buy the cheaper one, short the more expensive one, and wait for the gap to close.

The strategy sounds simple. The hard part is everything else.

---

## [Notebook 01: Finding a Pair](notebooks/01_pair_selection.ipynb)

**The problem:** I needed two stocks to trade. My first instinct was to find stocks that move similarly by looking at correlation, how much their daily returns line up. I ran the correlations across ten sector ETFs and most of them were above 0.80. Everything seemed to move together.

But high correlation turned out to be the wrong thing to look for. Two stocks can move in the same direction every day and still drift apart permanently over years. Correlation measures daily co-movement. What I actually needed to know was whether the *gap* between two prices stays bounded, whether it has a long-run average it keeps returning to.

Reading about this, I came across **cointegration**, a statistical test that measures exactly that. Instead of asking "do these move together day to day?", it asks "is there a stable long-run relationship between these prices?" That is what matters for this strategy.

I ran a cointegration screen across all pairs in the sector ETF universe. The test selected **XLP (consumer staples) and XLU (utilities)**: not the highest-correlation pair, and not an obvious combination. That made it interesting.

---

## [Notebook 02: First Backtest](notebooks/02_backtest.ipynb)

**The problem:** I had a pair. Now I needed to turn that into a trade. I needed to know when the gap was wide enough to enter, and how to measure "wide enough" in a consistent way.

The answer I found was the **z-score**: take the current spread, subtract its historical average, and divide by its standard deviation. A z-score of +2 means the spread is two standard deviations above its average, unusually wide. That is the signal to enter.

I built a backtest around this. Enter when the z-score crosses 2, exit when it returns to 0, close immediately if it hits 3 (a stop-loss). I also included transaction costs; every entry and exit costs 5 basis points, which adds up.

The in-sample Sharpe ratio looked decent. I should have been more suspicious of that.

---

## [Notebook 03: Understanding the Math](notebooks/03_ou_process.ipynb)

**The problem:** I was using z-score thresholds and a rolling window without really understanding why those numbers made sense. The backtest worked, but I could not explain *why* the spread should revert, or how long a typical trade should take.

Reading further, I found the **Ornstein-Uhlenbeck process**, a mathematical model for a value that gets pulled back toward an average over time with random noise added each day. It is the theoretical reason pairs trading works when it works. The spread between two cointegrated stocks behaves like an OU process.

The key number the OU model gives is the **half-life**: how many days it takes, on average, for the spread to close half the distance back to its mean after a trade is opened. A half-life of 20 days means a trade opened at z=2 should reach z=1 within about 20 days. This told me what window to use for the z-score and what holding period to expect.

---

## [Notebook 04: The Kalman Filter](notebooks/04_kalman.ipynb)

**The problem:** The hedge ratio, how many units of XLU to hold against each unit of XLP, was fixed. I had estimated it once from all the historical data. But the relationship between two stocks is not constant. Volatility regimes shift, sector dynamics change, correlations evolve. A hedge ratio set in 2010 might be meaningfully wrong by 2020.

I read about the **Kalman filter**, a technique originally developed for aerospace navigation. Applied to pairs trading, it treats the hedge ratio as a hidden quantity that changes over time and re-estimates it every day using only data available up to that point. Each day it makes a prediction, observes how wrong it was, and adjusts.

I implemented it and found something interesting: the Kalman hedge ratio was consistently lower than the static one throughout the entire period. This makes sense in hindsight. The static OLS estimate used 2024 data to decide the 2010 hedge ratio, which is a form of hindsight the Kalman filter does not have.

The result was surprising. Despite the more theoretically correct approach, the Kalman strategy performed **worse** than the static one. The adaptive hedge ratio absorbed too much of the spread variation, leaving less signal to trade on. A good reminder that more sophisticated is not always better.

---

## [Notebook 05: Walk-Forward Validation](notebooks/05_walk_forward.ipynb)

**The problem:** Everything I had built so far was in-sample. The hedge ratio, OU parameters, and z-score window were all estimated from the same data the backtest was trading on. The model had already seen every price it was evaluated on. That is not a real test.

I read about **walk-forward validation**, the standard way to get honest out-of-sample results. The idea: fit the model on two years of data, trade the next year without touching the model, then slide forward one year and repeat. Report only the test-period performance. The model never trades on data it was fitted with.

I built this and ran it across 12 test windows covering 2012 to 2024.

**The results were mixed but uninspiring.**

| Test Year | Static Sharpe | Result |
|-----------|--------------|--------|
| 2012 | -0.68 | loss |
| 2013 | +0.52 | win |
| 2014 | +0.03 | win |
| 2015 | +1.01 | win |
| 2016 | +0.50 | win |
| 2017 | -0.93 | loss |
| 2018 | +0.05 | win |
| 2019 | -0.61 | loss |
| 2020 | +0.30 | win |
| 2021 | +2.04 | win |
| 2022 | -0.31 | loss |
| 2023 | +2.03 | win |

Eight wins, four losses. OOS Sharpe of 0.25, total return 24.2%, max drawdown -15.9%. Technically positive, but modest: the in-sample Sharpe was 0.59, so nearly 60% of the apparent performance evaporated out-of-sample. **The gap between in-sample and out-of-sample is the key finding here, not the absolute number.**

XLP/XLU had no economic justification. Consumer staples and utilities are not structurally linked: one tracks defensive consumer demand, the other tracks interest rate sensitivity. The cointegration that showed up in the screen was a statistical artifact of the sample period, not a structural relationship. That distinction (statistical cointegration versus economically motivated cointegration) became the foundation of every notebook that followed.

This was the most important finding in the project, and it changed the direction of everything that followed.

---

## [Notebook 06: Starting Over with Better Pairs](notebooks/06_multi_pair.ipynb)

**The problem:** The sector ETF pair was selected by running a statistical screen and taking whatever came out on top. There was no economic reason to believe XLP and XLU should stay linked. Their sectors (consumer staples and utilities) are not structurally connected. The cointegration might have been real during the sample period, or it might have been coincidence.

Reading about how professional pairs traders select pairs, I kept seeing the same point: *the statistical test is a filter, not a reason*. The reason comes first. You identify two companies with a structural economic link (same inputs, same customers, same regulatory environment) and then verify that the link shows up in the data.

I built a universe of eight pairs, each chosen because there is a genuine reason they should trade together:

| Pair | Why they should move together |
|------|-------------------------------|
| NUE / STLD | US mini-mill steel producers: same scrap steel inputs, same flat-rolled output price |
| V / MA | Payment networks: identical business models, no credit risk, same transaction volume drivers |
| KO / PEP | Beverage duopoly: same input costs, same distribution, same consumer base |
| XOM / CVX | US oil majors: same crude exposure, same refining margin |
| GS / MS | Bulge-bracket banks: same trading and advisory revenue drivers |
| HD / LOW | Home improvement duopoly: same customer, same housing cycle |
| TMO / DHR | Life science instruments: same customers, same lab equipment capex cycle |
| AMAT / LRCX | Semiconductor equipment: same fab customers, same technology generation |

I ran walk-forward validation on all eight pairs and ranked them by out-of-sample Sharpe. Testing a universe and reporting the full distribution of results is a fairer picture than showing only the best-performing pair.

**But the first results were obviously wrong.**

The initial leaderboard showed max drawdowns of -189%, -353%, and -374%. A drawdown cannot exceed -100% on a long-short strategy; losing more than the total capital deployed is not possible. Something in the code was broken.

---

## What broke in notebook 06 and how I found it

Getting wrong numbers is not unusual. Getting *impossible* numbers is actually useful, because it means the bug is large enough to find.

I traced through three separate problems, each one hiding behind the previous.

**Bug 1: The z-score was frozen to the wrong reference point.**

The walk-forward code was centering the test-period spread using the training period's mean and standard deviation:

```python
zscore = (test_spread - train_mean) / train_std
```

For sector ETFs trading in a narrow price range, this is approximately fine. For stocks like GS ($80 in 2010, $200 in 2015) and MS ($15 in 2010, $40 in 2015), the spread level shifts dramatically between the training and test window. The z-score became permanently fixed at +5 or -5 for the entire test year, meaning the strategy was stuck in one direction all year and bled out whenever the spread kept drifting.

The fix was to use a rolling z-score warmed up with the training data rather than a frozen reference point. The hedge ratio stays frozen from training; only the mean-centering adapts.

**Bug 2: The hedge ratio was being applied in the wrong units.**

OLS regression estimates the hedge ratio in price space. For GS at $200 and MS at $40, it returns a hedge ratio of roughly 5, meaning one GS share hedges five MS shares in dollar terms. The backtest was then applying that ratio directly to daily returns:

```python
spread_return = r_GS - 5 * r_MS
```

On a normal day when both stocks rise 1%, this formula gives:
`0.01 - 5 × 0.01 = -0.04`

The code was recording a -4% loss. The actual dollar profit and loss was zero: long $200 of GS gained $2, short $200 of MS (five shares at $40) lost $2. The formula was amplifying returns by the price ratio between the two stocks rather than cancelling it.

The fix was to convert the price-space hedge ratio to return space before applying it to daily returns, which for any OLS-estimated pair resolves to approximately `r_t1 - r_t2`.

**Bug 3: The max drawdown was being computed on equity curves that reset to zero.**

The walk-forward produces one backtest per test window, each with an equity curve starting at zero. When concatenated and passed to the performance summary, the max drawdown calculation saw the equity drop from the end of one window (say +5%) back to zero at the start of the next, and recorded that as a -5% drawdown, even though that reset was just a bookkeeping boundary, not a real loss.

The fix was to rebuild the equity column as a single cumulative sum of daily returns across all windows before computing any statistics.

**The final results:**

| Pair | OOS Sharpe | Total Return | Max Drawdown | Win Rate |
|------|-----------|-------------|-------------|---------|
| KO/PEP | 0.50 | 54.3% | -16.0% | 62.1% |
| NUE/STLD | 0.37 | 61.4% | -39.6% | 62.9% |
| V/MA | 0.26 | 25.4% | -17.3% | 65.3% |
| GS/MS | 0.15 | 18.1% | -27.7% | 61.3% |
| HD/LOW | 0.10 | 16.2% | -29.0% | 53.2% |
| AMAT/LRCX | -0.07 | -12.5% | -66.6% | 62.6% |
| XOM/CVX | -0.18 | -25.5% | -54.5% | 60.0% |
| TMO/DHR | -0.23 | -47.0% | -87.8% | 54.3% |

Five of eight pairs produced positive out-of-sample Sharpe ratios across 12 test windows from 2012 to 2024. The top two (KO/PEP and NUE/STLD) have the strongest economic justifications: KO and PEP compete directly for the same consumers with the same cost structure, and NUE and STLD are the only two major US mini-mill steel producers using identical technology.

The three pairs with negative Sharpe are also meaningful. They tell you which economic links were not stable enough to trade on this timescale, which is a real research finding.

---

## [Notebook 07: KO/PEP Deep Dive and Parameter Optimization](notebooks/07_kopep_deep_dive.ipynb)

**The problem:** KO/PEP came out of the universe test with the highest out-of-sample Sharpe. The natural next question was whether the default parameters (entry at z=2.0, exit at z=0.0) were actually right for this pair, or just inherited from earlier notebooks without ever being tested.

I ran a parameter grid across six combinations of entry and exit thresholds, evaluating each one using walk-forward out-of-sample Sharpe only. No in-sample results were used to choose between them.

The defaults were already optimal. The parameter grid makes the cost structure visible: at entry=2.0, switching the exit from 0.0 to 0.5 adds only 11 trades (185 to 196) but drops the Sharpe from 0.50 to 0.32. Exiting early at the mean collects smaller wins on each trade while paying the same transaction cost, which is a losing trade-off at this turnover level. Tighter entry thresholds (1.5) generate nearly twice as many trades as entry=2.0, and the combination of high turnover and marginal signals produces near-zero Sharpe (0.01 to 0.18) across all exit configurations.

**But the parameter search led somewhere more useful than a better number.**

While investigating why exit=0.0 behaved the way it did, I found a bug in the signal generation code: the exit condition `abs(zscore) < exit` when `exit=0.0` evaluates to `abs(z) < 0.0`, which is always False. Positions were never being closed at the mean. They were staying open until the spread crossed the entry threshold on the other side, turning every intended exit into a forced reversal.

A second bug was in the transaction cost calculation. Reversals (going directly from short to long) were being charged 5 basis points when the correct charge is 10, because the code used a boolean flag for "did the position change" rather than counting how many legs were actually traded.

Both were fixed. The signal generator was rewritten as a state machine that tracks position direction correctly. The cost calculation now charges per leg.

**After the fixes, the Sharpe on KO/PEP moved from 0.48 to 0.50.** More importantly, re-running the full eight-pair universe from notebook 06 with the corrected code changed the result from 2 of 8 pairs profitable to 5 of 8. The bugs had been suppressing performance across the entire universe, not just the pair being examined.

KO/PEP returned 54.3% over 12 test years, roughly 3.7% annualised after transaction costs. That is well below a simple index fund. But this strategy is market-neutral: it does not depend on the market going up. The returns come from the *gap* between two stocks, not from broad market movement. The right comparison is not the S&P 500; it is a strategy that can be run alongside other positions, adding return without adding market exposure. Combining it with additional signals or other pairs would be the logical next step.

---

## [Notebook 08: Portfolio Construction](notebooks/08_portfolio.ipynb)

**The problem:** Five of eight pairs were profitable out-of-sample. Each was validated individually. But running five separate strategies raises an obvious question: what happens when you combine them?

I ran the same six-configuration parameter grid from notebook 07 on the four remaining profitable pairs (NUE/STLD, V/MA, GS/MS, and HD/LOW) and evaluated each combination out-of-sample only. Then I combined all five pairs into an equal-weight portfolio.

Equal weighting was deliberate. Optimizing the portfolio weights is another in-sample fitting step, another opportunity to overfit to historical data. If the portfolio works at equal weight, that is the cleaner result.

**The portfolio Sharpe was 0.75, with a max drawdown of -9.6%.**

For comparison, KO/PEP (the best individual pair) had a Sharpe of 0.50 and a max drawdown of -16.0%. The portfolio improved on both numbers, and not by a small margin.

The improvement came from diversification, not from the parameters. These five pairs have genuinely different return drivers: consumer staples, steel input costs, payment network volume, investment banking revenue, and the housing cycle. Their bad years do not coincide. When KO/PEP had a rough year, NUE/STLD might have been profitable. That imperfect correlation across pairs is what compresses the drawdown and lifts the Sharpe.

---

## [Notebook 09: Risk Parity Weighting](notebooks/09_risk_parity.ipynb)

**The problem:** The equal-weight portfolio allocated 20% capital to each pair, but not 20% risk. NUE/STLD is substantially more volatile than KO/PEP. Under equal weighting, the portfolio's daily swings were dominated by whichever pair happened to be most volatile; the risk was unbalanced even though the capital was not.

Risk parity fixes this: each pair receives capital inversely proportional to its training-period strategy volatility, so every pair contributes roughly equal daily fluctuation. Weights are estimated from training data only and frozen for each test period, the same walk-forward discipline applied throughout.

**The results:**

| Method | Sharpe | Total Return | Max Drawdown |
|--------|--------|-------------|-------------|
| Equal Weight | 0.75 | 39.9% | -9.6% |
| Risk Parity | 0.77 | 41.0% | -8.7% |

Risk parity improved on all three metrics. The gain was modest in calm years; equal weight had a small edge in those periods because NUE/STLD's high return justified its full weight. But risk parity outperformed in 2020 (COVID), 2022 (rate hikes), and 2023, the volatile environments where holding a high-vol pair at full weight is most dangerous. In 2022, the worst year in the dataset, equal weight lost 8.2% while risk parity lost only 6.8%.

The weights chart made the mechanism visible: NUE/STLD was downweighted to 9-19% every year, and V/MA was turned off entirely in three years (0.0% weight) where its training-period volatility was too high. No tuning was required; the formula has no free parameters.

**Sharpe 0.77, max drawdown -8.7%** is the strongest result in the project at this stage.

---

## [Notebook 10: Quality Weighting](notebooks/10_quality_weighting.ipynb)

**The question:** Risk parity makes no prediction about future returns; it only balances volatility. What if we allocated more capital to historically better-performing pairs? Two approaches were tested:

- **Sharpe-weighted:** `w ∝ max(0, training Sharpe)` - concentrate in past winners
- **Combined (reward-risk):** `w ∝ max(0, training Sharpe) / training vol` - reward pairs that are both high-quality and calm

**The finding: predicting quality did not improve risk-adjusted returns.**

| Method | Sharpe | Total Return | Max Drawdown |
|--------|--------|-------------|-------------|
| Equal Weight | 0.75 | 39.9% | -9.6% |
| Risk Parity | 0.77 | 41.0% | -8.7% |
| Sharpe-Weighted | 0.75 | 55.4% | -11.2% |
| Combined | 0.71 | 51.4% | -11.2% |

Sharpe-weighted produced the highest raw return (55.4%) by concentrating in historically stronger pairs, but the Sharpe was identical to equal weight (0.75). The extra return came with proportionally extra risk. Combined was the worst method by Sharpe (0.71), worse than equal weight, because doubly penalising volatile pairs created excessive concentration: GS/MS at 31.6% average weight and NUE/STLD at 25.3%, while starving KO/PEP (the best OOS performer) of capital at only 9.5%. That mismatch between training-period quality scores and out-of-sample performance eliminated the diversification that made the portfolio strong.

The reason: past volatility is persistent and can be reliably exploited by risk parity. Past Sharpe is not persistent; a pair that earned a high Sharpe over two training years has no reliable tendency to repeat it in the next test year. This is why risk parity remains the best allocation method for this portfolio.

---

## [Notebook 11: Does the Kalman Filter Help at the Portfolio Level?](notebooks/11_kalman_portfolio.ipynb)

**The question:** Notebooks 03 and 04 tested the Kalman filter on individual pairs. It underperformed static OLS there too. But maybe the degradation averages out across pairs, or maybe the pairs where Kalman genuinely helps (V/MA, GS/MS) carry the portfolio. The question was worth testing directly.

I built all four combinations (equal weight and risk parity, each with static OLS and Kalman hedge ratios) and compared them.

**The results:**

| Method | Sharpe | Total Return | Max Drawdown |
|--------|--------|-------------|-------------|
| EW, Static | 0.75 | 39.9% | -9.6% |
| EW, Kalman | 0.21 | 9.0% | -8.1% |
| RP, Static | 0.77 | 41.0% | -8.7% |
| RP, Kalman | 0.22 | 10.4% | -6.9% |

The Kalman portfolios are not marginally worse; they are near-total failures at the portfolio level.

The per-pair breakdown explains why. KO/PEP is the worst case: Kalman Sharpe -0.12 versus static 0.50, a swing of -0.62. NUE/STLD is similarly hurt: Kalman 0.06 versus static 0.38. These pairs have well-defined, persistent cointegration relationships. The Kalman filter finds nothing real to adapt to, so it adapts to noise: the hedge ratio slowly drifts to fit recent price movements, the spread loses its mean-reverting structure, and most trades enter on false signals that never revert.

The most instructive result is HD/LOW, which actually improved under Kalman (static 0.21, Kalman 0.31). HD/LOW's hedge ratio genuinely drifted over the housing cycle (the relationship between Home Depot and Lowe's shifted as their pricing strategies diverged), so the adaptive estimator had real signal to track. V/MA and GS/MS also showed smaller degradation, for similar reasons: payment network dynamics and investment banking revenue mix do evolve over time.

This is not a bug. It is a real finding about when adaptive estimation is and is not appropriate. **Cointegration and the Kalman filter are in tension.** Cointegration is a statement that a long-run equilibrium exists and the spread is stationary. The Kalman filter is a method for tracking a relationship that changes over time. For the pairs where the relationship is genuinely stable (KO/PEP, NUE/STLD), applying an adaptive estimator introduces the instability it was designed to handle. For the pairs where the relationship genuinely evolves (HD/LOW, V/MA, GS/MS), the Kalman filter earns its keep. Static OLS wins at the portfolio level because the stable pairs dominate.

---

## [Notebook 12: VIX Regime Detection](notebooks/12_regime_detection.ipynb)

**The question:** Notebook 09 showed that risk parity outperformed equal weight specifically in turbulent years: 2020, 2022, 2023. That pattern raised an obvious hypothesis. If we could detect turbulent conditions in advance, we could switch to risk parity before they arrive and back to equal weight in calm periods, getting the best of both methods.

The VIX index (implied volatility from S&P 500 options, a direct measure of market fear available in real time) is the most natural signal for this. The test: when VIX is elevated (above 20, the widely-used convention), use risk parity; when VIX is calm (below 20), use equal weight. The VIX is applied with a one-day lag so today's allocation uses only yesterday's information.

**The results:**

| Method | Sharpe | Total Return | Max Drawdown |
|--------|--------|-------------|-------------|
| Equal Weight | 0.75 | 39.9% | -9.6% |
| Risk Parity | 0.77 | 41.0% | -8.7% |
| Binary Regime (VIX > 20) | 0.73 | 40.4% | -9.0% |
| Soft Blend (VIX 18-22) | 0.74 | 40.8% | -9.0% |

Both regime variants underperformed pure risk parity. I also tested a soft-threshold version that replaces the binary switch with a continuous linear blend (`w = clip((VIX - 18) / (22 - 18), 0, 1)`) to reduce the whipsaw from abrupt full-portfolio flips every time VIX crosses 20. The continuous blend was marginally better than binary switching, but still did not beat risk parity.

The threshold sensitivity table (VIX = 15 through 30) confirmed the result was not specific to the chosen threshold. The regime signal underperformed across the full range, not just at VIX = 20.

**The reason is structural.** Risk parity already handles volatility by downweighting the most volatile pairs when markets are stressed. The VIX-based regime switch is trying to solve a problem that risk parity's own weighting has already partially solved. Adding an external regime signal on top of that adds switching overhead without adding new information.

The notebook does establish a modular `blend_weight()` architecture: a function that maps any signal to a portfolio weight, with a clean interface that can be swapped for a Hidden Markov Model state probability or any other estimator. For this portfolio, VIX happened not to be the right signal. The architecture is the reusable result.

---

## [Notebook 13: Diagnosing the Gaps](notebooks/13_diagnosis.ipynb)

This notebook is different from the others. Every chapter before this one started with a known gap and a known tool. This one started with a pattern I noticed and did not yet have a name for.

After notebooks 11 and 12 both failed to improve on risk parity (Kalman filter and VIX regime detection), I could see that something was wrong with my approach to improving the strategy, but I did not have a clean framework for naming it. Both experiments had failed for the same structural reason: each was adding a layer that solved a problem an existing layer had already solved. The Kalman filter was trying to improve the hedge ratio, but cointegration already guarantees the relationship is stable. VIX regime detection was trying to make the portfolio defensive when volatility is high, but risk parity already does exactly that at the pair level.

Finding the name for that pattern required reading. It comes up in quantitative finance literature on strategy construction as **layer redundancy**: adding complexity to a system when a simpler layer has already addressed the same problem. The same reading introduced three concepts I had not previously known about: cointegration health monitoring (using training-period statistical test quality as a gate before trading a window), mean-reversion momentum filtering (using the spread's recent direction at entry as a signal quality check), and cross-pair correlation monitoring (tracking whether the five pairs stay independent of each other or converge during stress events).

Rather than immediately implementing fixes, this notebook ran diagnostics on the existing backtest data to test whether each gap was actually showing up in the numbers. The reasoning: if a hypothesized gap does not manifest in the data, implementing a fix addresses a theoretical problem that is not causing real losses.

The three analyses: (1) correlating training-period Engle-Granger p-values and OU half-lives against test-period Sharpe ratios (does weak cointegration during training reliably predict bad out-of-sample results?), (2) comparing 10-day forward PnL for trades entered while the spread is still building versus trades entered when it has already started reverting (does entry timing matter?), (3) tracking 60-day rolling pairwise correlations between all five pairs (do the pairs lose their independence in the portfolio's worst years?).

All three came back as weak signals. The portfolio's construction quality (economically selected pairs, one per industry, five genuinely independent return sources) meant the standard literature improvements did not apply. Cointegration held reliably, entry timing did not matter, and the pairs stayed structurally independent even in stressed markets. The only remaining path was outside the existing portfolio: more genuinely independent pairs.

---

## [Notebook 14: Universe Expansion](notebooks/14_universe_expansion.ipynb)

**The problem:** Notebook 13 ran diagnostics on three hypothesised gaps and found all three were weak signals. The portfolio's construction quality meant the standard optimisation levers were already saturated. The only remaining structural path to a higher Sharpe was more pairs.

But "more pairs" hides a hard problem.

If you test every pair in the S&P 500 at a raw p < 0.05 cointegration threshold, you are testing roughly 124,750 combinations. At that threshold, you expect around 6,000 false discoveries: pairs that look cointegrated historically but have no structural reason to stay that way. The result is not a better portfolio. It is a portfolio full of statistical coincidences that will break the moment conditions shift.

The solution has two parts: curate the candidate universe on economic grounds first, then apply the **Benjamini-Hochberg procedure** to correct for the multiple testing problem. Rather than controlling the probability of any single false positive (which Bonferroni does, aggressively), BH controls the *fraction* of accepted pairs that are false. At FDR ≤ 5%, no more than 5% of accepted pairs should be spurious. The procedure itself is five lines of arithmetic.

I chose seven industries not already covered by the current five pairs (airlines, integrated energy, refining, large-cap banks, insurance, semiconductors, telecom) and wrote one structural rationale for each before looking at any data. That produced 29 candidate pairs.

**The funnel:**

| Stage | Count | Notes |
|-------|-------|-------|
| Economic curation | 29 pairs | 7 industries, one structural rationale per industry |
| Raw p < 0.05 | 4 pairs | Expected ~1-2 spurious at this stage |
| BH-corrected (FDR ≤ 5%) | 1 pair | TRV/CB (Insurance), p = 0.001 |
| OU half-life 5-126 days | 1 pair | half-life = 52.7 days |
| Strategy correlation < 0.30 | 1 pair | mean correlation with existing pairs = 0.015 |
| Walk-forward validated | 1 pair | OOS Sharpe = 0.67, 10/12 windows profitable |

One pair, **Travelers/Chubb (TRV/CB)**, survived all four filters. Its p-value is an order of magnitude below the next candidate, its half-life sits near the middle of the tradeable range, and its strategy returns are essentially uncorrelated with every existing pair. Insurance spreads are driven by underwriting cycles and catastrophe pricing, neither of which has any relationship to steel input costs, beverage market share, payment network volume, investment banking revenue, or home improvement retail traffic.

**The result:**

| Method | Sharpe | Total Return | Max Drawdown |
|--------|--------|-------------|-------------|
| RP, 5 pairs (baseline) | 0.77 | 41.0% | -8.7% |
| RP, 6 pairs (+ TRV/CB) | 0.96 | 43.5% | -5.8% |

The Sharpe improvement and the drawdown reduction are both traceable to mechanism. TRV/CB adds almost nothing in calm years but helps materially in the two worst years in the dataset: 2020 (+2.4pp) and 2022 (+2.8pp, cutting the portfolio's annual loss from -6.8% to -4.0%).

The scaling chart (portfolio Sharpe plotted as pairs are added one at a time in order of individual OOS Sharpe) produced a clean concave curve: 0.70 → 0.80 → 0.92 → 0.97 → 0.99 → 0.96. The concave shape is the signature of genuine diversification. The slight dip at six pairs also raised a question: should we drop HD/LOW and keep the "greedy-optimal" five-pair combination at SR=0.99?

The answer is no, and the reason matters. Selecting which pairs to include based on the combined Sharpe observed across 14 years of data is in-sample portfolio optimisation. HD/LOW is not noise; it has a real economic story and a positive individual out-of-sample Sharpe. The 6-pair portfolio at 0.96 is the honest result. SR=0.99 is what the data would have told us to do in hindsight, which is the kind of reasoning this project has been working to avoid since notebook 06.

---

## [Notebook 15: Robustness Analysis](notebooks/15_robustness.ipynb)

**The problem:** SR=0.96 is the result under one specific set of assumptions: 5 basis points per leg, the particular (entry_z, exit_z) parameters that scored best in the grid search, and the exact pair composition chosen through the universe expansion funnel. Before trusting that number enough to trade on, it is worth asking how sensitive it is to each of those assumptions.

Three stress tests were run.

**Transaction cost sensitivity.** The 5 bps assumption was already baked into every backtest; SR=0.96 is a post-cost number, not gross. The cost sweep varied that assumption from 0 to 30 bps and rebuilt the portfolio at each level without re-running walk-forward. At zero cost SR=1.11, confirming that 5 bps accounts for 0.15 Sharpe points of friction. The decline is nearly linear: each additional 5 bps removes roughly 0.17 Sharpe points.

| cost (bps) | Sharpe | Total Return |
|---|---|---|
| 0 | 1.11 | ~52% |
| 5 *(current)* | 0.96 | ~43% |
| 10 | 0.79 | ~36% |
| 15 | 0.60 | ~27% |
| 20 | 0.41 | ~19% |

The SR=0.5 viability threshold is crossed around 17 bps, a margin of 12 bps above the current assumption. A retail broker charging 10 bps all-in would still produce SR≈0.79.

**Parameter stability.** The grid search selected (entry_z, exit_z) per pair using OOS Sharpe on this exact dataset, so confirming those choices are "best" on the same data is circular. What the heatmap does show is the shape of the parameter landscape: entry=2.0 and entry=2.5 produce nearly identical portfolio Sharpes (0.68-0.70), while entry=1.5 drops to 0.50. The strategy is not sitting on a fragile peak; it is on a plateau where similar parameter choices produce similar results. The exit threshold (0.0 vs 0.5) makes almost no difference.

**Trade frequency.** Five of the six pairs trade 7-14 times per year with average holding periods of 6-8.5 days. Daily monitoring is sufficient; no pair requires intraday attention. NUE/STLD is the outlier at 31 trades per year and 3.5-day holds, almost three times more active than any other pair. NUE/STLD is the one pair where the 5 bps cost assumption deserves extra scrutiny, since higher turnover amplifies any difference between assumed and actual execution cost.

**The verdict.** SR=0.96 is likely to hold in practice. It already accounts for realistic transaction costs, it sits on a parameter plateau rather than a fragile peak, and it survives a doubling of cost assumptions with SR still above 0.79. Even at 15 bps (more than most retail brokers charge for liquid large-caps) the portfolio remains profitable at SR=0.60. The strategy is not dependent on things going exactly right.

---

## [Notebook 16: Live Signal Generation](notebooks/16_live_signals.ipynb)

**The problem:** All prior notebooks worked on historical data. The strategy had been validated, stress-tested, and tuned, but it could not answer the only question that matters for actual trading: what should I be doing right now?

This notebook is the bridge. For each of the six pairs, it fits the hedge ratio and OU half-life on the trailing two years of real price data, computes today's z-score using the same rolling window logic as the walk-forward backtests, and produces a signal. There is no test period; we are at the edge of the data, and today's z-score is today's signal.

**What the signal provides:**
- **Direction:** LONG, SHORT, or FLAT per pair
- **Legs:** the exact two tickers to buy and sell short (e.g. SHORT KO/PEP = buy PEP, sell KO short)
- **Sizing:** two options depending on the consuming bot's philosophy:
  - *Fixed weights*: RP weights across all 6 pairs; flat pairs stay in cash; matches the backtest (SR=0.96 applies)
  - *Normalized weights*: RP weights across active pairs only; always fully deployed; larger positions than the backtest implied (SR=0.96 does not apply)

The output is a `signals_output.json` file written every time the notebook runs. A trading bot reads this file without needing to understand any of the analysis. The schema includes `buy_ticker`, `sell_ticker`, `signal`, `zscore`, `confidence` (relative RP weight), `days_in_position`, both dollar exposure estimates, and two health fields added after the first live run.

**Pair health monitoring.** Running the strategy live immediately raised two questions: what if the cointegration relationship has weakened since the strategy was built? And what if the hedge ratio has gone negative, meaning the relationship has inverted? Both are now checked on every run. `pvalue` reports the current cointegration p-value on the trailing training window. `health` is a flag: `OK`, `WARN_PVALUE` (p ≥ 0.05, no longer statistically cointegrated), `WARN_HEDGE_RATIO` (negative hedge ratio, relationship has inverted), or `WARN_BOTH`. Health warnings print to the cron log and bot output so issues are visible without reading the JSON. KO/PEP triggered `WARN_HEDGE_RATIO` on the first live run (hedge ratio = -0.10), flagging that its relationship has inverted in the current two-year window despite the position being open.

**As of 2026-03-06:** Two pairs active, both SHORT (KO/PEP at 18 days, z=+0.82 and HD/LOW at 5 days, z=+0.34). Four pairs flat. Under fixed weights, 52% of the portfolio is deployed; under normalized weights, 100%.

---

## [Notebook 17: Signal Infrastructure](notebooks/17_signal_infrastructure.ipynb)

**The problem:** Notebook 16 generates signals correctly but requires opening Jupyter and running cells manually. A live trading system needs signals on a schedule, generated automatically every morning before the market opens, without any human involvement.

Two pieces complete the operational pipeline:

**`run_signals.py`** is a standalone script in the project root. It imports the same modules as notebook 16 (`strategy.pairs_config`, `strategy.live`, `data.loader`) and produces the same output without any charts or tables. Running `python run_signals.py` fetches today's prices, generates signals for all validated pairs, writes `signals_output.json` (overwritten each run) and appends one line to `signals_history.jsonl` (append-only audit trail). Adding a new pair requires no changes to this script; it reads from `strategy/pairs_config.py` automatically.

**`signals_history.jsonl`** is the day-by-day record of every signal the strategy produced in live operation. Over time it enables two things: debugging (tracing exactly why a position was entered on a specific date) and evaluating whether live signals match what the backtest predicted.

**Scheduling:** add a single cron line to run the script at 7am Monday-Friday. Bots that are Python processes can trigger it via `subprocess.run()` and immediately read `signals_output.json`. The notebook includes ready-to-use code for both approaches.

The notebook also includes history analysis cells (signal activity per pair, percentage of days active, and a signal change log) that grow more informative as runs accumulate.

**First run (2026-03-06):** Script executed successfully. `signals_history.jsonl` created with first entry. Two active positions recorded (KO/PEP SHORT and HD/LOW SHORT) matching notebook 16's output exactly.

---

## [Notebook 18: Universe Expansion Round 2](notebooks/18_universe_expansion_2.ipynb)

**The problem:** Notebook 14 found one new pair (TRV/CB) from seven new industries. The US large-cap universe still had untouched sectors. With the signal infrastructure operational and the baseline solid at SR=0.96, the logical question was whether a second expansion pass across a different set of industries could find another genuine pair.

Seven new industries were selected, none overlapping with the first expansion round. The same pipeline was applied: economic curation first, statistical testing second, BH correction to control the false discovery rate at FDR ≤ 5%.

**The result: zero survivors.**

Every candidate pair either failed the raw cointegration screen or was eliminated by BH correction. The false discovery rate procedure is not conservative; at 29 candidates in round 1, it passed TRV/CB easily. At zero raw survivors passing p < 0.05, there was nothing for BH to accept.

The US large-cap equity universe is exhausted at FDR ≤ 5%. The pairs that work are those where two companies share input costs, customers, or business models so directly that the relationship is structural rather than coincidental. That set is finite, and the six validated pairs are the natural members of it within S&P 500 large-caps. Further expansion would require a different asset class: futures, ETF pairs, or ADR pairs.

---

## [Notebook 19: Pre-2010 Stress Test (GFC Window)](notebooks/19_pre2010_stress_test.ipynb)

**The problem:** Every backtest in this project starts in 2010. That boundary was practical (Visa's March 2008 IPO meant V/MA had almost no data before the crisis) but it meant the strategy had never been tested through the 2008-2009 financial crisis, the most severe equity stress event in decades. A strategy that only works in post-crisis conditions is not the same as one that works across market regimes.

Five of the six pairs have sufficient pre-crisis data: KO/PEP, NUE/STLD, GS/MS, HD/LOW, and TRV/CB. Three questions were tested: did the spreads visually widen and revert (or break permanently)? Did cointegration hold across four regime windows (pre-crisis, crisis, recovery, post-crisis)? And did mean-reversion speed change materially?

**The spread charts are the clearest result.** All five pairs widened under crisis stress and then reverted; none show a permanent level shift. KO/PEP spiked to -4σ in early 2008 and came back. GS/MS reached +4σ at the crisis onset and mean-reverted over roughly a year. The economic relationships survived the GFC intact.

**The hedge ratio table revealed the more nuanced story.** GS/MS compressed 34% during the crisis (4.41 to 2.91), NUE/STLD was nearly halved (2.59 to 1.17), and HD/LOW doubled over the full period. These are genuine structural changes, exactly the kind of parameter instability the walk-forward framework was designed to handle by refitting the hedge ratio in every training window. KO/PEP and TRV/CB were the most stable, with one exception: KO/PEP's pre-crisis (0.321), crisis (0.343), and post-crisis (0.297) ratios are tightly grouped, but the recovery window shows 0.673, a clear outlier that the walk-forward framework would handle by refitting in that training period. TRV/CB's pre- and post-crisis ratios are identical at 0.808, with the crisis (0.582) and recovery (0.557) windows lower but converging back.

TRV/CB was the standout in the OU half-life analysis: 8-49 days across all four regimes, and faster during the crisis (8d) than any other period. The insurance relationship tightened, not loosened, under stress. GS/MS and HD/LOW raised a concern: post-crisis half-lives of 158d and 111d respectively, near the practical limit for a 1-year test window.

The strategy's foundations are sound across regimes. Spreads widened and closed; no pair showed a structural break. The parameter instability that did occur is precisely the problem the walk-forward refitting was designed to solve.

---

## What I learned

**In-sample performance tells you almost nothing.** A backtest that fits and trades on the same data will look good almost every time. Walk-forward validation is the only honest test.

**Statistical pairs need economic justification.** Finding two stocks that happen to be cointegrated over some historical window is not enough. The cointegration needs a reason, or it will not survive out-of-sample.

**More sophisticated is not automatically better.** The Kalman filter is theoretically more correct than a fixed hedge ratio. It still underperformed. Tools are only as good as the setting they are applied to.

**Reporting what did not work is part of the research.** Notebook 05 showed that XLP/XLU had a modest positive OOS Sharpe (0.25) but with a 60% degradation from in-sample, and no economic reason to hold up going forward. That finding drove the decision to rebuild around economically motivated pairs. A mediocre result on a unjustified pair is just as useful as a clean failure.

**Impossible results are a gift.** When notebook 06 returned drawdowns of -353%, the response was not to hide the numbers or reframe them. It was to treat them as a signal that something in the implementation was wrong. Tracing through the code led to three real bugs, each one subtle enough that it would have been easy to miss without the impossible number flagging it. Correct results that are merely bad are much harder to debug than results that cannot possibly be right.

**Failing to improve a number is still a result.** Notebook 07 set out to beat a Sharpe of 0.48 through parameter optimization and could not. But the investigation uncovered two implementation bugs that, once fixed, improved the result anyway, and revealed that the problem was never the parameters.

**Risk parity requires no optimization but still improves results.** Weighting pairs by inverse volatility (one formula, zero free parameters) improved Sharpe from 0.75 to 0.77 and reduced max drawdown from -9.6% to -8.7%. The improvement came from structural risk balancing, not data fitting.

**Volatility is persistent. Returns are not.** Sharpe-weighting (concentrating in past winners) produced the same risk-adjusted return as equal weighting. Combined weighting produced worse results. Past vol reliably predicts future vol. Past Sharpe does not reliably predict future Sharpe. The exploitable regularity is in risk, not return.

**Diversification works better than optimization.** A 5-pair equal-weight portfolio achieved a Sharpe of 0.75 (50% higher than the best individual pair) with a max drawdown of -9.6% versus -16.0%. No weights were fitted. The improvement came entirely from combining pairs with different losing years. Getting diversification right is worth more than squeezing the last decimal out of a single strategy's parameters.

**Cointegration and Kalman filtering are in tension.** Cointegration asserts a stable long-run relationship. The Kalman filter is designed for relationships that change over time. Applying an adaptive estimator to a stable pair means it adapts to noise: the hedge ratio drifts, the spread loses its mean-reverting structure, and the strategy trades on signals that do not revert. For pairs chosen because of genuine long-run economic equilibria, static OLS is not just simpler. It is the right tool.

**Adding an external regime signal on top of a structural solution rarely helps.** Risk parity already adapts to volatility by downweighting volatile pairs. A VIX-based overlay that tries to shift the whole portfolio to risk parity when markets are fearful is solving a problem the weighting scheme has already handled. The switching overhead costs more than the signal earns. This is the regime detection equivalent of the Kalman finding: before adding a layer of complexity, check whether the layer below has already addressed the same problem.

**Diagnose before you fix.** When two consecutive experiments fail, the instinct is to try a third. The more useful response is to ask why both failed and whether the same cause is producing both failures. Both Kalman and VIX failed because of layer redundancy, a pattern that is only visible once you name it. Running diagnostics on the existing data before implementing new solutions separates real gaps from theoretical ones. A fix that addresses a gap not present in the data will produce the same result as the experiments that preceded it.

**Not knowing a tool is not the same as not needing it.** Cointegration health monitoring, mean-reversion momentum filtering, and cross-pair correlation tracking are all standard concepts in the quantitative finance literature. None of them appeared in earlier notebooks because the earlier problems did not require them. The right response when encountering a gap whose solution is unknown is to read before building, not to invent something from scratch and discover later that the standard tool already exists and has known properties.

**Multiple testing correction is not optional when screening many candidates.** Testing 29 pairs at a raw p < 0.05 threshold expected roughly 1-2 false discoveries. The Benjamini-Hochberg procedure reduced 4 raw survivors to 1 by controlling the false discovery rate rather than banning all discoveries. Without it, the portfolio would have taken on spurious pairs that looked good historically for no structural reason.

**The scaling curve is illustrative, not a selection tool.** Plotting portfolio Sharpe as pairs are added in order of individual OOS Sharpe produces a useful visualisation of the diversification effect. But using that curve to decide which pairs to remove is in-sample portfolio optimisation: selecting composition based on observed outcomes across the full historical period. A result reached by inspecting the data and then adjusting the portfolio to improve it is not out-of-sample, even if the underlying walk-forward steps were.

**A strategy scales through the addition of genuinely independent signals, not through further optimisation of existing ones.** The jump from SR=0.77 to SR=0.96 came from adding one pair with mean strategy correlation of 0.015 to the existing five. That is the clearest demonstration in the project of why the architecture works: not because any individual signal is exceptional, but because six genuinely independent mean-reversion trades compound without cancelling each other out.

**A universe is finite.** Seven new industries in the second expansion round produced zero survivors after BH correction. The US large-cap equity universe is exhausted at FDR ≤ 5%. The pairs that work are those with structural economic links direct enough to remain cointegrated across regimes, and within S&P 500 large-caps, there are only so many of them.

**Spread charts are more informative than short-window p-values for regime stress tests.** The Engle-Granger test consistently fails on 2-year windows; there is not enough data for it to reliably reject the null. OU half-life stability and visual spread behaviour are the right tools for asking whether a relationship survived a crisis. All five pairs' spreads widened and closed through the GFC, confirming the economic relationships held even when the test lacked the power to confirm it statistically.

---

## Project structure

```
data/
  loader.py           fetch and cache price data from yfinance

pairs/
  selection.py        cointegration testing, pair screening, spread computation
  metrics.py          OU process fitting, half-life estimation

signals/
  zscore.py           rolling z-score computation, entry/exit signal generation
  kalman.py           Kalman filter, dynamic hedge ratio estimation

strategy/
  backtest.py         vectorized backtest engine with transaction costs
  walk_forward.py     rolling train/test validation framework
  portfolio.py        equal-weight, risk-parity, and weighted portfolio builders
  live.py             live signal generation and position sizing
  pairs_config.py     single source of truth for validated pairs and parameters

analysis/
  performance.py      Sharpe ratio, drawdown, win rate, trade statistics

notebooks/
  01_pair_selection.ipynb
  02_backtest.ipynb
  03_ou_process.ipynb
  04_kalman.ipynb
  05_walk_forward.ipynb
  06_multi_pair.ipynb
  07_kopep_deep_dive.ipynb
  08_portfolio.ipynb
  09_risk_parity.ipynb
  10_quality_weighting.ipynb
  11_kalman_portfolio.ipynb
  12_regime_detection.ipynb
  13_diagnosis.ipynb
  14_universe_expansion.ipynb
  15_robustness.ipynb
  16_live_signals.ipynb
  17_signal_infrastructure.ipynb
  18_universe_expansion_2.ipynb
  19_pre2010_stress_test.ipynb

run_signals.py        run daily to generate fresh signals
signals_output.json   latest signals (overwritten each run)
signals_history.jsonl day-by-day signal log (append-only)
```

## Setup

```bash
pip install pandas numpy yfinance statsmodels matplotlib seaborn pyarrow
```

Clone the repo and run the notebooks in order. Each one builds on the previous. Data is cached locally after the first download so subsequent runs are fast.
