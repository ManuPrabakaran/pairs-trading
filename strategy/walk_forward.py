import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from pairs.selection import test_cointegration, compute_spread
from pairs.metrics import fit_ou
from signals.zscore import compute_zscore, generate_signals
from signals.kalman import run_kalman, generate_kalman_signals
from strategy.backtest import run_backtest, BacktestConfig
from analysis.performance import summarize


def run_walk_forward(
    prices: pd.DataFrame,
    t1: str,
    t2: str,
    train_years: int = 2,
    test_years: int = 1,
    cost_bps: float = 5.0,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: float = 3.0,
) -> dict:
    """
    Walk-forward validation for a pairs trading strategy.

    Why walk-forward?
    -----------------
    Any backtest that estimates its parameters (hedge ratio, z-score window)
    from the same data it trades on is in-sample. The results are optimistic
    because the model has already seen the data. Walk-forward validation
    prevents this by:

      1. Fitting all parameters on a TRAINING period only
      2. Running the strategy on a TEST period the model has never seen
      3. Rolling the window forward and repeating
      4. Reporting ONLY the test period results

    The out-of-sample Sharpe from walk-forward is the number you can trust.

    How it works
    ------------
    Given 5 years of data, train_years=2, test_years=1:

      Window 1: Train 2020-2021, Test 2022
      Window 2: Train 2021-2022, Test 2023
      Window 3: Train 2022-2023, Test 2024

    For each window, two strategies are evaluated:

    Static:
      - Estimate hedge ratio from training data only (frozen for test period)
      - Estimate OU half-life from training spread (sets z-score window)
      - Run backtest on test period with frozen parameters

    Kalman:
      - Run the Kalman filter on training data to warm up the state estimate
      - Continue running on test data from the warm state (no re-fitting)
      - Record PnL only from the test period
      - The Kalman hedge ratio continues adapting during test (this is correct —
        it only uses information available up to each day)

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history. Must cover train + all test windows.
    t1, t2 : str
        Ticker symbols.
    train_years : int
        Length of each training window in years.
    test_years : int
        Length of each test window in years.
    cost_bps : float
        One-way transaction cost in basis points.
    entry_z, exit_z, stop_z : float
        Signal thresholds, same meaning as in generate_signals().

    Returns
    -------
    dict with keys:
        windows         List of dicts: {train_start, train_end, test_start, test_end}
        static_results  List of backtest DataFrames, one per test window (static)
        kalman_results  List of backtest DataFrames, one per test window (Kalman)
        static_equity   pd.Series: concatenated equity across all test windows
        kalman_equity   pd.Series: concatenated equity across all test windows
        static_stats    dict: summarize() on concatenated static results
        kalman_stats    dict: summarize() on concatenated Kalman results
        window_stats    list of dicts: per-window stats for both methods
    """
    dates = prices.index
    start_date = dates[0].to_pydatetime()
    end_date = dates[-1].to_pydatetime()

    # --- Build rolling windows ---
    windows = []
    train_start = start_date
    while True:
        train_end = train_start + relativedelta(years=train_years)
        test_end  = train_end  + relativedelta(years=test_years)
        if test_end > end_date:
            break
        windows.append({
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  train_end + pd.offsets.BDay(1),
            "test_end":    test_end,
        })
        train_start = train_start + relativedelta(years=test_years)

    if not windows:
        raise ValueError(
            f"Not enough data for even one window. "
            f"Need at least {train_years + test_years} years, "
            f"have {(end_date - start_date).days / 365:.1f}."
        )

    static_results  = []
    kalman_results  = []
    window_stats    = []

    for w in windows:
        # Slice train and test price data
        train_prices = prices.loc[w["train_start"]:w["train_end"]]
        test_prices  = prices.loc[w["test_start"]:w["test_end"]]

        # ----------------------------------------------------------------
        # STATIC METHOD
        # Fit everything on training data, freeze for test period.
        # ----------------------------------------------------------------
        coint = test_cointegration(train_prices[t1], train_prices[t2])
        hedge_ratio_static = coint["hedge_ratio"]

        train_spread = compute_spread(train_prices, t1, t2, hedge_ratio_static)
        ou = fit_ou(train_spread)
        # Use half-life as z-score window; fall back to 30 days if OU fit is invalid
        window_size = int(round(ou["half_life"])) if ou["is_valid"] else 30

        # Compute z-score using a rolling window warmed up on training data.
        # Prepending the training spread gives the rolling window window_size days of
        # history before the test period starts, avoiding a cold-start bias.
        # The hedge ratio and window_size are frozen from training — no lookahead.
        # The rolling mean/std adapts to the current spread level in the test period,
        # which prevents catastrophic drift when spread levels shift between windows.
        test_spread_static = compute_spread(test_prices, t1, t2, hedge_ratio_static)
        combined_spread = pd.concat([train_spread, test_spread_static])
        combined_spread = combined_spread[~combined_spread.index.duplicated(keep="last")]
        zscore_combined = compute_zscore(combined_spread, window=window_size)
        zscore_static = zscore_combined.loc[test_spread_static.index[0]:].rename("zscore")

        signals_static = generate_signals(
            zscore_static, entry=entry_z, exit=exit_z, stop=stop_z
        )
        config_static = BacktestConfig(
            t1=t1, t2=t2, hedge_ratio=hedge_ratio_static, cost_bps=cost_bps
        )
        result_static = run_backtest(test_prices, signals_static, config=config_static)
        static_results.append(result_static)

        # ----------------------------------------------------------------
        # KALMAN METHOD
        # Warm up on training data, continue into test, keep only test PnL.
        # ----------------------------------------------------------------
        # Run filter on the full train+test period so the state is continuous.
        # We will only record PnL from the test portion.
        combined_prices = prices.loc[w["train_start"]:w["test_end"]]
        kf_full = run_kalman(combined_prices, t1, t2)

        # Extract the test-period portion of the Kalman output
        kf_test = kf_full.loc[w["test_start"]:w["test_end"]]

        # Z-score the Kalman spread using a rolling window warmed up on training data.
        # Same logic as the static fix: prepend training spread so the rolling window
        # has warm-up history and adapts to the current spread level in the test period.
        kf_train = kf_full.loc[w["train_start"]:w["train_end"]]
        kalman_combined = pd.concat([kf_train["spread"], kf_test["spread"]])
        kalman_combined = kalman_combined[~kalman_combined.index.duplicated(keep="last")]
        zscore_kalman_full = compute_zscore(kalman_combined, window=window_size)
        zscore_kalman = zscore_kalman_full.loc[kf_test.index[0]:].rename("zscore")

        signals_kalman = generate_signals(
            zscore_kalman, entry=entry_z, exit=exit_z, stop=stop_z
        )
        config_kalman = BacktestConfig(
            t1=t1, t2=t2,
            hedge_ratio=kf_test["hedge_ratio"],
            cost_bps=cost_bps
        )
        result_kalman = run_backtest(test_prices, signals_kalman, config=config_kalman)
        kalman_results.append(result_kalman)

        # Per-window stats
        window_stats.append({
            "test_period": f"{w['test_start'].strftime('%Y-%m')} to {w['test_end'].strftime('%Y-%m')}",
            "static_sharpe":   summarize(result_static)["sharpe_ratio"],
            "static_return":   summarize(result_static)["total_return"],
            "kalman_sharpe":   summarize(result_kalman)["sharpe_ratio"],
            "kalman_return":   summarize(result_kalman)["total_return"],
        })

    # --- Concatenate test-period equity curves ---
    # Reset each window's equity to start from where the previous one ended
    # so the curve is continuous across windows.
    def concat_equity(results_list):
        pieces = []
        offset = 0.0
        for r in results_list:
            eq = r["net_pnl"].cumsum() + offset
            pieces.append(eq)
            offset = eq.iloc[-1]
        return pd.concat(pieces)

    static_equity = concat_equity(static_results)
    kalman_equity = concat_equity(kalman_results)

    # Combined results DataFrames for overall summarize().
    # The per-window equity columns each start at 0, which would inflate max drawdown
    # and misreport total return. Replace equity with a globally continuous cumsum of
    # net_pnl so that summarize() sees the true cross-window equity curve.
    combined_static = pd.concat(static_results).copy()
    combined_static["equity"] = combined_static["net_pnl"].cumsum()
    combined_kalman = pd.concat(kalman_results).copy()
    combined_kalman["equity"] = combined_kalman["net_pnl"].cumsum()

    return {
        "windows":        windows,
        "static_results": static_results,
        "kalman_results": kalman_results,
        "static_equity":  static_equity,
        "kalman_equity":  kalman_equity,
        "static_stats":   summarize(combined_static),
        "kalman_stats":   summarize(combined_kalman),
        "window_stats":   window_stats,
    }


def run_parameter_grid(
    prices: pd.DataFrame,
    t1: str,
    t2: str,
    entry_zs: list,
    exit_zs: list,
    train_years: int = 2,
    test_years: int = 1,
    cost_bps: float = 5.0,
) -> dict:
    """
    Run walk-forward validation for every (entry_z, exit_z) combination.

    stop_z is fixed at entry_z + 1.0 for all configs, keeping the risk/reward
    structure consistent across parameter combinations.

    All evaluation is out-of-sample (walk-forward). No in-sample data is used
    to select between configurations. However, choosing the best result from
    multiple OOS configurations still introduces mild selection bias — the winner
    may be partly lucky. Report the full grid when publishing results.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history for t1 and t2.
    t1, t2 : str
        Ticker symbols.
    entry_zs : list of float
        Entry z-score thresholds to test (e.g. [1.5, 2.0, 2.5]).
    exit_zs : list of float
        Exit z-score thresholds to test (e.g. [0.0, 0.5]).
    train_years, test_years : int
        Walk-forward window lengths.
    cost_bps : float
        One-way transaction cost in basis points.

    Returns
    -------
    dict keyed by (entry_z, exit_z) → wf result dict from run_walk_forward.
    """
    results = {}
    for entry_z in entry_zs:
        for exit_z in exit_zs:
            wf = run_walk_forward(
                prices, t1, t2,
                train_years=train_years,
                test_years=test_years,
                cost_bps=cost_bps,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=entry_z + 1.0,
            )
            results[(entry_z, exit_z)] = wf
    return results
