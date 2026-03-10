import numpy as np
import pandas as pd

from signals.zscore import compute_zscore, generate_signals


def run_kalman(
    prices: pd.DataFrame,
    t1: str,
    t2: str,
    delta: float = 1e-4,
) -> pd.DataFrame:
    """
    Kalman filter for pairs trading with a time-varying hedge ratio.

    The Problem with a Static Hedge Ratio
    --------------------------------------
    The OLS hedge ratio is estimated once from all historical data and then
    held fixed. This assumes the relationship between t1 and t2 never changes.
    In reality, volatility regimes shift, correlations change, and sector
    dynamics evolve. A fixed hedge ratio becomes stale.

    What the Kalman Filter Does
    ---------------------------
    It treats the hedge ratio (and intercept) as hidden variables that we
    cannot observe directly but can estimate from prices. Every day it:

      1. PREDICTS where the hedge ratio probably is today, based on where
         it was yesterday (the "predict" step).

      2. OBSERVES the actual price of t1 today and computes the prediction
         error — how wrong was our predicted hedge ratio? This error is
         called the INNOVATION.

      3. UPDATES the hedge ratio estimate by moving it in the direction that
         would have reduced the innovation (the "update" step). How much it
         moves depends on the Kalman gain K — a number between 0 and 1 that
         balances trust in the model vs trust in the new data.

    The key insight: the innovation divided by its standard deviation is
    already a normalized signal in z-score-like units. We do not need a
    separate z-score step. The filter produces the signal directly.

    State-Space Model
    -----------------
    Observation equation (what we can measure):
        price_t1 = hedge_ratio_t * price_t2 + intercept_t + noise_obs

    State transition (how the hidden state evolves — a random walk):
        [hedge_ratio_t, intercept_t] = [hedge_ratio_{t-1}, intercept_{t-1}] + noise_state

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices with ticker columns.
    t1 : str
        The first ticker (treated as the dependent variable / y).
    t2 : str
        The second ticker (treated as the independent variable / x).
    delta : float
        State noise parameter. Controls how fast the hedge ratio is allowed
        to change. Higher delta = faster adaptation, more noise in the signal.
        Lower delta = slower adaptation, smoother but more lagged.
        Typical range: 1e-5 (very slow) to 1e-3 (fast). Default 1e-4.

    Returns
    -------
    pd.DataFrame indexed by date with columns:
        hedge_ratio     Time-varying hedge ratio estimate on each day
        intercept       Time-varying intercept estimate on each day
        spread          price_t1 - hedge_ratio * price_t2 - intercept
        innovation      Raw prediction error: actual price_t1 - predicted price_t1
        innovation_std  Expected standard deviation of the innovation
        zscore          innovation / innovation_std  (use this as the signal)
    """
    p1 = prices[t1].values
    p2 = prices[t2].values
    dates = prices.index
    n = len(p1)

    # --- Initialize state ---
    # State vector: beta = [hedge_ratio, intercept], shape (2,)
    # We start with a naive OLS estimate for the first 20 days to warm up.
    warmup = min(20, n // 5)
    beta = np.array([
        np.polyfit(p2[:warmup], p1[:warmup], 1)[0],  # initial hedge ratio
        np.mean(p1[:warmup] - np.polyfit(p2[:warmup], p1[:warmup], 1)[0] * p2[:warmup])
    ])

    # State covariance matrix: how uncertain we are about beta, shape (2, 2)
    # Start with a large value to express high initial uncertainty.
    P = np.eye(2) * 1.0

    # State noise covariance Q: how much beta can change per day
    # delta controls this — small delta = small Q = slow drift
    Q = delta / (1 - delta) * np.eye(2)

    # Observation noise variance R: estimated from the first warmup period
    resid_warmup = p1[:warmup] - (np.polyfit(p2[:warmup], p1[:warmup], 1)[0] * p2[:warmup] +
                                   np.mean(p1[:warmup] - np.polyfit(p2[:warmup], p1[:warmup], 1)[0] * p2[:warmup]))
    R = float(np.var(resid_warmup)) if len(resid_warmup) > 1 else 1.0

    # --- Storage ---
    hedge_ratios   = np.full(n, np.nan)
    intercepts     = np.full(n, np.nan)
    innovations    = np.full(n, np.nan)
    innovation_stds = np.full(n, np.nan)

    # --- Main filter loop ---
    # Cannot be vectorized: each step depends on the previous step's output.
    for t in range(n):
        # Observation vector: [price_t2, 1] so that beta @ obs = hedge_ratio*p2 + intercept
        obs = np.array([p2[t], 1.0])

        # --- PREDICT step ---
        # Our best guess for today's beta is yesterday's beta (random walk assumption).
        # Our uncertainty grows by Q each day because the state can drift.
        beta_pred = beta
        P_pred = P + Q

        # Predicted price of t1 using yesterday's hedge ratio estimate
        price_pred = obs @ beta_pred

        # --- INNOVATION ---
        # How wrong was our prediction?
        innovation = p1[t] - price_pred

        # Expected variance of the innovation (how surprised should we be?)
        S = obs @ P_pred @ obs + R

        # --- UPDATE step ---
        # Kalman gain: how much should we trust this new observation?
        # K close to 1 = trust the data, update aggressively
        # K close to 0 = trust the model, update slowly
        K = P_pred @ obs / S

        # Update the state estimate and covariance
        beta = beta_pred + K * innovation
        P = (np.eye(2) - np.outer(K, obs)) @ P_pred

        # --- Store results ---
        hedge_ratios[t]    = beta[0]
        intercepts[t]      = beta[1]
        innovations[t]     = innovation
        innovation_stds[t] = np.sqrt(S)

    # Do NOT subtract the intercept here. The intercept absorbs the price level
    # difference each day, which would make the spread nearly flat and kill the
    # z-score signal. The z-score step handles mean-centering, just like the
    # static approach. We keep the intercept column for reference only.
    spreads = p1 - hedge_ratios * p2

    return pd.DataFrame({
        "hedge_ratio":    hedge_ratios,
        "intercept":      intercepts,
        "spread":         spreads,
        "innovation":     innovations,
        "innovation_std": innovation_stds,
    }, index=dates)


def generate_kalman_signals(
    kalman_df: pd.DataFrame,
    window: int = None,
    entry: float = 2.0,
    exit: float = 0.0,
    stop: float = 3.0,
) -> pd.Series:
    """
    Generate trading signals from Kalman filter output.

    Why not use the raw innovations as the signal?
    -----------------------------------------------
    The Kalman filter fits the price-level relationship so well after warmup
    that day-to-day innovations become very small in absolute terms — too small
    to ever produce a meaningful z-score. This is expected behavior for a
    well-functioning filter on price levels.

    The solution is to separate the two jobs:
      - Kalman filter: estimate the dynamic hedge ratio (its real strength)
      - Rolling z-score: generate the trading signal from the dynamic spread

    This computes the z-score of the Kalman spread (p1 - hedge_ratio * p2 - intercept)
    using a rolling window, then applies the same entry/exit threshold logic.

    Parameters
    ----------
    kalman_df : pd.DataFrame
        Output of run_kalman().
    window : int, optional
        Rolling window for z-score. If None, uses full-history mean/std.
        Recommended: use the half-life from fit_ou() on the Kalman spread.
    entry, exit, stop : float
        Same meaning as in generate_signals() in zscore.py.

    Returns
    -------
    pd.Series of +1 / -1 / 0 signals.
    """
    zscore = compute_zscore(kalman_df["spread"], window=window)
    return generate_signals(zscore, entry=entry, exit=exit, stop=stop)
