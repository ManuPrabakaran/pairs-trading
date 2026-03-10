import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from pairs.selection import test_cointegration, compute_spread
from pairs.metrics import fit_ou
from signals.zscore import compute_zscore, generate_signals


def generate_live_signals(
    prices: pd.DataFrame,
    pairs: list,
    configs: dict,
    train_years: int = 2,
) -> pd.DataFrame:
    """
    For each pair, fit on the trailing train_years window and return
    the current signal, z-score, and position metadata.

    This is the bridge between historical backtesting and live trading.
    No walk-forward loop — a single training window ending today,
    evaluated on the most recent data point.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history up to today. Must cover at least train_years.
    pairs : list of (t1, t2) tuples
        Pairs to evaluate.
    configs : dict
        Keyed by (t1, t2) -> (entry_z, exit_z). Use the validated
        parameters from the walk-forward grid search.
    train_years : int
        Length of the training window in years.

    Returns
    -------
    pd.DataFrame with one row per pair and columns:
        pair             — 't1/t2' label
        signal           — current position: +1 (long), -1 (short), 0 (flat)
        zscore           — current z-score of the spread
        entry_z          — entry threshold used
        hedge_ratio      — OLS hedge ratio fitted on the training window
        half_life        — OU half-life in days (from training window)
        days_in_position — consecutive days in the current position
        last_trade_date  — most recent date where position changed
        pvalue           — cointegration p-value on the current training window
        health           — 'OK', 'WARN_PVALUE', 'WARN_HEDGE_RATIO', or 'WARN_BOTH'

    Health flags
    ------------
    WARN_PVALUE      : p-value >= 0.05 — pair no longer statistically cointegrated
                       in the current training window. Signal may be unreliable.
    WARN_HEDGE_RATIO : hedge ratio is negative — the fitted relationship has inverted.
                       Economically unusual for pairs selected on positive co-movement.
    WARN_BOTH        : both conditions apply.
    OK               : neither warning applies.
    """
    today = prices.index[-1]
    train_start = today - relativedelta(years=train_years)

    rows = []
    for (t1, t2) in pairs:
        entry_z, exit_z = configs[(t1, t2)]
        stop_z = entry_z + 1.0

        train_px = prices[[t1, t2]].loc[train_start:]

        # Fit hedge ratio and OU parameters on training window
        coint = test_cointegration(train_px[t1], train_px[t2])
        hedge_ratio = coint["hedge_ratio"]
        pvalue      = coint["pvalue"]

        # Health checks
        warn_pvalue       = pvalue >= 0.05
        warn_hedge_ratio  = hedge_ratio < 0
        if warn_pvalue and warn_hedge_ratio:
            health = "WARN_BOTH"
        elif warn_pvalue:
            health = "WARN_PVALUE"
        elif warn_hedge_ratio:
            health = "WARN_HEDGE_RATIO"
        else:
            health = "OK"

        train_spread = compute_spread(train_px, t1, t2, hedge_ratio)
        ou = fit_ou(train_spread)
        window_size = int(round(ou["half_life"])) if ou["is_valid"] else 30

        # Rolling z-score and signal over the training window
        zscore = compute_zscore(train_spread, window=window_size)
        signals = generate_signals(zscore, entry=entry_z, exit=exit_z, stop=stop_z)

        current_signal = int(signals.iloc[-1])
        current_zscore = float(zscore.iloc[-1])

        # Days consecutively in current position
        if current_signal != 0:
            reversed_signals = signals.iloc[::-1]
            changes = (reversed_signals != current_signal)
            days_in = int(changes.argmax()) if changes.any() else len(signals)
        else:
            days_in = 0

        # Most recent date where position changed
        trade_mask = signals.diff().fillna(0) != 0
        if trade_mask.any():
            last_trade = signals.index[trade_mask][-1].strftime("%Y-%m-%d")
        else:
            last_trade = "—"

        rows.append({
            "pair":             f"{t1}/{t2}",
            "signal":           current_signal,
            "zscore":           round(current_zscore, 3),
            "entry_z":          entry_z,
            "hedge_ratio":      round(hedge_ratio, 4),
            "half_life":        round(ou["half_life"], 1),
            "days_in_position": days_in,
            "last_trade_date":  last_trade,
            "pvalue":           round(pvalue, 4),
            "health":           health,
        })

    return pd.DataFrame(rows)


def compute_sizing(
    signals_df: pd.DataFrame,
    prices: pd.DataFrame,
    pairs: list,
    train_years: int = 2,
    portfolio_size: float = 100_000,
) -> pd.DataFrame:
    """
    Compute dollar exposure for each pair under two sizing modes.

    Both modes use risk parity weights (w ∝ 1/vol), but they differ in
    how they handle pairs that are currently flat (signal = 0).

    Mode: 'fixed'  (matches the backtest)
    ------------------------------------------
    Weights are computed across ALL pairs regardless of their current signal.
    Flat pairs receive zero dollar exposure — their capital allocation sits
    in cash until that pair enters a position.

    Pro: Faithful to the walk-forward backtest. The SR=0.96 result assumed
         this behaviour — flat pairs never deployed capital. You are trading
         exactly the strategy you validated.
    Con: When most pairs are flat, most of the portfolio is idle. You might
         deploy only 30–40% of capital on a day when only 2 pairs are active.

    Mode: 'normalized'  (always fully deployed)
    ------------------------------------------
    Weights are recomputed using ONLY the active pairs (signal ≠ 0),
    then normalised to sum to 1. All capital goes to whichever pairs are
    currently in a position.

    Pro: Capital is never left idle. On a day with 2 active pairs you deploy
         100% of the portfolio rather than 30–40%.
    Con: Positions are larger than the backtest implied — risk per active pair
         is higher. The SR=0.96 result does NOT apply to this mode. You would
         need to re-run the backtest with this sizing to know the real numbers.
         It also concentrates the portfolio when few pairs are active, which
         can amplify drawdowns in bad periods.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output of generate_live_signals().
    prices : pd.DataFrame
        Full price history (same DataFrame passed to generate_live_signals).
    pairs : list of (t1, t2) tuples
    train_years : int
        Training window length — must match what was used in generate_live_signals.
    portfolio_size : float
        Total portfolio value in dollars.

    Returns
    -------
    pd.DataFrame with columns:
        pair | signal | signal_label | rp_weight_fixed | exposure_fixed
             | rp_weight_normalized | exposure_normalized
    """
    today       = prices.index[-1]
    train_start = today - relativedelta(years=train_years)

    # Compute trailing spread volatility for each pair
    vols = {}
    for (t1, t2) in pairs:
        train_px = prices[[t1, t2]].loc[train_start:]
        coint    = test_cointegration(train_px[t1], train_px[t2])
        spread   = compute_spread(train_px, t1, t2, coint["hedge_ratio"])
        vols[f"{t1}/{t2}"] = float(spread.diff().std())

    # Fixed weights: RP across all pairs, sum to 1
    inv_vol_all   = {p: 1.0 / v for p, v in vols.items() if v > 0}
    total_all     = sum(inv_vol_all.values())
    weights_fixed = {p: w / total_all for p, w in inv_vol_all.items()}

    # Normalized weights: RP across active pairs only, sum to 1.
    # Pairs with WARN_PVALUE or WARN_BOTH are excluded from new capital deployment —
    # cointegration failing is the foundational requirement for this strategy.
    # WARN_HEDGE_RATIO alone is not blocking: the relationship structure is intact,
    # the sign is temporarily noisy. Existing positions run to their natural exit.
    blocked = {"WARN_PVALUE", "WARN_BOTH"}
    active_pairs  = [
        row["pair"] for _, row in signals_df.iterrows()
        if row["signal"] != 0 and row["health"] not in blocked
    ]
    inv_vol_active = {p: inv_vol_all[p] for p in active_pairs if p in inv_vol_all}
    total_active   = sum(inv_vol_active.values())
    weights_norm   = {
        p: (w / total_active if total_active > 0 else 0.0)
        for p, w in inv_vol_active.items()
    }

    rows = []
    for _, sig_row in signals_df.iterrows():
        pair    = sig_row["pair"]
        sig     = int(sig_row["signal"])
        health  = sig_row["health"]
        is_blocked = health in blocked
        wf   = weights_fixed.get(pair, 0.0)
        wn   = weights_norm.get(pair, 0.0)
        # Zero out exposure for blocked pairs — warning has teeth for pvalue failures
        if is_blocked:
            wf = 0.0
            wn = 0.0
        rows.append({
            "pair":                 pair,
            "signal":               sig,
            "signal_label":         {1: "LONG", -1: "SHORT", 0: "FLAT"}[sig],
            "rp_weight_fixed":      wf,
            "exposure_fixed":       wf * portfolio_size * sig,
            "rp_weight_normalized": wn,
            "exposure_normalized":  wn * portfolio_size * sig,
        })

    return pd.DataFrame(rows)
