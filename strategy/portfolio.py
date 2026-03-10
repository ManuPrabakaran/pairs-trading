import pandas as pd
import numpy as np

from pairs.selection import test_cointegration, compute_spread
from pairs.metrics import fit_ou
from signals.zscore import compute_zscore, generate_signals
from strategy.backtest import run_backtest, BacktestConfig


def build_equal_weight_portfolio(
    best_wfs: dict,
    result_key: str = "static_results",
) -> pd.Series:
    """
    Equal-weight portfolio: average daily net_pnl across all pairs.
    Each pair contributes equal capital. Flat days contribute 0.

    Parameters
    ----------
    result_key : str
        'static_results' (OLS hedge ratio, frozen per window) or
        'kalman_results' (adaptive Kalman hedge ratio).
    """
    pnl_df = pd.DataFrame({
        f"{t1}/{t2}": pd.concat(wf[result_key])["net_pnl"]
        for (t1, t2), wf in best_wfs.items()
    }).fillna(0)
    return pnl_df.mean(axis=1)


def compute_train_stats(
    prices: pd.DataFrame,
    best_wfs: dict,
    best_configs: dict,
    cost_bps: float = 5.0,
) -> list:
    """
    For every walk-forward window and every pair, run a backtest on the
    training period and return PnL volatility and Sharpe ratio.

    This is the shared computation that all three weighting schemes need.
    Computing it once avoids re-fitting the same training data three times.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history.
    best_wfs : dict
        Keyed by (t1, t2) → walk-forward result dict.
    best_configs : dict
        Keyed by (t1, t2) → (entry_z, exit_z).
    cost_bps : float
        Must match what was used in run_walk_forward.

    Returns
    -------
    list of dicts, one per window.
    Each dict maps (t1, t2) → {'vol': float, 'sharpe': float}.
    vol    : daily PnL standard deviation in the training period.
    sharpe : annualised Sharpe ratio in the training period.
             Negative if the strategy lost money during training.
    """
    pairs  = list(best_wfs.keys())
    ref_wf = best_wfs[pairs[0]]
    all_stats = []

    for k, w in enumerate(ref_wf["windows"]):
        window_stats = {}
        for (t1, t2) in pairs:
            entry_z, exit_z = best_configs[(t1, t2)]
            stop_z = entry_z + 1.0

            train_px = prices[[t1, t2]].loc[w["train_start"]:w["train_end"]]

            coint        = test_cointegration(train_px[t1], train_px[t2])
            hedge_ratio  = coint["hedge_ratio"]
            train_spread = compute_spread(train_px, t1, t2, hedge_ratio)
            ou           = fit_ou(train_spread)
            window_size  = int(round(ou["half_life"])) if ou["is_valid"] else 30

            zs      = compute_zscore(train_spread, window=window_size)
            sig     = generate_signals(zs, entry=entry_z, exit=exit_z, stop=stop_z)
            cfg     = BacktestConfig(t1=t1, t2=t2,
                                     hedge_ratio=hedge_ratio, cost_bps=cost_bps)
            result  = run_backtest(train_px, sig, config=cfg)

            pnl     = result["net_pnl"]
            vol     = float(pnl.std())
            ann_vol = vol * np.sqrt(252)
            sharpe  = (float(pnl.mean()) * 252) / ann_vol if ann_vol > 0 else 0.0

            window_stats[(t1, t2)] = {"vol": vol, "sharpe": sharpe}

        all_stats.append(window_stats)

    return all_stats


def build_weighted_portfolio(
    best_wfs: dict,
    train_stats: list,
    method: str = "risk_parity",
    result_key: str = "static_results",
) -> tuple:
    """
    Build a portfolio using per-window weights from training statistics.

    Three weighting schemes:

    'risk_parity'
        w ∝ 1 / vol
        Pairs contributing equal daily risk. Calm pairs get more capital.
        Makes no use of whether training returns were positive or negative.

    'sharpe_weighted'
        w ∝ max(0, sharpe)
        Pairs that earned more per unit of risk during training get more capital.
        Pairs with negative training Sharpe receive zero weight that window.
        Concentrates capital in historically better strategies.

    'combined'  (reward-risk weighted)
        w ∝ max(0, sharpe) / vol
        A pair must earn both: high Sharpe (good returns per risk) AND low vol
        (calm daily moves). High-vol pairs are penalised twice — once for their
        vol in the denominator and again because high-vol strategies tend to
        have lower Sharpes. KO/PEP (calm, consistent) will typically dominate.

    All weights are training-only. No test data is used. Fallback to equal
    weights if all raw scores are zero for a given window.

    Parameters
    ----------
    best_wfs : dict
        Keyed by (t1, t2) → walk-forward result dict.
    train_stats : list
        Output of compute_train_stats().
    method : str
        One of 'risk_parity', 'sharpe_weighted', 'combined'.
    result_key : str
        'static_results' or 'kalman_results'. Weights are always derived
        from training-period static backtests; this controls which test-period
        results the weights are applied to.

    Returns
    -------
    portfolio_daily : pd.Series  — daily portfolio net_pnl.
    window_weights  : list of dict — per-window weights for inspection.
    """
    pairs      = list(best_wfs.keys())
    ref_wf     = best_wfs[pairs[0]]
    n_windows  = len(ref_wf["windows"])

    window_pnl_pieces = []
    window_weights    = []

    for k in range(n_windows):
        stats = train_stats[k]

        raw = {}
        for pair in pairs:
            s = stats[pair]
            vol    = s["vol"]
            sharpe = s["sharpe"]

            if method == "risk_parity":
                raw[pair] = 1.0 / vol if vol > 0 else 0.0
            elif method == "sharpe_weighted":
                raw[pair] = max(0.0, sharpe)
            elif method == "combined":
                raw[pair] = max(0.0, sharpe) / vol if vol > 0 else 0.0
            else:
                raise ValueError(f"Unknown method: {method!r}")

        total = sum(raw.values())
        if total == 0:
            weights = {pair: 1.0 / len(pairs) for pair in pairs}
        else:
            weights = {pair: score / total for pair, score in raw.items()}

        window_weights.append({f"{t1}/{t2}": weights[(t1, t2)] for t1, t2 in pairs})

        test_pnls = {}
        for (t1, t2) in pairs:
            test_pnls[f"{t1}/{t2}"] = (
                best_wfs[(t1, t2)][result_key][k]["net_pnl"]
                * weights[(t1, t2)]
            )

        window_pnl = pd.DataFrame(test_pnls).fillna(0).sum(axis=1)
        window_pnl_pieces.append(window_pnl)

    portfolio_daily = pd.concat(window_pnl_pieces).sort_index()
    return portfolio_daily, window_weights


def build_risk_parity_portfolio(
    prices: pd.DataFrame,
    best_wfs: dict,
    best_configs: dict,
    cost_bps: float = 5.0,
    result_key: str = "static_results",
) -> tuple:
    """Backward-compatible wrapper. Prefer compute_train_stats + build_weighted_portfolio."""
    train_stats = compute_train_stats(prices, best_wfs, best_configs, cost_bps)
    return build_weighted_portfolio(best_wfs, train_stats, method="risk_parity",
                                    result_key=result_key)


def portfolio_stats(daily_pnl: pd.Series) -> dict:
    """Annualised Sharpe, total return, and max drawdown for a daily PnL series."""
    equity    = daily_pnl.cumsum()
    ann_ret   = daily_pnl.mean() * 252
    ann_vol   = daily_pnl.std() * np.sqrt(252)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
    total_ret = equity.iloc[-1]
    max_dd    = (equity - equity.cummax()).min()
    return {
        "sharpe_ratio": sharpe,
        "total_return": total_ret,
        "max_drawdown": max_dd,
    }
