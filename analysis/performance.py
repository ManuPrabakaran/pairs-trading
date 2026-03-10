import numpy as np
import pandas as pd


def summarize(results: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    Compute performance metrics from a backtest results DataFrame
    (as returned by strategy/backtest.py run_backtest).
    """
    net_pnl = results["net_pnl"].dropna()
    equity = results["equity"].dropna()

    total_return = equity.iloc[-1]
    annualized_return = net_pnl.mean() * periods_per_year

    std = net_pnl.std()
    sharpe = (net_pnl.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = equity - rolling_max
    max_drawdown = drawdown.min()

    trades = results["trade"]
    num_trades = int(trades.sum())

    # A "trade" PnL: sum net_pnl within each position holding period
    position = results["position"]
    trade_pnls = []
    in_trade = False
    trade_pnl = 0.0
    for i in range(len(results)):
        pos = position.iloc[i]
        if pos != 0:
            in_trade = True
            trade_pnl += net_pnl.iloc[i]
        elif in_trade:
            trade_pnls.append(trade_pnl)
            trade_pnl = 0.0
            in_trade = False
    if in_trade:
        trade_pnls.append(trade_pnl)

    win_rate = (sum(p > 0 for p in trade_pnls) / len(trade_pnls)) if trade_pnls else 0.0
    avg_trade_pnl = (sum(trade_pnls) / len(trade_pnls)) if trade_pnls else 0.0

    return {
        "total_return": round(total_return, 4),
        "annualized_return": round(annualized_return, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 3),
        "avg_trade_pnl": round(avg_trade_pnl, 6),
    }
