from dataclasses import dataclass
from typing import Union
import pandas as pd


@dataclass
class BacktestConfig:
    """
    All parameters needed to run a single pairs backtest.

    Grouping them here makes it easy to pass a single config object
    to the backtest, store it alongside results, and iterate over
    multiple configs when running parameter sweeps later.

    Attributes
    ----------
    t1 : str
        Ticker symbol of the first leg (the one we go long when spread is cheap).
    t2 : str
        Ticker symbol of the second leg (the one we short when spread is cheap).
    hedge_ratio : float or pd.Series
        How many units of t2 to hold per unit of t1.
        Pass a float for a static hedge ratio (OLS approach).
        Pass a pd.Series for a time-varying hedge ratio (Kalman approach).
        Spread = price(t1) - hedge_ratio * price(t2).
    cost_bps : float
        One-way transaction cost in basis points (1 bp = 0.01%).
        A round-trip (open + close) costs 2 * cost_bps.
        5 bps is a reasonable estimate for liquid ETFs.
    """
    t1: str
    t2: str
    hedge_ratio: Union[float, pd.Series]
    cost_bps: float = 5.0


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig = None,
    # Legacy keyword args kept for backward compatibility
    t1: str = None,
    t2: str = None,
    hedge_ratio: float = None,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Vectorized pairs backtest.

    Accepts either a BacktestConfig object or individual keyword arguments.

    How it works
    ------------
    Each day we hold a position of +1 (long spread) or -1 (short spread)
    or 0 (flat), as determined by the signal.

    The spread return on day t is:
        spread_ret = return(t1) - hedge_ratio * return(t2)

    If we are long the spread (+1) and the spread widens, we make money.
    If we are short the spread (-1) and the spread narrows, we make money.

    IMPORTANT: The signal is shifted 1 day forward before being applied.
    This means a signal generated at market close on day t is executed
    at market close on day t+1. This prevents lookahead bias (we never
    trade on information we could not have had at the time).

    Transaction costs are charged on the day the position changes.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices, columns are ticker symbols.
    signals : pd.Series
        Integer series of +1 / -1 / 0, indexed by date.
        Typically the output of signals.zscore.generate_signals().
    config : BacktestConfig, optional
        If provided, t1/t2/hedge_ratio/cost_bps are taken from here.

    Returns
    -------
    pd.DataFrame with one row per trading day:
        t1_ret      Daily return of t1 (percentage move as a decimal)
        t2_ret      Daily return of t2
        position    Active position on that day (+1, -1, or 0)
        gross_pnl   Return earned before transaction costs
        trade       True if the position changed from the previous day
        cost        Transaction cost paid (0 on non-trade days)
        net_pnl     Return after transaction costs (gross_pnl - cost)
        equity      Cumulative sum of net_pnl (starts at 0)
    """
    # Resolve config vs legacy kwargs
    if config is not None:
        t1 = config.t1
        t2 = config.t2
        hedge_ratio = config.hedge_ratio
        cost_bps = config.cost_bps

    if t1 is None or t2 is None or hedge_ratio is None:
        raise ValueError("Provide either a BacktestConfig or t1, t2, hedge_ratio kwargs.")

    # --- Daily returns ---
    # pct_change() gives NaN on the first row; that row is dropped at the end.
    t1_ret = prices[t1].pct_change()
    t2_ret = prices[t2].pct_change()

    # --- Apply the 1-day execution lag ---
    # Signal is generated using prices known at close of day t.
    # We assume execution happens at close of day t+1.
    # shift(1) moves the signal forward by one row.
    position = signals.shift(1).fillna(0).astype(int)
    position = position.reindex(t1_ret.index, fill_value=0)

    # --- Spread return ---
    # When position = +1: we own t1 and are short hedge_ratio units of t2.
    # Our daily PnL is therefore: return(t1) - hedge_ratio * return(t2).
    # When position = -1: the opposite trade, so PnL flips sign.
    #
    # If hedge_ratio is a Series (Kalman filter output), align it to the
    # price index so each day uses that day's estimated hedge ratio.
    # The hedge ratio is also shifted 1 day to avoid lookahead bias —
    # we can only use yesterday's estimate when executing today's trade.
    if isinstance(hedge_ratio, pd.Series):
        hedge_ratio = hedge_ratio.shift(1).reindex(t1_ret.index).ffill()

    # Convert hedge ratio from price space to return space.
    # OLS estimates hedge_ratio in price levels, so hedge_ratio ≈ P_t1 / P_t2.
    # The correct dollar-neutral spread return is:
    #   r_t1 - (hedge_ratio * P_t2 / P_t1) * r_t2
    # Since hedge_ratio ≈ P_t1/P_t2, this normalises to approximately r_t1 - r_t2.
    # Without this, pairs where one stock trades at a large price multiple of the
    # other (e.g. GS=$200, MS=$40 → hedge_ratio=5) record spread returns that are
    # 5x too large, producing fictitious drawdowns and inflated volatility.
    p2_over_p1 = (prices[t2] / prices[t1]).shift(1).reindex(t1_ret.index).ffill()
    hedge_ratio_ret = hedge_ratio * p2_over_p1

    spread_ret = t1_ret - hedge_ratio_ret * t2_ret
    gross_pnl = position * spread_ret

    # --- Transaction costs ---
    # diff().abs() counts the number of legs traded on each day:
    #   0 -> +1 or +1 -> 0: 1 leg traded (5 bps)
    #   -1 -> +1 or +1 -> -1: 2 legs traded (10 bps) — reversal
    # Using the raw magnitude instead of a boolean ensures reversals pay
    # the correct round-trip cost rather than being undercharged by half.
    trade_legs = position.diff().abs()
    trade = trade_legs > 0
    cost = trade_legs * (cost_bps / 10_000)

    net_pnl = gross_pnl - cost
    equity = net_pnl.cumsum()

    return pd.DataFrame({
        "t1_ret":    t1_ret,
        "t2_ret":    t2_ret,
        "position":  position,
        "gross_pnl": gross_pnl,
        "trade":     trade,
        "cost":      cost,
        "net_pnl":   net_pnl,
        "equity":    equity,
    }).dropna()
