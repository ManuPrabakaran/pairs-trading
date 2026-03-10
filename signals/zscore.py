import numpy as np
import pandas as pd


def compute_zscore(spread: pd.Series, window: int = None) -> pd.Series:
    """
    Standardize the spread into a z-score.

    window=None: use full-history mean and std (static).
    window=int:  use rolling mean and std (dynamic, avoids lookahead bias).
    """
    if window is None:
        z = (spread - spread.mean()) / spread.std()
    else:
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        z = (spread - mean) / std

    z.name = "zscore"
    return z


def generate_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.0,
    stop: float = 3.0,
) -> pd.Series:
    """
    Convert a z-score series into a position signal.

    Returns a Series of integers:
      +1  long the spread  (z-score fell below -entry: spread is cheap)
      -1  short the spread (z-score rose above +entry: spread is expensive)
       0  flat             (z-score crossed exit threshold, or hit stop)

    Position persists (forward-filled) until an exit or stop condition fires.
    Signals are NOT shifted here — shift by 1 day before applying to returns
    in the backtest to avoid lookahead bias.
    """
    values = zscore.values
    n = len(values)
    pos = np.zeros(n, dtype=int)
    current = 0

    for i in range(n):
        z = values[i]
        if np.isnan(z):
            pos[i] = current
            continue

        if current == 0:
            # Flat: enter on a large-enough move, skip if already past stop
            if z < -entry and abs(z) <= stop:
                current = 1
            elif z > entry and abs(z) <= stop:
                current = -1
        elif current == 1:
            # Long the spread: exit when z rises back toward (or past) the exit threshold,
            # or stop out if z falls further past the stop on the long side
            if z >= -exit or z < -stop:
                current = 0
        elif current == -1:
            # Short the spread: exit when z falls back toward (or past) the exit threshold,
            # or stop out if z rises further past the stop on the short side
            if z <= exit or z > stop:
                current = 0

        pos[i] = current

    signal = pd.Series(pos, index=zscore.index, name="signal")
    return signal
