import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def compute_correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna().corr()


def test_cointegration(s1: pd.Series, s2: pd.Series) -> dict:
    """Engle-Granger cointegration test. Returns p-value and OLS hedge ratio."""
    s1, s2 = s1.dropna().align(s2.dropna(), join="inner")
    _, pvalue, _ = coint(s1, s2)

    # OLS hedge ratio: s1 = hedge_ratio * s2 + c
    hedge_ratio = np.polyfit(s2, s1, 1)[0]

    return {"pvalue": pvalue, "hedge_ratio": hedge_ratio}


def find_pairs(
    prices: pd.DataFrame, p_threshold: float = 0.05
) -> list[dict]:
    """
    Screen all ticker pairs for cointegration.
    Returns list of dicts sorted by p-value (ascending).
    """
    tickers = prices.columns.tolist()
    results = []

    for t1, t2 in itertools.combinations(tickers, 2):
        result = test_cointegration(prices[t1], prices[t2])
        if result["pvalue"] < p_threshold:
            results.append({"pair": (t1, t2), **result})

    return sorted(results, key=lambda x: x["pvalue"])


def compute_spread(prices: pd.DataFrame, t1: str, t2: str, hedge_ratio: float) -> pd.Series:
    """Spread = s1 - hedge_ratio * s2, aligned on common index."""
    s1, s2 = prices[t1].align(prices[t2], join="inner")
    spread = s1 - hedge_ratio * s2
    spread.name = f"{t1}/{t2} spread"
    return spread
