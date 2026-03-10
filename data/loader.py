import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".cache"


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily adjusted close prices, with parquet file cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = "_".join(sorted(tickers)) + f"_{start}_{end}"
    cache_path = CACHE_DIR / f"{key}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    prices = raw["Close"]
    prices.columns.name = None  # yfinance 1.x adds a column axis name; remove it
    prices = prices.dropna(how="all")

    prices.to_parquet(cache_path)
    return prices
