import numpy as np
import pandas as pd


def fit_ou(spread: pd.Series) -> dict:
    """
    Fit an Ornstein-Uhlenbeck (OU) process to a spread series.

    The OU process is the continuous-time model that describes mean reversion:
        dX = theta * (mu - X) * dt + sigma * dW

    We fit it by running OLS on its discrete-time equivalent (AR(1)):
        X_t = a + b * X_{t-1} + epsilon

    From the regression coefficients we recover the OU parameters:
        b      = exp(-theta * dt)  ->  theta = -log(b)     (dt = 1 day)
        a      = mu * (1 - b)      ->  mu    = a / (1 - b)
        sigma  = std(epsilon)

    Half-life is the number of days for the spread to close half the distance
    to its mean:
        half_life = log(2) / theta

    Parameters
    ----------
    spread : pd.Series
        Daily spread values (e.g. from compute_spread).

    Returns
    -------
    dict with keys:
        b           AR(1) coefficient (must be in (0,1) for mean reversion)
        theta       Mean reversion speed (per day)
        mu          Long-run mean of the spread
        sigma       Volatility of daily spread changes
        half_life   Days to revert halfway to the mean
        r_squared   Goodness of fit of the AR(1) regression
        is_valid    True if the spread is mean-reverting (0 < b < 1)
    """
    spread = spread.dropna()

    # Build the lagged regression: X_t ~ a + b * X_{t-1}
    X_lag = spread.shift(1).dropna()
    X_cur = spread.iloc[1:]  # align with X_lag

    # OLS via numpy: [a, b] = polyfit(X_lag, X_cur, 1) but we want intercept
    # Use the closed-form OLS solution
    n = len(X_lag)
    sum_x = X_lag.sum()
    sum_y = X_cur.sum()
    sum_xy = (X_lag * X_cur).sum()
    sum_xx = (X_lag ** 2).sum()

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    a = (sum_y - b * sum_x) / n

    residuals = X_cur - (a + b * X_lag)
    sigma = residuals.std()

    # R-squared: how well does the AR(1) model explain daily spread changes?
    ss_res = (residuals ** 2).sum()
    ss_tot = ((X_cur - X_cur.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Recover OU parameters from regression coefficients
    # Clamp b slightly away from 0 and 1 to avoid log(0) or log(1)/0
    b_clamped = float(np.clip(b, 1e-6, 1 - 1e-6))
    theta = -np.log(b_clamped)           # mean reversion speed per day
    mu = a / (1 - b) if abs(1 - b) > 1e-10 else float(spread.mean())
    half_life = np.log(2) / theta if theta > 0 else np.inf

    # A spread is tradeable if it reverts (0 < b < 1) and half-life is
    # between 1 day and 1 trading year
    is_valid = (0 < b < 1) and (1 <= half_life <= 252)

    return {
        "b": round(float(b), 6),
        "theta": round(float(theta), 6),
        "mu": round(float(mu), 6),
        "sigma": round(float(sigma), 6),
        "half_life": round(float(half_life), 2),
        "r_squared": round(float(r_squared), 4),
        "is_valid": bool(is_valid),
    }


def half_life(spread: pd.Series) -> float:
    """Return the mean-reversion half-life in days. Convenience wrapper."""
    return fit_ou(spread)["half_life"]
