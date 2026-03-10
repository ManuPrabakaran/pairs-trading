"""
pairs/screening.py — Reusable universe expansion funnel.

Encapsulates the three-stage screening pipeline from notebook 14:
  1. Benjamini-Hochberg multiple-testing correction across all candidate pairs
  2. OU half-life and cross-correlation quality filters
  3. Walk-forward out-of-sample validation

Notebook 14 documents the reasoning in detail; this module lets later
notebooks call the same logic without duplicating code.

Typical usage:
    from pairs.screening import run_expansion_funnel

    results = run_expansion_funnel(
        candidate_tickers=CANDIDATE_TICKERS,
        existing_pairs=VALIDATED_PAIRS,
        prices=prices,
    )
    finalists      = results['finalists']       # DataFrame
    wf_results     = results['wf_results']      # {(t1,t2): wf_data}
    best_configs   = results['best_configs']    # {(t1,t2): (entry_z, exit_z)}
"""

import itertools

import numpy as np
import pandas as pd

from pairs.selection import test_cointegration, compute_spread
from pairs.metrics import fit_ou
from strategy.walk_forward import run_parameter_grid


# ---------------------------------------------------------------------------
# Stage 1 — statistical screening
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues, alpha=0.05):
    """
    Return a boolean mask: True for each entry that survives BH correction
    at a False Discovery Rate of alpha.

    Parameters
    ----------
    pvalues : array-like of float
    alpha   : float — FDR threshold (default 0.05)

    Returns
    -------
    np.ndarray of bool, same length as pvalues
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = pvalues[order] <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    k = int(np.where(below)[0].max())
    mask = np.zeros(m, dtype=bool)
    mask[order[: k + 1]] = True
    return mask


def screen_candidates(candidate_tickers, prices, bh_alpha=0.05, completeness_max=0.10):
    """
    Run cointegration tests on all pairs from candidate_tickers and apply
    Benjamini-Hochberg correction.

    Parameters
    ----------
    candidate_tickers : dict  — {'Industry': ['T1', 'T2', ...], ...}
    prices            : pd.DataFrame — full price history
    bh_alpha          : float — FDR threshold
    completeness_max  : float — drop tickers with more than this fraction of NaNs

    Returns
    -------
    df_pairs   : pd.DataFrame — all pairs tested, with pvalue, hedge_ratio, bh_pass
    survivors  : pd.DataFrame — rows where bh_pass is True, sorted by pvalue
    """
    # Drop tickers with too many missing values
    valid_candidates = {}
    for industry, tickers in candidate_tickers.items():
        valid_candidates[industry] = [
            t for t in tickers
            if t in prices.columns and prices[t].isna().mean() <= completeness_max
        ]

    rows = []
    for industry, tickers in valid_candidates.items():
        for t1, t2 in itertools.combinations(tickers, 2):
            result = test_cointegration(prices[t1], prices[t2])
            rows.append({"industry": industry, "t1": t1, "t2": t2, **result})

    if not rows:
        empty = pd.DataFrame(columns=["industry", "t1", "t2", "pvalue", "hedge_ratio", "bh_pass"])
        return empty, empty

    df_pairs = pd.DataFrame(rows)
    df_pairs["bh_pass"] = benjamini_hochberg(df_pairs["pvalue"].values, alpha=bh_alpha)

    survivors = (
        df_pairs[df_pairs["bh_pass"]]
        .sort_values("pvalue")
        .reset_index(drop=True)
    )
    return df_pairs, survivors


# ---------------------------------------------------------------------------
# Stage 2 — quality filters
# ---------------------------------------------------------------------------

def quality_filter(survivors, existing_pairs, prices, ou_min=5, ou_max=126, corr_max=0.30):
    """
    Filter BH survivors by OU half-life and correlation with existing pairs.

    Parameters
    ----------
    survivors      : pd.DataFrame — output of screen_candidates()
    existing_pairs : list of (t1, t2) tuples
    prices         : pd.DataFrame
    ou_min, ou_max : int — tradeable half-life range in trading days
    corr_max       : float — maximum tolerated absolute correlation with any existing pair

    Returns
    -------
    quality_df : pd.DataFrame — all survivors with filter results
    finalists  : pd.DataFrame — rows passing both filters
    """
    # Pre-compute spread returns for existing pairs
    existing_spread_rets = {}
    for t1, t2 in existing_pairs:
        coint = test_cointegration(prices[t1], prices[t2])
        spread = compute_spread(prices, t1, t2, coint["hedge_ratio"])
        existing_spread_rets[f"{t1}/{t2}"] = spread.diff().dropna()

    rows = []
    for _, row in survivors.iterrows():
        t1, t2, industry = row["t1"], row["t2"], row["industry"]

        spread = compute_spread(prices, t1, t2, row["hedge_ratio"])
        ou = fit_ou(spread.dropna())

        cand_ret = spread.diff().dropna()
        corrs = []
        for key, exist_ret in existing_spread_rets.items():
            aligned_cand, aligned_exist = cand_ret.align(exist_ret, join="inner")
            if len(aligned_cand) > 60:
                corrs.append(abs(float(aligned_cand.corr(aligned_exist))))
        max_corr = max(corrs) if corrs else 0.0

        passes_ou = ou_min <= ou["half_life"] <= ou_max
        passes_corr = max_corr < corr_max

        rows.append({
            "industry":          industry,
            "t1":                t1,
            "t2":                t2,
            "half_life":         round(ou["half_life"], 1),
            "ou_valid":          ou["is_valid"],
            "max_corr_existing": round(max_corr, 3),
            "passes_ou":         passes_ou,
            "passes_corr":       passes_corr,
            "finalist":          passes_ou and passes_corr,
        })

    quality_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    finalists = (
        quality_df[quality_df["finalist"]].reset_index(drop=True)
        if not quality_df.empty
        else pd.DataFrame()
    )
    return quality_df, finalists


# ---------------------------------------------------------------------------
# Stage 3 — walk-forward validation
# ---------------------------------------------------------------------------

def validate_finalists(
    finalists,
    prices,
    train_years=2,
    test_years=1,
    cost_bps=5.0,
    entry_zs=None,
    exit_zs=None,
):
    """
    Run the parameter grid walk-forward on each finalist pair.

    Parameters
    ----------
    finalists   : pd.DataFrame — output of quality_filter()
    prices      : pd.DataFrame
    train_years, test_years, cost_bps : walk-forward settings
    entry_zs, exit_zs : lists of thresholds to grid-search

    Returns
    -------
    wf_results   : dict  — {(t1, t2): wf_data}  (full grid)
    best_configs : dict  — {(t1, t2): (entry_z, exit_z)}
    best_wfs     : dict  — {(t1, t2): best wf_data entry}
    """
    if entry_zs is None:
        entry_zs = [1.5, 2.0, 2.5]
    if exit_zs is None:
        exit_zs = [0.0, 0.5]

    wf_results = {}
    best_configs = {}
    best_wfs = {}

    for _, row in finalists.iterrows():
        t1, t2 = row["t1"], row["t2"]
        grid = run_parameter_grid(
            prices[[t1, t2]], t1, t2,
            entry_zs=entry_zs, exit_zs=exit_zs,
            train_years=train_years, test_years=test_years, cost_bps=cost_bps,
        )
        wf_results[(t1, t2)] = grid

        best_key = max(
            grid,
            key=lambda k: grid[k]["static_stats"]["sharpe_ratio"],
        )
        best_configs[(t1, t2)] = best_key
        best_wfs[(t1, t2)] = grid[best_key]

    return wf_results, best_configs, best_wfs


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_expansion_funnel(
    candidate_tickers,
    existing_pairs,
    prices,
    bh_alpha=0.05,
    completeness_max=0.10,
    ou_min=5,
    ou_max=126,
    corr_max=0.30,
    train_years=2,
    test_years=1,
    cost_bps=5.0,
    entry_zs=None,
    exit_zs=None,
    verbose=True,
):
    """
    Run the full three-stage expansion funnel.

    Stage 1 — BH-corrected cointegration screening
    Stage 2 — OU half-life and cross-correlation quality filters
    Stage 3 — Walk-forward out-of-sample validation

    Parameters
    ----------
    candidate_tickers : dict — {'Industry': ['T1', 'T2', ...], ...}
    existing_pairs    : list of (t1, t2) tuples — current portfolio pairs
    prices            : pd.DataFrame — full price history
    bh_alpha          : float — FDR threshold (default 0.05)
    completeness_max  : float — max tolerated NaN fraction per ticker
    ou_min, ou_max    : int — tradeable half-life range in trading days
    corr_max          : float — max tolerated absolute correlation with any existing pair
    train_years       : int — walk-forward training window
    test_years        : int — walk-forward test window
    cost_bps          : float — round-trip transaction cost in basis points
    entry_zs, exit_zs : lists — parameter grid for walk-forward
    verbose           : bool — print progress

    Returns
    -------
    dict with keys:
        'all_pairs'    — pd.DataFrame of every pair tested + pvalue + bh_pass
        'survivors'    — pd.DataFrame of BH survivors
        'quality'      — pd.DataFrame of quality filter results
        'finalists'    — pd.DataFrame of pairs passing all pre-validation filters
        'wf_results'   — dict {(t1, t2): full grid}
        'best_configs' — dict {(t1, t2): (entry_z, exit_z)}
        'best_wfs'     — dict {(t1, t2): best walk-forward data}
    """
    if entry_zs is None:
        entry_zs = [1.5, 2.0, 2.5]
    if exit_zs is None:
        exit_zs = [0.0, 0.5]

    # Stage 1 — statistical screening
    if verbose:
        total_candidates = sum(
            len(list(itertools.combinations(
                [t for t in tickers if t in prices.columns and prices[t].isna().mean() <= completeness_max],
                2
            )))
            for tickers in candidate_tickers.values()
        )
        print(f"Stage 1 — BH screening ({total_candidates} candidate pairs, FDR={bh_alpha})...")

    df_pairs, survivors = screen_candidates(
        candidate_tickers, prices, bh_alpha=bh_alpha, completeness_max=completeness_max
    )

    if verbose:
        print(f"  {len(df_pairs)} tested → {len(survivors)} BH survivors")

    # Stage 2 — quality filters
    if verbose:
        print(f"Stage 2 — quality filters (OU {ou_min}–{ou_max} days, corr < {corr_max})...")

    quality_df, finalists = quality_filter(
        survivors, existing_pairs, prices,
        ou_min=ou_min, ou_max=ou_max, corr_max=corr_max,
    )

    if verbose:
        print(f"  {len(survivors)} survivors → {len(finalists)} finalists")

    # Stage 3 — walk-forward validation
    wf_results, best_configs, best_wfs = {}, {}, {}
    if not finalists.empty:
        if verbose:
            print(f"Stage 3 — walk-forward validation ({len(finalists)} pairs)...")
        wf_results, best_configs, best_wfs = validate_finalists(
            finalists, prices,
            train_years=train_years, test_years=test_years, cost_bps=cost_bps,
            entry_zs=entry_zs, exit_zs=exit_zs,
        )
        if verbose:
            for (t1, t2), wf in best_wfs.items():
                sr = wf["static_stats"]["sharpe_ratio"]
                cfg = best_configs[(t1, t2)]
                print(f"  {t1}/{t2}: OOS SR={sr:.2f}, params={cfg}")
    else:
        if verbose:
            print("  No finalists — skipping walk-forward.")

    return {
        "all_pairs":    df_pairs,
        "survivors":    survivors,
        "quality":      quality_df,
        "finalists":    finalists,
        "wf_results":   wf_results,
        "best_configs": best_configs,
        "best_wfs":     best_wfs,
    }
