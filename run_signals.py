#!/usr/bin/env python3
# On Mac, run as: python3 run_signals.py
"""
run_signals.py — Daily signal generation script.

Run from the project root:
    python run_signals.py

Writes:
    signals_output.json     — latest signals (overwritten each run)
    signals_history.jsonl   — one record per run (appended, never overwritten)

Schedule with cron (runs at 7am Monday–Friday):
    0 7 * * 1-5 cd /path/to/quant-project && /path/to/venv/bin/python run_signals.py

Or call from a trading bot:
    import subprocess, sys
    subprocess.run([sys.executable, 'run_signals.py'], cwd='/path/to/quant-project')
    # then read signals_output.json
"""

import json
import datetime
import sys
from pathlib import Path

import pandas as pd

from data.loader import fetch_prices
from strategy.pairs_config import VALIDATED_PAIRS, PAIR_CONFIGS
from strategy.live import generate_live_signals, compute_sizing

START          = '2010-01-01'
TRAIN_YEARS    = 2
PORTFOLIO_SIZE = 100_000
OUTPUT_PATH    = Path('signals_output.json')
HISTORY_PATH   = Path('signals_history.jsonl')


def build_record(signals_df, sizing_df, today, last_price_date):
    record = {
        "generated_at":    today,
        "last_price_date": last_price_date,
        "portfolio_size":  PORTFOLIO_SIZE,
        "signals":         [],
    }

    for _, row in sizing_df.iterrows():
        sig_row = signals_df[signals_df['pair'] == row['pair']].iloc[0]
        t1, t2  = row['pair'].split('/')
        sig     = int(row['signal'])

        if sig == 1:
            buy_ticker, sell_ticker = t1, t2
        elif sig == -1:
            buy_ticker, sell_ticker = t2, t1
        else:
            buy_ticker, sell_ticker = None, None

        record["signals"].append({
            "pair":                       row['pair'],
            "signal":                     sig,
            "signal_label":               row['signal_label'],
            "buy_ticker":                 buy_ticker,
            "sell_ticker":                sell_ticker,
            "zscore":                     float(sig_row['zscore']),
            "confidence":                 round(float(row['rp_weight_fixed']), 6),
            "days_in_position":           int(sig_row['days_in_position']),
            "last_trade_date":            sig_row['last_trade_date'],
            "pvalue":                     float(sig_row['pvalue']),
            "health":                     sig_row['health'],
            "hedge_ratio":                float(sig_row['hedge_ratio']),
            "half_life_days":             float(sig_row['half_life']),
            "fixed_dollar_exposure":      round(float(row['exposure_fixed']), 2),
            "normalized_dollar_exposure": round(float(row['exposure_normalized']), 2),
        })

    return record


def main():
    today = datetime.date.today().isoformat()
    print(f"[{today}] Fetching prices...")

    all_tickers = sorted({t for pair in VALIDATED_PAIRS for t in pair})
    prices      = fetch_prices(all_tickers, START, today)
    last_price_date = str(prices.index[-1].date())

    print(f"[{today}] Last price date: {last_price_date}. Generating signals...")

    signals_df = generate_live_signals(
        prices, VALIDATED_PAIRS, PAIR_CONFIGS, train_years=TRAIN_YEARS
    )
    sizing_df = compute_sizing(
        signals_df, prices, VALIDATED_PAIRS,
        train_years=TRAIN_YEARS,
        portfolio_size=PORTFOLIO_SIZE,
    )

    record = build_record(signals_df, sizing_df, today, last_price_date)

    # Overwrite latest signals
    OUTPUT_PATH.write_text(json.dumps(record, indent=2))
    print(f"[{today}] Written to {OUTPUT_PATH}")

    # Append to history (never overwrite)
    with open(HISTORY_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f"[{today}] Appended to {HISTORY_PATH}")

    # Summary
    active = [
        f"{s['pair']} {s['signal_label']}"
        for s in record['signals'] if s['signal'] != 0
    ]
    if active:
        print(f"[{today}] Active: {', '.join(active)}")
    else:
        print(f"[{today}] No active positions.")

    # Health warnings
    unhealthy = [s for s in record['signals'] if s['health'] != 'OK']
    blocked   = {'WARN_PVALUE', 'WARN_BOTH'}
    for s in unhealthy:
        action = "new entries blocked" if s['health'] in blocked else "existing positions run to exit"
        print(f"[{today}] HEALTH WARNING — {s['pair']}: {s['health']} "
              f"(pvalue={s['pvalue']:.4f}, hedge_ratio={s['hedge_ratio']:.4f}) — {action}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
