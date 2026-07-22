"""BTC-Only Experiment Suite — find optimal entry engine for BTC/USDT:USDT only."""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from functools import reduce
from freqtrade.configuration import Configuration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_btc_data(timerange_str):
    from freqtrade.configuration.timerange import TimeRange
    tr = TimeRange.parse_timerange(timerange_str)
    base = Path("user_data/data/binance/futures")
    start = pd.Timestamp(datetime.fromtimestamp(tr.startts), tz="UTC")
    end = pd.Timestamp(datetime.fromtimestamp(tr.stopts), tz="UTC")
    pair = "BTC/USDT:USDT"
    fname = pair.replace("/", "_").replace(":", "_") + "-5m-futures.feather"
    fpath = base / fname
    df = pd.read_feather(fpath)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    return df


def compute_indicators(df, config):
    from strategies.PhoenixScalperV5 import PhoenixScalperV5
    strategy = PhoenixScalperV5(config)
    strategy.dp = None
    d = strategy.populate_indicators(df.copy(), {"pair": "BTC/USDT:USDT"})
    return d, strategy


def label_entries(df, entry_mask):
    labels = []
    entry_indices = df.index[entry_mask]
    for idx in entry_indices:
        candle_idx = df.index.get_loc(idx)
        if candle_idx + 12 >= len(df):
            labels.append({"pnl": 0.0, "win": False, "date": df.iloc[candle_idx]["date"]})
            continue
        entry_price = df.iloc[candle_idx]["close"]
        future = df.iloc[candle_idx + 1 : candle_idx + 13]
        worst = future["low"].min()
        best = future["high"].max()
        tp = entry_price * 0.95
        sl = entry_price * 1.04
        hit_tp = worst <= tp
        hit_sl = best >= sl
        if hit_tp:
            labels.append({"pnl": 0.05, "win": True, "date": df.iloc[candle_idx]["date"]})
        elif hit_sl:
            labels.append({"pnl": -0.04, "win": False, "date": df.iloc[candle_idx]["date"]})
        else:
            exit_price = df.iloc[candle_idx + 12]["close"]
            pnl = (entry_price - exit_price) / entry_price
            labels.append({"pnl": pnl, "win": pnl > 0, "date": df.iloc[candle_idx]["date"]})
    return pd.DataFrame(labels)


def compute_metrics(trades_df):
    if trades_df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp": 0.0, "mdd": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    n = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    wr = len(wins) / n
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    exp = trades_df["pnl"].mean()
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0
    cum_pnl = trades_df["pnl"].cumsum()
    mdd = (cum_pnl.cummax() - cum_pnl).max()
    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "mdd": mdd, "avg_win": avg_win, "avg_loss": avg_loss}


def walk_forward(trades_df, start, end):
    current = start
    n_pass = 0
    n_total = 0
    window_results = []
    td = trades_df.copy()
    td["date"] = pd.to_datetime(td["date"]).dt.tz_localize("UTC") if td["date"].dt.tz is None else td["date"]
    while current + pd.Timedelta(days=5) <= end:
        w_end = current + pd.Timedelta(days=5)
        w_trades = td[(td["date"] >= current) & (td["date"] < w_end)]
        if len(w_trades) >= 2:
            wm = compute_metrics(w_trades)
            pf_str = f"{wm['pf']:.2f}" if wm['pf'] < 100 else "inf"
            status = 'PASS' if wm['pf'] > 1.30 and wm['exp'] > 0 else 'FAIL'
            if status == 'PASS':
                n_pass += 1
            n_total += 1
            window_results.append({"start": current, "end": w_end, "n": wm['n'], "wr": wm['wr'], "pf": wm['pf'], "exp": wm['exp'], "status": status})
        current += pd.Timedelta(days=1)
    return n_pass, n_total, window_results


def run_experiment(name, df, entry_mask, start_date, end_date):
    print(f"\n{'='*55}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*55}")

    if not entry_mask.any():
        print("  No entries.")
        return None

    labels = label_entries(df, entry_mask)
    dates = df.loc[entry_mask, ["date"]].reset_index(drop=True)
    trades = pd.DataFrame({
        "date": pd.to_datetime(df.loc[entry_mask, "date"].values).tz_localize("UTC"),
        "pnl": labels["pnl"].values,
        "win": labels["win"].values,
    })

    m = compute_metrics(trades)
    pf_str = f"{m['pf']:.3f}" if m['pf'] < 100 else "inf"
    print(f"  N={m['n']}  WR={m['wr']:.1%}  PF={pf_str}  Exp={m['exp']:.5f}  MDD={m['mdd']:.4f}")
    print(f"  Avg Win={m['avg_win']:.5f}  Avg Loss={m['avg_loss']:.5f}")

    wf_pass, wf_total, wf_windows = walk_forward(trades, start_date, end_date)
    wf_rate = wf_pass / max(wf_total, 1)
    print(f"  WF: {wf_pass}/{wf_total} ({wf_rate:.0%})")

    pf_g = "PASS" if m['pf'] > 1.30 else "FAIL"
    exp_g = "PASS" if m['exp'] > 0 else "FAIL"
    wf_g = "PASS" if wf_rate > 0.60 else "FAIL"
    aw_al = "PASS" if m['avg_win'] > abs(m['avg_loss']) else "FAIL"
    overall = "PASS" if all([pf_g == "PASS", exp_g == "PASS", wf_g == "PASS", aw_al == "PASS"]) else "FAIL"
    print(f"  Gates: PF {pf_g} | Exp {exp_g} | WF {wf_g} | W>L {aw_al} | {overall}")

    return {"name": name, "metrics": m, "wf_rate": wf_rate, "wf_pass": wf_pass, "wf_total": wf_total, "overall": overall, "windows": wf_windows}


def main():
    config = Configuration.from_files(["config-v4.json"])
    timerange = "20260420-20260720"
    start_date = pd.Timestamp("2026-04-20", tz="UTC")
    end_date = pd.Timestamp("2026-07-20", tz="UTC")

    print("Loading BTC data (90 days)...")
    df = load_btc_data(timerange)
    print(f"Loaded {len(df)} candles ({df['date'].min()} to {df['date'].max()})")

    df, strategy = compute_indicators(df, config)

    results = []

    # Experiment 1: V5 baseline (BTC only)
    mask = strategy.populate_entry_trend(df.copy(), {"pair": "BTC/USDT:USDT"})["enter_short"] == 1
    r = run_experiment("btc_v5_baseline", df, mask, start_date, end_date)
    if r: results.append(r)

    # Experiment 2: V5 + ADX>=25
    mask2 = mask & (df["adx"] >= 25)
    r = run_experiment("btc_adx25", df, mask2, start_date, end_date)
    if r: results.append(r)

    # Experiment 3: V5 + ADX>=30
    mask3 = mask & (df["adx"] >= 30)
    r = run_experiment("btc_adx30", df, mask3, start_date, end_date)
    if r: results.append(r)

    # Experiment 4: V5 + volume>=1.5
    mask4 = mask & (df["volume_ratio"] >= 1.5)
    r = run_experiment("btc_vol15", df, mask4, start_date, end_date)
    if r: results.append(r)

    # Experiment 5: V5 + volume>=2.0
    mask5 = mask & (df["volume_ratio"] >= 2.0)
    r = run_experiment("btc_vol20", df, mask5, start_date, end_date)
    if r: results.append(r)

    # Experiment 6: Strict trend (close<ema50 + minus_di>plus_di + ADX>=25 + vol>=1.5 + macdhist<0 + RSI<50)
    mask6 = (
        mask &
        (df["close"] < df["ema_50"]) &
        (df["minus_di"] > df["plus_di"]) &
        (df["adx"] >= 25) &
        (df["volume_ratio"] >= 1.5) &
        (df["macdhist"] < 0) &
        (df["rsi_5"] < 50)
    )
    r = run_experiment("btc_strict_trend", df, mask6, start_date, end_date)
    if r: results.append(r)

    # Experiment 7: Ultra strict (ADX>=30 + vol>=2.0 + RSI<45 + close<ema50 + macdhist<0)
    mask7 = (
        mask &
        (df["adx"] >= 30) &
        (df["volume_ratio"] >= 2.0) &
        (df["rsi_5"] < 45) &
        (df["close"] < df["ema_50"]) &
        (df["macdhist"] < 0)
    )
    r = run_experiment("btc_ultra_strict", df, mask7, start_date, end_date)
    if r: results.append(r)

    # Experiment 8: Breakdown only + strict filters
    lb = 14
    breakdown = df["close"] < df["low"].rolling(lb).min().shift(1)
    mask8 = (
        breakdown &
        (df["adx"] >= 25) &
        (df["volume_ratio"] >= 1.5) &
        (df["close"] < df["ema_50"]) &
        (df["minus_di"] > df["plus_di"]) &
        (df["close"] < df["open"]) &
        (df["rsi_5"] < 50) &
        (df["macdhist"] < 0) &
        (df["volume"] > 0)
    )
    r = run_experiment("btc_breakdown_strict", df, mask8, start_date, end_date)
    if r: results.append(r)

    # Experiment 9: Rally fail only + strict filters
    rally_fail = (
        (df["high"] >= df["ema_21"] * 0.995) &
        (df["close"] < df["ema_21"]) &
        (df["close"] < df["open"]) &
        (df["rsi_5"] > 50) &
        (df["rsi_5"] < 75) &
        (df["volume_ratio"] >= 1.5) &
        (df["adx"] >= 25) &
        (df["minus_di"] > df["plus_di"]) &
        (df["volume"] > 0)
    )
    mask9 = rally_fail & (df["close"] < df["ema_50"]) & (df["macdhist"] < 0) & (df["adx"] >= 30)
    r = run_experiment("btc_rally_fail_strict", df, mask9, start_date, end_date)
    if r: results.append(r)

    # Experiment 10: Bear momentum only + strict
    bear_mom = (
        (df["close"] < df["open"]) &
        (df["volume_ratio"] >= 1.5 * 1.3) &
        (df["adx"] >= 25 * 1.1) &
        (df["minus_di"] > df["plus_di"]) &
        (df["rsi_5"] < 55) &
        (df["macdhist"] < 0)
    )
    mask10 = bear_mom & (df["close"] < df["ema_50"]) & (df["adx"] >= 30) & (df["volume_ratio"] >= 2.0)
    r = run_experiment("btc_bear_mom_strict", df, mask10, start_date, end_date)
    if r: results.append(r)

    # Experiment 11: Score-based (score>=50 + ADX>=25 + trend down)
    e = strategy.populate_entry_trend(df.copy(), {"pair": "BTC/USDT:USDT"})
    mask11 = (
        (e["enter_short"] == 1) &
        (e["short_score"] >= 50) &
        (df["adx"] >= 25) &
        (df["close"] < df["ema_50"]) &
        (df["minus_di"] > df["plus_di"])
    )
    r = run_experiment("btc_score50_trend", df, mask11, start_date, end_date)
    if r: results.append(r)

    # Experiment 12: Score-based (score>=55)
    mask12 = (
        (e["enter_short"] == 1) &
        (e["short_score"] >= 55) &
        (df["adx"] >= 25) &
        (df["close"] < df["ema_50"])
    )
    r = run_experiment("btc_score55", df, mask12, start_date, end_date)
    if r: results.append(r)

    # Experiment 13: Simple — close<ema50 + bearish candle + ADX>=25 + vol>=1.5 + RSI<50
    mask13 = (
        (df["close"] < df["ema_50"]) &
        (df["close"] < df["open"]) &
        (df["adx"] >= 25) &
        (df["volume_ratio"] >= 1.5) &
        (df["rsi_5"] < 50) &
        (df["minus_di"] > df["plus_di"]) &
        (df["macdhist"] < 0) &
        (df["volume"] > 0)
    )
    r = run_experiment("btc_simple_strict", df, mask13, start_date, end_date)
    if r: results.append(r)

    # Experiment 14: Minimal — just breakdown pattern + ADX>=20 + vol>=1.0
    mask14 = (
        breakdown &
        (df["adx"] >= 20) &
        (df["volume_ratio"] >= 1.0) &
        (df["close"] < df["open"]) &
        (df["volume"] > 0)
    )
    r = run_experiment("btc_minimal_breakdown", df, mask14, start_date, end_date)
    if r: results.append(r)

    print(f"\n\n{'='*65}")
    print("SUMMARY — BTC-ONLY EXPERIMENTS (90-day)")
    print(f"{'='*65}")
    print(f"{'Experiment':<25} {'N':>5} {'WR%':>6} {'PF':>8} {'Exp':>10} {'WF%':>6} {'Gate':>6}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.get('wf_rate', 0), reverse=True):
        pf_s = f"{r['metrics']['pf']:.3f}" if r['metrics']['pf'] < 100 else "inf"
        print(f"{r['name']:<25} {r['metrics']['n']:>5} {r['metrics']['wr']:>5.0%} {pf_s:>8} {r['metrics']['exp']:>10.5f} {r['wf_rate']:>5.0%} {r['overall']:>6}")

    results_path = Path("research/btc_only_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
