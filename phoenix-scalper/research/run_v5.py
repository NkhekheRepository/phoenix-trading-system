"""V5 Experiment Runner — tests V5 entry engine against V4.1 on 90-day data."""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from freqtrade.configuration import Configuration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_all_data(pairs, timerange_str):
    from freqtrade.configuration.timerange import TimeRange
    tr = TimeRange.parse_timerange(timerange_str)
    base = Path("user_data/data/binance/futures")
    start = pd.Timestamp(datetime.fromtimestamp(tr.startts), tz="UTC")
    end = pd.Timestamp(datetime.fromtimestamp(tr.stopts), tz="UTC")
    dataframes = {}
    for pair in pairs:
        fname = pair.replace("/", "_").replace(":", "_") + "-5m-futures.feather"
        fpath = base / fname
        if not fpath.exists():
            continue
        df = pd.read_feather(fpath)
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(df) > 100:
            dataframes[pair] = df
    return dataframes


def compute_indicators(raw_data, config):
    from strategies.PhoenixScalperV5 import PhoenixScalperV5
    strategy = PhoenixScalperV5(config)
    strategy.dp = None
    indicator_data = {}
    for pair, df in raw_data.items():
        d = strategy.populate_indicators(df.copy(), {"pair": pair})
        indicator_data[pair] = d
    return indicator_data, strategy


def compute_metrics(trades_df):
    if trades_df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp": 0.0, "mdd": 0.0}
    n = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    wr = len(wins) / n if n > 0 else 0.0
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    exp = trades_df["pnl"].mean() if n > 0 else 0.0

    cum_pnl = trades_df["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    mdd = drawdown.max() if len(drawdown) > 0 else 0.0

    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "mdd": mdd}


def label_entries(entries_df, indicator_data):
    labels = []
    for _, row in entries_df.iterrows():
        pair = row["__pair"]
        idx = row["__df_idx"]
        df = indicator_data[pair]
        candle_idx = df.index.get_loc(idx) if idx in df.index else None
        if candle_idx is None:
            entry_idx = entries_df.index.get_loc(_)
            candle_idx = entry_idx
        if candle_idx + 12 >= len(df):
            labels.append({"pnl": 0.0, "win": False})
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
            labels.append({"pnl": 0.05, "win": True})
        elif hit_sl:
            labels.append({"pnl": -0.04, "win": False})
        else:
            exit_price = df.iloc[candle_idx + 12]["close"]
            pnl = (entry_price - exit_price) / entry_price
            labels.append({"pnl": pnl, "win": pnl > 0})
    return pd.DataFrame(labels)


def run_v5_experiment(experiment_name, indicator_data, strategy):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")

    all_entries = []
    for pair, df in indicator_data.items():
        df = strategy.populate_entry_trend(df.copy(), {"pair": pair})
        short_mask = df["enter_short"] == 1
        if short_mask.any():
            entries = df[short_mask].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[short_mask]
            all_entries.append(entries)

    if not all_entries:
        print("  No entries generated.")
        return {"name": experiment_name, "metrics": compute_metrics(pd.DataFrame()), "trades": pd.DataFrame()}

    entries = pd.concat(all_entries, ignore_index=True)
    labels = label_entries(entries, indicator_data)
    trades = pd.concat([
        entries[["__pair", "__direction"]].reset_index(drop=True),
        entries[["short_score"]].reset_index(drop=True),
        entries[["date"]].reset_index(drop=True),
        labels.reset_index(drop=True)
    ], axis=1)

    m = compute_metrics(trades)
    pf_str = f"{m['pf']:.3f}" if m['pf'] < 100 else "inf"
    print(f"  Trades: {m['n']}")
    print(f"  Win Rate: {m['wr']:.1%}")
    print(f"  PF: {pf_str}")
    print(f"  Expectancy: {m['exp']:.5f}")
    print(f"  MDD: {m['mdd']:.4f}")

    pf_pass = m['pf'] > 1.30
    exp_pass = m['exp'] > 0
    print(f"  Gates: PF>1.30 {'PASS' if pf_pass else 'FAIL'} | Exp>0 {'PASS' if exp_pass else 'FAIL'}")

    return {"name": experiment_name, "metrics": m, "trades": trades}


def run_walk_forward(trades_df, start_str="2026-04-20", end_str="2026-07-20"):
    start = pd.Timestamp(start_str, tz="UTC")
    end = pd.Timestamp(end_str, tz="UTC")
    current = start
    n_pass = 0
    n_total = 0

    print(f"\nWalk-Forward (5-day window, 1-day step):")
    print(f"  {'Period':>12}  {'N':>3}  {'WR%':>5}  {'PF':>8}  {'Exp':>8}  [Status]")
    print(f"  {'-'*55}")

    while current + pd.Timedelta(days=5) <= end:
        w_end = current + pd.Timedelta(days=5)
        w_trades = trades_df[(trades_df["date"] >= current) & (trades_df["date"] < w_end)]
        if len(w_trades) >= 2:
            wm = compute_metrics(w_trades)
            pf_str = f"{wm['pf']:.2f}" if wm['pf'] < 100 else "inf"
            status = 'PASS' if wm['pf'] > 1.30 and wm['exp'] > 0 else 'FAIL'
            if status == 'PASS':
                n_pass += 1
            n_total += 1
            print(f"  {current.strftime('%m-%d')}-{w_end.strftime('%m-%d')}: {wm['n']:>3}  {wm['wr']:>5.0%}  {pf_str:>8}  {wm['exp']:>8.4f}  [{status}]")
        current += pd.Timedelta(days=1)

    wf_rate = n_pass / max(n_total, 1)
    print(f"\n  WF pass rate: {n_pass}/{n_total} ({wf_rate:.0%})")
    return wf_rate


def run_per_pair(trades_df):
    print(f"\nPer-pair breakdown:")
    for pair in sorted(trades_df["__pair"].unique()):
        pt = trades_df[trades_df["__pair"] == pair]
        pm = compute_metrics(pt)
        pf_str = f"{pm['pf']:.2f}" if pm['pf'] < 100 else "inf"
        pair_g = 'PASS' if pm['pf'] >= 1.0 else 'FAIL'
        print(f"  {pair:>20}: N={pm['n']:>3}  WR={pm['wr']:.0%}  PF={pf_str:>6}  Exp={pm['exp']:.5f}  [{pair_g}]")


def run_per_month(trades_df):
    print(f"\nPer-month breakdown:")
    trades_df = trades_df.copy()
    trades_df["month"] = pd.to_datetime(trades_df["date"]).dt.to_period("M")
    for month, mt in trades_df.groupby("month"):
        mm = compute_metrics(mt)
        pf_str = f"{mm['pf']:.2f}" if mm['pf'] < 100 else "inf"
        print(f"  {month}: N={mm['n']:>3}  WR={mm['wr']:.0%}  PF={pf_str:>6}  Exp={mm['exp']:.5f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("timerange", help="Timerange string, e.g. 20260420-20260720")
    parser.add_argument("--config", default="config-v4.json")
    parser.add_argument("--experiment", default="v5_default")
    args = parser.parse_args()

    config = Configuration.from_files([args.config])

    TOP10 = [
        'SOL/USDT:USDT', 'BTC/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
        'ETH/USDT:USDT', 'ADA/USDT:USDT', 'LINK/USDT:USDT', 'LTC/USDT:USDT',
    ]

    print(f"Loading data for {len(TOP10)} pairs, timerange {args.timerange}...")
    raw_data = load_all_data(TOP10, args.timerange)
    print(f"Loaded {len(raw_data)} pairs")

    indicator_data, strategy = compute_indicators(raw_data, config)

    result = run_v5_experiment(args.experiment, indicator_data, strategy)

    if not result["trades"].empty:
        run_walk_forward(result["trades"], args.timerange.split("-")[0][:4] + "-" + args.timerange.split("-")[0][4:6] + "-" + args.timerange.split("-")[0][6:],
                         args.timerange.split("-")[1][:4] + "-" + args.timerange.split("-")[1][4:6] + "-" + args.timerange.split("-")[1][6:])
        run_per_pair(result["trades"])
        run_per_month(result["trades"])

    results_path = Path("research/v5_experiment_results.json")
    existing = []
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
    existing.append({
        "name": result["name"],
        "metrics": result["metrics"],
        "timestamp": datetime.now().isoformat(),
    })
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)


if __name__ == "__main__":
    main()
