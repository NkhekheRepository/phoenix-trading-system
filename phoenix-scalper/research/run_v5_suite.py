"""V5 Experiment Suite — tests multiple V5 entry engine variants on 90-day data."""

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

FILTER_PAIRS = set()


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


def label_entries(entries_df, indicator_data):
    labels = []
    for _, row in entries_df.iterrows():
        pair = row["__pair"]
        idx = row["__df_idx"]
        df = indicator_data[pair]
        try:
            candle_idx = df.index.get_loc(idx)
        except KeyError:
            candle_idx = entries_df.index.get_loc(_)
        if candle_idx + 12 >= len(df):
            labels.append({"pnl": 0.0, "win": False})
            continue
        entry_price = df.iloc[candle_idx]["close"]
        future = df.iloc[candle_idx + 1 : candle_idx + 13]
        worst = future["low"].min()
        tp = entry_price * 0.95
        sl = entry_price * 1.04
        hit_tp = worst <= tp
        hit_sl = future["high"].max() >= sl
        if hit_tp:
            labels.append({"pnl": 0.05, "win": True})
        elif hit_sl:
            labels.append({"pnl": -0.04, "win": False})
        else:
            exit_price = df.iloc[candle_idx + 12]["close"]
            pnl = (entry_price - exit_price) / entry_price
            labels.append({"pnl": pnl, "win": pnl > 0})
    return pd.DataFrame(labels)


def compute_metrics(trades_df):
    if trades_df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp": 0.0, "mdd": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    n = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    wr = len(wins) / n if n > 0 else 0.0
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    exp = trades_df["pnl"].mean() if n > 0 else 0.0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0

    cum_pnl = trades_df["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    mdd = (running_max - cum_pnl).max() if len(cum_pnl) > 0 else 0.0

    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "mdd": mdd, "avg_win": avg_win, "avg_loss": avg_loss}


EXPERIMENTS = {}


def register_experiment(name, desc):
    def decorator(func):
        EXPERIMENTS[name] = {"name": name, "desc": desc, "func": func}
        return func
    return decorator


@register_experiment("v5_original", "V5 as-is from PhoenixScalperV5 (no score override)")
def exp_v5_original(indicator_data, strategy):
    all_entries = []
    for pair, df in indicator_data.items():
        df = strategy.populate_entry_trend(df.copy(), {"pair": pair})
        mask = df["enter_short"] == 1
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = df.index[mask]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


@register_experiment("v5_strict_adx", "V5 + ADX>=30 + volume>=2.0 + RSI<50 + macdhist<0")
def exp_v5_strict(indicator_data, strategy):
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        e = df.copy()
        e = strategy.populate_entry_trend(e, {"pair": pair})
        mask = (
            (e["enter_short"] == 1) &
            (e["adx"] >= 30) &
            (e["volume_ratio"] >= 2.0) &
            (e["rsi_5"] < 50) &
            (e["macdhist"] < 0)
        )
        if mask.any():
            entries = e[mask].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = e.index[mask]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


@register_experiment("v5_trend_only", "Pure trend: close<ema50+minus_di>plus_di+ADX>=25+vol>=1.5+macdhist<0+pattern")
def exp_v5_trend_only(indicator_data, strategy):
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        e = df.copy()
        trend_down = (
            (e["close"] < e["ema_50"]) &
            (e["minus_di"] > e["plus_di"]) &
            (e["adx"] >= 25)
        )
        vol_spike = e["volume_ratio"] >= 1.5
        bearish = (e["close"] < e["open"]) & (e["rsi_5"] < 55) & (e["macdhist"] < 0)
        e = strategy.populate_entry_trend(e, {"pair": pair})
        pattern_mask = (e["enter_short"] == 1)
        entry = trend_down & vol_spike & bearish & pattern_mask
        if entry.any():
            entries = e[entry].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = e.index[entry]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


@register_experiment("v5_strict_breakdown", "Breakdown only: strict breakdown + ADX>=30 + vol>=2.0")
def exp_v5_breakdown(indicator_data, strategy):
    all_entries = []
    lb = 14
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        e = df.copy()
        breakdown = e["close"] < e["low"].rolling(lb).min().shift(1)
        strict = (
            breakdown &
            (e["adx"] >= 30) &
            (e["volume_ratio"] >= 2.0) &
            (e["close"] < e["ema_50"]) &
            (e["minus_di"] > e["plus_di"]) &
            (e["close"] < e["open"]) &
            (e["rsi_5"] < 50) &
            (e["macdhist"] < 0) &
            (e["volume"] > 0)
        )
        if strict.any():
            entries = e[strict].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = e.index[strict]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


@register_experiment("v5_ultra_conservative", "ADX>=35+vol>=2.5+RSI<45+close<ema50+minus_di>plus_di+macdhist<0+bearish_candle+score>=50")
def exp_v5_ultra(indicator_data, strategy):
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        e = df.copy()
        e = strategy.populate_entry_trend(e, {"pair": pair})
        strict = (
            (e["enter_short"] == 1) &
            (e["adx"] >= 35) &
            (e["volume_ratio"] >= 2.5) &
            (e["rsi_5"] < 45) &
            (e["close"] < e["ema_50"]) &
            (e["minus_di"] > e["plus_di"]) &
            (e["macdhist"] < 0) &
            (e["close"] < e["open"])
        )
        if strict.any():
            entries = e[strict].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = e.index[strict]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


@register_experiment("v5_btc_eth_only", "V5 but only BTC+ETH pairs")
def exp_v5_btc_eth(indicator_data, strategy):
    allowed = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    all_entries = []
    for pair, df in indicator_data.items():
        if pair not in allowed:
            continue
        df = strategy.populate_entry_trend(df.copy(), {"pair": pair})
        mask = df["enter_short"] == 1
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = df.index[mask]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


@register_experiment("v5_btc_eth_strict", "BTC+ETH only + ADX>=30 + vol>=2.0 + RSI<50")
def exp_v5_btc_eth_strict(indicator_data, strategy):
    allowed = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    all_entries = []
    for pair, df in indicator_data.items():
        if pair not in allowed:
            continue
        e = df.copy()
        e = strategy.populate_entry_trend(e, {"pair": pair})
        strict = (
            (e["enter_short"] == 1) &
            (e["adx"] >= 30) &
            (e["volume_ratio"] >= 2.0) &
            (e["rsi_5"] < 50)
        )
        if strict.any():
            entries = e[strict].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = e.index[strict]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


@register_experiment("v5_remove_avax_bnb", "V5 but remove AVAX+BNB (E12 insight)")
def exp_v5_no_avax_bnb(indicator_data, strategy):
    block = {"AVAX/USDT:USDT", "BNB/USDT:USDT"}
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in block:
            continue
        df = strategy.populate_entry_trend(df.copy(), {"pair": pair})
        mask = df["enter_short"] == 1
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__df_idx"] = df.index[mask]
            all_entries.append(entries)
    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


def walk_forward(trades_df, start, end):
    current = start
    n_pass = 0
    n_total = 0
    while current + pd.Timedelta(days=5) <= end:
        w_end = current + pd.Timedelta(days=5)
        w_trades = trades_df[(trades_df["date"] >= current) & (trades_df["date"] < w_end)]
        if len(w_trades) >= 2:
            wm = compute_metrics(w_trades)
            status = 'PASS' if wm['pf'] > 1.30 and wm['exp'] > 0 else 'FAIL'
            if status == 'PASS':
                n_pass += 1
            n_total += 1
        current += pd.Timedelta(days=1)
    return n_pass, n_total, n_pass / max(n_total, 1)


def main():
    TOP10 = [
        'SOL/USDT:USDT', 'BTC/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
        'ETH/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'BNB/USDT:USDT',
        'LINK/USDT:USDT', 'LTC/USDT:USDT',
    ]

    timerange = "20260420-20260720"
    config = Configuration.from_files(["config-v4.json"])

    print(f"Loading 90-day data for {len(TOP10)} pairs...")
    raw_data = load_all_data(TOP10, timerange)
    print(f"Loaded {len(raw_data)} pairs")

    indicator_data, strategy = compute_indicators(raw_data, config)

    start_date = pd.Timestamp("2026-04-20", tz="UTC")
    end_date = pd.Timestamp("2026-07-20", tz="UTC")

    all_results = []

    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"\n{'='*65}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"  {exp_info['desc']}")
        print(f"{'='*65}")

        entries = exp_info["func"](indicator_data, strategy)

        if entries.empty:
            print("  No entries generated.")
            all_results.append({"name": exp_name, "desc": exp_info["desc"], "n": 0, "wr": 0, "pf": 0, "exp": 0, "wf_pass": 0, "wf_total": 0, "wf_rate": 0})
            continue

        labels = label_entries(entries, indicator_data)
        trades = pd.concat([
            entries[["__pair"]].reset_index(drop=True),
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
        print(f"  Avg Win: {m['avg_win']:.5f}  Avg Loss: {m['avg_loss']:.5f}")

        wf_pass, wf_total, wf_rate = walk_forward(trades, start_date, end_date)
        print(f"  WF: {wf_pass}/{wf_total} ({wf_rate:.0%})")

        pf_g = "PASS" if m['pf'] > 1.30 else "FAIL"
        exp_g = "PASS" if m['exp'] > 0 else "FAIL"
        wf_g = "PASS" if wf_rate > 0.60 else "FAIL"
        overall = "PASS" if all([pf_g == "PASS", exp_g == "PASS", wf_g == "PASS"]) else "FAIL"
        print(f"  Gates: PF {pf_g} | Exp {exp_g} | WF {wf_g} | OVERALL: {overall}")

        print(f"\n  Per-pair:")
        for pair in sorted(trades["__pair"].unique()):
            pt = trades[trades["__pair"] == pair]
            pm = compute_metrics(pt)
            pf_s = f"{pm['pf']:.2f}" if pm['pf'] < 100 else "inf"
            pg = 'PASS' if pm['pf'] >= 1.0 else 'FAIL'
            print(f"    {pair:>20}: N={pm['n']:>3}  WR={pm['wr']:.0%}  PF={pf_s:>6}  [{pg}]")

        all_results.append({
            "name": exp_name, "desc": exp_info["desc"],
            "n": m['n'], "wr": m['wr'], "pf": m['pf'], "exp": m['exp'], "mdd": m['mdd'],
            "avg_win": m['avg_win'], "avg_loss": m['avg_loss'],
            "wf_pass": wf_pass, "wf_total": wf_total, "wf_rate": wf_rate,
            "overall": overall,
        })

    print(f"\n\n{'='*65}")
    print("SUMMARY — ALL V5 EXPERIMENTS")
    print(f"{'='*65}")
    print(f"{'Experiment':<30} {'N':>5} {'WR%':>6} {'PF':>8} {'Exp':>10} {'WF%':>6} {'Gate':>6}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: x.get('pf', 0) if isinstance(x.get('pf', 0), (int, float)) else 0, reverse=True):
        pf_s = f"{r['pf']:.3f}" if r['pf'] < 100 else "inf"
        wf_s = f"{r.get('wf_rate', 0):.0%}"
        print(f"{r['name']:<30} {r['n']:>5} {r['wr']:>5.0%} {pf_s:>8} {r['exp']:>10.5f} {wf_s:>6} {r.get('overall', 'FAIL'):>6}")

    results_path = Path("research/v5_experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
