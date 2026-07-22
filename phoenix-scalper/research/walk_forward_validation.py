"""
E7 — Walk-forward validation for PhoenixScalperV4_1.

Splits the 32-day window into rolling sub-periods, evaluates entry performance
per window, and checks against validation gates:
  PF > 1.30, Expectancy > 0, MDD < 15%, Avg winner > avg loser

Uses the same direct-feather + strategy.indicators approach as E6.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import datetime as dt

TOP10 = [
    "SOL/USDT:USDT", "BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "BNB/USDT:USDT", "ETH/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT",
]

HOLD_BARS = 12  # 1h at 5m
LEVERAGE = 10   # V4.1 uses 10x
SL_PCT = 0.005  # 5% SL at 10x (non-leveraged 0.5%)


def load_all_data(pairs, timerange_str):
    """Load 5m futures data from feather files."""
    from freqtrade.configuration.timerange import TimeRange
    tr = TimeRange.parse_timerange(timerange_str)
    base = Path("user_data/data/binance/futures")
    start = pd.Timestamp(dt.datetime.fromtimestamp(tr.startts), tz="UTC")
    end = pd.Timestamp(dt.datetime.fromtimestamp(tr.stopts), tz="UTC")
    dataframes = {}
    for pair in pairs:
        fname = pair.replace("/", "_").replace(":", "_") + "-5m-futures.feather"
        fpath = base / fname
        if not fpath.exists():
            continue
        df = pd.read_feather(fpath)
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(df) > 60:
            dataframes[pair] = df
    return dataframes


def evaluate_entries(all_entries, indicator_data, hold_bars=HOLD_BARS):
    """Label entries and compute performance metrics."""
    results = []
    for _, entry in all_entries.iterrows():
        pair = entry["__pair"]
        direction = entry["__direction"]
        df_idx = int(entry["__df_idx"])
        entry_price = entry["close"]
        entry_date = entry["date"]

        df = indicator_data[pair]
        future_idx = df_idx + hold_bars
        if future_idx >= len(df):
            continue

        exit_price = df.iloc[future_idx]["close"]
        if direction == "long":
            pct = (exit_price - entry_price) / entry_price
        else:
            pct = (entry_price - exit_price) / entry_price

        # Apply SL: if price moved against > SL_PCT, capped at SL
        max_loss_pct = -SL_PCT
        pnl = max(pct, max_loss_pct) * LEVERAGE

        results.append({
            "pair": pair,
            "direction": direction,
            "date": entry_date,
            "pnl_pct": pnl,
            "win": 1 if pnl > 0 else 0,
        })
    return pd.DataFrame(results)


def compute_gates(trades_df):
    """Compute validation gate metrics."""
    if trades_df.empty or len(trades_df) < 3:
        return {"PASS": False, "reason": "insufficient trades", "n_trades": len(trades_df),
                "win_rate": 0, "PF": 0, "expectancy": 0, "max_dd_pct": 0,
                "avg_win": 0, "avg_loss": 0, "PF_PASS": False, "EXP_PASS": False,
                "DD_PASS": False, "WIN_GT_LOSS": False, "all_pairs_pass": False,
                "pair_pf": {}}

    n = len(trades_df)
    wins = trades_df[trades_df["win"] == 1]
    losses = trades_df[trades_df["win"] == 0]
    n_win = len(wins)
    n_loss = len(losses)
    wr = n_win / n

    gross_profit = wins["pnl_pct"].sum() if n_win > 0 else 0
    gross_loss = abs(losses["pnl_pct"].sum()) if n_loss > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = wins["pnl_pct"].mean() if n_win > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if n_loss > 0 else 0
    expectancy = trades_df["pnl_pct"].mean()

    # MDD from cumulative PnL
    cum = trades_df["pnl_pct"].cumsum()
    peak = cum.cummax()
    dd = peak - cum
    max_dd_pct = dd.max() / 100 * 100  # already in % terms

    # Per-pair PF
    pair_pf = {}
    for pair in trades_df["pair"].unique():
        p = trades_df[trades_df["pair"] == pair]
        pw = p[p["win"] == 1]["pnl_pct"].sum()
        pl = abs(p[p["win"] == 0]["pnl_pct"].sum())
        pair_pf[pair] = pw / pl if pl > 0 else float("inf") if pw > 0 else 0

    gates = {
        "PF": pf,
        "PF_PASS": pf > 1.30,
        "expectancy": expectancy,
        "EXP_PASS": expectancy > 0,
        "max_dd_pct": max_dd_pct,
        "DD_PASS": max_dd_pct < 15.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "WIN_GT_LOSS": avg_win > avg_loss if n_win > 0 and n_loss > 0 else False,
        "win_rate": wr,
        "n_trades": n,
        "n_win": n_win,
        "n_loss": n_loss,
        "pair_pf": pair_pf,
        "all_pairs_pass": all(v >= 1.0 for v in pair_pf.values()),
    }
    gates["PASS"] = all([
        gates["PF_PASS"], gates["EXP_PASS"], gates["DD_PASS"],
        gates["WIN_GT_LOSS"], gates["all_pairs_pass"],
    ])
    return gates


def main():
    timerange = sys.argv[1] if len(sys.argv) > 1 else "20260618-20260720"
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config-v4.json"

    from freqtrade.configuration import Configuration
    config = Configuration.from_files([config_path])
    config["strategy"] = "PhoenixScalperV4_1"
    config["strategy_path"] = "strategies"

    # Step 1: Load data
    print("Step 1: Loading data...")
    raw_data = load_all_data(TOP10, timerange)
    print(f"  Loaded {len(raw_data)} pairs")

    from freqtrade.resolvers import StrategyResolver
    strategy = StrategyResolver.load_strategy(config)

    # Step 2: Populate indicators (once)
    print("\nStep 2: Computing indicators...")
    indicator_data = {}
    all_entries = []
    for pair in TOP10:
        if pair not in raw_data:
            continue
        df = raw_data[pair].copy()
        df = strategy.populate_indicators(df, {"pair": pair})
        df = strategy.populate_entry_trend(df, {"pair": pair})
        indicator_data[pair] = df

        for direction, col in [("long", "enter_long"), ("short", "enter_short")]:
            if col in df.columns:
                mask = df[col] == 1
                if mask.any():
                    entries = df[mask].copy()
                    entries["__pair"] = pair
                    entries["__direction"] = direction
                    entries["__df_idx"] = df.index[mask]
                    all_entries.append(entries)

    if not all_entries:
        print("No entries found.")
        return

    all_entries = pd.concat(all_entries, ignore_index=True)
    print(f"  Total entries: {len(all_entries)}")

    # Step 3: Label all entries
    print("\nStep 3: Labeling entries...")
    all_trades = evaluate_entries(all_entries, indicator_data)
    print(f"  {len(all_trades)} trades labeled")

    # === FULL PERIOD EVALUATION ===
    print("\n" + "=" * 70)
    print("FULL PERIOD EVALUATION")
    print("=" * 70)
    full_gates = compute_gates(all_trades)
    print(f"  Trades: {full_gates['n_trades']}")
    print(f"  Win rate: {full_gates['win_rate']:.1%}")
    print(f"  PF: {full_gates['PF']:.3f} {'PASS' if full_gates['PF_PASS'] else 'FAIL'}")
    print(f"  Expectancy: {full_gates['expectancy']:.3f}% {'PASS' if full_gates['EXP_PASS'] else 'FAIL'}")
    print(f"  Max DD: {full_gates['max_dd_pct']:.2f}% {'PASS' if full_gates['DD_PASS'] else 'FAIL'}")
    print(f"  Avg win: {full_gates['avg_win']:.3f}% > Avg loss: {full_gates['avg_loss']:.3f}% {'PASS' if full_gates['WIN_GT_LOSS'] else 'FAIL'}")
    print(f"  Per-pair PF >= 1.0: {'PASS' if full_gates['all_pairs_pass'] else 'FAIL'}")
    for p, pf in sorted(full_gates["pair_pf"].items(), key=lambda x: x[1]):
        print(f"    {p}: {pf:.3f}")
    print(f"\n  >>> VERDICT: {'GO' if full_gates['PASS'] else 'NO-GO'} <<<")

    # === WALK-FORWARD WINDOWS ===
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION (5-day windows, 1-day step)")
    print("=" * 70)

    dates = all_trades["date"].sort_values()
    window_start = dates.min()
    window_end = dates.max()
    window_days = 5
    step_days = 1

    window_results = []
    current = window_start
    while current + pd.Timedelta(days=window_days) <= window_end + pd.Timedelta(days=1):
        w_end = current + pd.Timedelta(days=window_days)
        w_trades = all_trades[(all_trades["date"] >= current) & (all_trades["date"] < w_end)]
        if len(w_trades) >= 3:
            wg = compute_gates(w_trades)
            window_results.append({
                "start": current.strftime("%Y-%m-%d"),
                "end": w_end.strftime("%Y-%m-%d"),
                "trades": wg["n_trades"],
                "wr": wg["win_rate"],
                "pf": wg["PF"],
                "exp": wg["expectancy"],
                "pass": wg["PASS"],
            })
            status = "PASS" if wg["PASS"] else "FAIL"
            print(f"  {current.strftime('%m-%d')} to {w_end.strftime('%m-%d')}: "
                  f"{wg['n_trades']} trades, WR={wg['win_rate']:.0%}, "
                  f"PF={wg['PF']:.2f}, EXP={wg['expectancy']:+.3f}% [{status}]")
        current += pd.Timedelta(days=step_days)

    if window_results:
        wf_df = pd.DataFrame(window_results)
        n_pass = wf_df["pass"].sum()
        n_total = len(wf_df)
        print(f"\n  Windows: {n_pass}/{n_total} passed ({100*n_pass/n_total:.0f}%)")
        print(f"  Avg PF across windows: {wf_df['pf'].mean():.3f}")
        print(f"  Avg WR across windows: {wf_df['wr'].mean():.1%}")
        wf_pass = n_pass / n_total >= 0.6
    else:
        wf_pass = False
        print("\n  No valid windows")

    # === FINAL VERDICT ===
    print("\n" + "=" * 70)
    print("FINAL VALIDATION VERDICT")
    print("=" * 70)
    print(f"  Full period PF: {full_gates['PF']:.3f}")
    print(f"  Full period expectancy: {full_gates['expectancy']:.3f}%")
    print(f"  Full period MDD: {full_gates['max_dd_pct']:.2f}%")
    print(f"  Walk-forward pass rate: {n_pass}/{n_total}")
    print(f"  Gates pass: {'YES' if full_gates['PASS'] else 'NO'}")
    print(f"  Walk-forward pass (>60%): {'YES' if wf_pass else 'NO'}")
    print(f"\n  >>> OVERALL: {'CONDITIONAL GO' if full_gates['PASS'] and wf_pass else 'NO-GO'} <<<")

    if not full_gates["PASS"]:
        failed = []
        if not full_gates["PF_PASS"]: failed.append(f"PF={full_gates['PF']:.3f}<1.30")
        if not full_gates["EXP_PASS"]: failed.append(f"EXP={full_gates['expectancy']:.3f}%<=0")
        if not full_gates["DD_PASS"]: failed.append(f"DD={full_gates['max_dd_pct']:.2f}%>=15%")
        if not full_gates["WIN_GT_LOSS"]: failed.append(f"avg_win={full_gates['avg_win']:.3f}<=avg_loss={full_gates['avg_loss']:.3f}")
        if not full_gates["all_pairs_pass"]: failed.append("some pairs PF<1.0")
        print(f"  Failed gates: {', '.join(failed)}")

    # Save
    import json
    out = {
        "full_period": {k: v for k, v in full_gates.items() if k != "pair_pf"},
        "full_period_pair_pf": full_gates.get("pair_pf", {}),
        "walk_forward": window_results if window_results else [],
        "wf_pass_rate": f"{n_pass}/{n_total}" if window_results else "N/A",
        "overall": "CONDITIONAL GO" if full_gates["PASS"] and wf_pass else "NO-GO",
    }
    with open("research/walk_forward_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved research/walk_forward_result.json")


if __name__ == "__main__":
    main()
