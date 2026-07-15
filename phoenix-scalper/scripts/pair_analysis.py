#!/usr/bin/env python3
"""Per-pair performance breakdown + MC validation."""
import sys, os, json, zipfile, re
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.monte_carlo import MonteCarloValidator, TradeResult

DATA_DIR = "/freqtrade/user_data/data/binance/futures"
BT_DIR = "/freqtrade/user_data/backtest_results"

zips = sorted([f for f in os.listdir(BT_DIR) if f.endswith(".zip")])
with zipfile.ZipFile(os.path.join(BT_DIR, zips[-1])) as z:
    for n in z.namelist():
        if n.endswith(".json") and "_config" not in n:
            raw = z.read(n)
            data = json.loads(re.search(rb'\{.*', raw, re.DOTALL).group())
td = data["strategy"]["PhoenixScalper"]["trades"]

pairs = defaultdict(list)
for t in td:
    pairs[t["pair"]].append(t)

print(f"{'Pair':<25s} {'N':>4s} {'Wins':>5s} {'Losses':>6s} {'WR':>6s} {'AvgProfit':>10s} {'Stdev':>8s} {'PF':>7s}")
print("-" * 75)
for pk in sorted(pairs.keys()):
    sub = pairs[pk]
    wins = sum(1 for t in sub if t["profit_ratio"] > 0)
    n = len(sub)
    wr = wins / n * 100
    profits = [t["profit_ratio"] * 100 for t in sub]
    avg_p = float(np.mean(profits))
    std_p = float(np.std(profits))
    tp = sum(p for p in profits if p > 0)
    tl = abs(sum(p for p in profits if p < 0))
    pf = tp / (tl + 1e-10)
    print(f"{pk:<25s} {n:>4d} {wins:>5d} {n-wins:>6d} {wr:>5.1f}% {avg_p:>+8.2f}% {std_p:>6.2f}% {pf:>6.2f}")

print()
print(f"{'Pair':<25s} {'N':>4s} {'MC_WR':>7s} {'PF':>6s} {'Ruin':>6s}")
print("-" * 50)
validator = MonteCarloValidator(n_simulations=3000)
for pk in sorted(pairs.keys()):
    sub = pairs[pk]
    if len(sub) < 10:
        continue
    tr = [TradeResult(t["profit_ratio"] * 100, t["profit_ratio"] > 0,
                      t["trade_duration"] / 60, 0, 0, 0, 0, 0.5) for t in sub]
    mc = validator.simulate_trade_sequences(tr)
    mw = float(np.percentile(mc["win_rate"], 50) * 100)
    mp = float(np.percentile(mc["profit_factor"], 50))
    mr = float(np.mean(mc["ruin_prob"]) * 100)
    print(f"{pk:<25s} {len(sub):>4d} {mw:>6.1f}% {mp:>5.2f} {mr:>5.0f}%")
