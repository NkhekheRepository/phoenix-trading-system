#!/usr/bin/env python3
"""TP/SL optimizer for FLOW+short_bear_momentum subset. Target: 76%+ MC WR, PF>1.5."""
import sys, os, json, zipfile, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.monte_carlo import MonteCarloValidator, TradeResult

BT_DIR = "/freqtrade/user_data/backtest_results"

# Load latest backtest
zips = sorted([f for f in os.listdir(BT_DIR) if f.endswith(".zip")])
with zipfile.ZipFile(os.path.join(BT_DIR, zips[-1])) as z:
    for n in z.namelist():
        if n.endswith(".json") and "_config" not in n:
            raw = z.read(n)
            data = json.loads(re.search(rb'\{.*', raw, re.DOTALL).group())
td = data["strategy"]["PhoenixScalper"]["trades"]

# Filter: FLOW + short_bear_momentum
trades = [t for t in td if t["pair"] == "FLOW/USDT:USDT" and t.get("enter_tag") == "short_bear_momentum"]
print(f"FLOW short_bear trades: {len(trades)}", flush=True)
print(f"Current: WR={sum(1 for t in trades if t['profit_ratio']>0)/len(trades)*100:.1f}%", flush=True)
print(f"  TP wins: {sum(1 for t in trades if t['exit_reason']=='mc1_tp')}, "
      f"SL losses: {sum(1 for t in trades if t['exit_reason']=='stop_loss')}", flush=True)

profits = [t["profit_ratio"]*100 for t in trades]
print(f"  Profit range: {min(profits):+.2f}% to {max(profits):+.2f}%", flush=True)

def simulate(trades, tp_pct, sl_pct):
    results = []
    for t in trades:
        o, lo, hi = t["open_rate"], t["min_rate"], t["max_rate"]
        if t.get("is_short", False):
            tp_hit, sl_hit = lo <= o * (1 - tp_pct), hi >= o * (1 + sl_pct)
        else:
            tp_hit, sl_hit = hi >= o * (1 + tp_pct), lo <= o * (1 - sl_pct)
        if tp_hit and not sl_hit:
            p, w = tp_pct * 100, True
        elif sl_hit and not tp_hit:
            p, w = -sl_pct * 100, False
        elif tp_hit and sl_hit:
            p, w = t["profit_ratio"]*100, t["profit_ratio"] > 0
        else:
            p, w = t["profit_ratio"]*100, t["profit_ratio"] > 0
        results.append(TradeResult(p, w, t["trade_duration"]/60, o, 0, 0, 0, 0.5))
    return results

validator = MonteCarloValidator(n_simulations=5000)

print(f"\n{'='*80}", flush=True)
print("TP/SL MC GRID: FLOW short_bear momentum (28 trades)", flush=True)
print(f"{'='*80}", flush=True)
print(f"{'TP':>8s} {'SL':>8s} {'MC_WR':>8s} {'PF':>8s} {'Ruin':>8s} {'Wins':>6s} {'Losses':>7s}", flush=True)
print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7}", flush=True)

best = []
for tp in [0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08]:
    for sl in [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02]:
        sim = simulate(trades, tp, sl)
        mc = validator.simulate_trade_sequences(sim)
        mw = np.percentile(mc["win_rate"], 50) * 100
        mp = np.percentile(mc["profit_factor"], 50)
        mr = np.mean(mc["ruin_prob"]) * 100
        nw = sum(1 for s in sim if s.win)
        nl = sum(1 for s in sim if not s.win)
        # Only show combos that approach target
        if mw >= 70 or (mp > 1.2 and mr < 20):
            print(f"{tp*100:>7.1f}% {sl*100:>7.1f}% {mw:>7.1f}% {mp:>7.2f} {mr:>7.0f}% {nw:>6d} {nl:>7d}", flush=True)
        best.append((mw, mp, mr, tp, sl))

# Rank: highest MC WR, then PF > 1, then lowest ruin
best.sort(key=lambda x: (-x[0], -x[1], x[2]))
print(f"\n{'='*80}", flush=True)
print("TOP 5 by MC WR:", flush=True)
for mw, mp, mr, tp, sl in best[:5]:
    print(f"  TP={tp*100:.1f}% SL={sl*100:.1f}% → MC_WR={mw:.1f}% PF={mp:.2f} Ruin={mr:.0f}%", flush=True)

# Find best that meets ALL criteria
print(f"\nRanked by (76-WR) + PF<1.5 penalty + ruin penalty:", flush=True)
def score(mw, mp, mr):
    return abs(76 - mw) + max(0, 1.5 - mp) * 5 + max(0, mr - 10) * 0.3

best2 = sorted(best, key=lambda x: score(x[0], x[1], x[2]))
for mw, mp, mr, tp, sl in best2[:5]:
    s = score(mw, mp, mr)
    print(f"  TP={tp*100:.1f}% SL={sl*100:.1f}% → MC_WR={mw:.1f}% PF={mp:.2f} Ruin={mr:.0f}% score={s:.3f}", flush=True)
