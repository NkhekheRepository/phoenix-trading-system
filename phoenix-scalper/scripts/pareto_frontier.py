#!/usr/bin/env python3
"""Pareto frontier: max aligned WR vs trades/day with 50x leverage."""
import sys, os, json, zipfile, re
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.monte_carlo import MonteCarloValidator, TradeResult

BT_DIR = "/freqtrade/user_data/backtest_results"

zips = sorted([f for f in os.listdir(BT_DIR) if f.endswith(".zip")])
for zf in reversed(zips):
    with zipfile.ZipFile(os.path.join(BT_DIR, zf)) as z:
        for n in z.namelist():
            if n.endswith(".json") and "_config" not in n:
                raw = z.read(n)
                td = json.loads(re.search(rb'\{.*', raw, re.DOTALL).group())["strategy"]["PhoenixScalper"]["trades"]
                up = set(t.get("pair","") for t in td)
                if len(td) > 150 and len(up) > 1:
                    break
        else:
            continue
        break

combos = defaultdict(list)
for t in td:
    ckey = (t["pair"], t.get("enter_tag",""), "short" if t.get("is_short",False) else "long")
    combos[ckey].append(t)

validator = MonteCarloValidator(n_simulations=5000)
results = []
for ckey, sub in combos.items():
    if len(sub) < 3:
        continue
    awr = sum(1 for t in sub if t["profit_ratio"]>0)/len(sub)*100
    tr = [TradeResult(t["profit_ratio"]*100, t["profit_ratio"]>0, t["trade_duration"]/60, 0, 0, 0, 0, 0.5) for t in sub]
    mc = validator.simulate_trade_sequences(tr)
    mw = float(np.percentile(mc["win_rate"], 50)*100)
    mp = float(np.percentile(mc["profit_factor"], 50))
    mr = float(np.mean(mc["ruin_prob"])*100)
    results.append((*ckey, len(sub), round(awr,1), round(mw,1), round(mp,2), round(mr,1)))

results.sort(key=lambda x: -x[4])
print(f"{'Pair':<20s} {'Sig':<25s} {'Dir':<5s} {'N':>3s} {'WR':>5s} {'MC_WR':>6s} {'PF':>6s} {'Ruin':>5s}")
print("-"*80)
for r in results:
    p,sig,d,n,awr,mw,mp,mr = r
    print(f"{p:<20s} {sig:<25s} {d:<5s} {n:>3d} {awr:>4.1f}% {mw:>5.1f}% {mp:>5.2f} {mr:>4.0f}%")

DAYS = 59
bw_by_target = {}
for target_tpd in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    target_n = int(target_tpd * DAYS)
    selected = []
    cum = 0
    for r in results:
        if cum >= target_n:
            break
        selected.append(r)
        cum += r[3]
    if cum <= 0:
        continue
    bw = sum(r[4]*r[3] for r in selected) / sum(r[3] for r in selected)
    wr = sum(r[3] for r in selected)
    bw_by_target[target_tpd] = (cum, round(bw, 1), wr)

print(f"\n{'='*60}")
print(f"{'Target tpd':>12s} {'Unique tr':>10s} {'Blended WR':>12s}")
print(f"{'-'*12} {'-'*10} {'-'*12}")
for tpd in sorted(bw_by_target.keys()):
    cum, bw, wr = bw_by_target[tpd]
    print(f"{tpd:>11.1f}/day {cum:>9d} {bw:>10.1f}%")

print(f"\n{'='*60}")
print("HONEST ASSESSMENT (50x, 9 pairs):")
tier1 = [r for r in results if r[4] >= 60]  # >= 60% WR
tier2 = [r for r in results if 45 <= r[4] < 60]
tier3 = [r for r in results if r[4] < 45]
print(f"  Tier 1 (>=60% WR): {sum(r[3] for r in tier1)} trades/59d = {sum(r[3] for r in tier1)/DAYS:.1f}/day")
for r in tier1:
    print(f"    {r[0]:20s} {r[1]:25s} {r[2]:5s} N={r[3]:3d} WR={r[4]:.1f}%")
print(f"  Tier 2 (45-59%): {sum(r[3] for r in tier2)} trades/59d = {sum(r[3] for r in tier2)/DAYS:.1f}/day")
print(f"  Tier 3 (<45%): {sum(r[3] for r in tier3)} trades/59d = {sum(r[3] for r in tier3)/DAYS:.1f}/day")
total_trades = sum(r[3] for r in results)
avg_wr = sum(r[4]*r[3] for r in results) / total_trades
print(f"\n  All unique: {total_trades} trades/59d = {total_trades/DAYS:.1f}/day at {avg_wr:.1f}% WR")
