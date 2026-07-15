#!/usr/bin/env python3
"""WinRate optimizer — finds TP/SL params that achieve ~86% Monte Carlo win rate."""

import sys, os, json, zipfile, re, itertools, logging
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.monte_carlo import MonteCarloValidator, TradeResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Load latest backtest result
results_dir = "/freqtrade/user_data/backtest_results"
zips = sorted([f for f in os.listdir(results_dir) if f.endswith(".zip")])
if not zips:
    print("No backtest results found.")
    sys.exit(1)

zip_path = os.path.join(results_dir, zips[-1])
print(f"Loading backtest: {zips[-1]}")

with zipfile.ZipFile(zip_path) as z:
    for name in z.namelist():
        if name.endswith(".json") and "_config" not in name:
            raw = z.read(name)
            m = re.search(rb'\{.*', raw, re.DOTALL)
            data = json.loads(m.group())

trades = data["strategy"]["PhoenixScalper"]["trades"]
print(f"Trades loaded: {len(trades)}")

# Extract features for each trade
@dataclass
class Trade:
    profit_ratio: float
    open_rate: float
    close_rate: float
    min_rate: float
    max_rate: float
    leverage: float
    is_short: bool
    duration_min: float

trade_list = []
for t in trades:
    trade_list.append(Trade(
        profit_ratio=t["profit_ratio"],
        open_rate=t["open_rate"],
        close_rate=t["close_rate"],
        min_rate=t.get("min_rate", t["open_rate"] * 0.99),
        max_rate=t.get("max_rate", t["open_rate"] * 1.01),
        leverage=t.get("leverage", 50),
        is_short=t.get("is_short", False),
        duration_min=t.get("trade_duration", 5),
    ))

def simulate_trades(tp_pct: float, sl_pct: float, trades: list) -> list:
    """Apply hypothetical TP/SL rules to all trades and return TradeResult list."""
    results = []
    for t in trades:
        if t.is_short:
            tp_hit = t.min_rate <= t.open_rate * (1 - tp_pct)
            sl_hit = t.max_rate >= t.open_rate * (1 + sl_pct)
        else:
            tp_hit = t.max_rate >= t.open_rate * (1 + tp_pct)
            sl_hit = t.min_rate <= t.open_rate * (1 - sl_pct)

        if tp_hit and not sl_hit:
            profit = tp_pct * 100
            win = True
        elif sl_hit and not tp_hit:
            profit = -sl_pct * 100
            win = False
        elif tp_hit and sl_hit:
            profit = t.profit_ratio * 100
            win = profit > 0
        else:
            profit = t.profit_ratio * 100
            win = profit > 0

        results.append(TradeResult(
            profit_pct=profit, win=win,
            duration_hours=t.duration_min / 60,
            entry_price=t.open_rate, exit_price=0,
            regime=0, kf_direction=0, kf_confidence=0.5,
        ))
    return results

def score_params(tp_pct: float, sl_pct: float, n_sims: int = 2000) -> dict:
    sim_trades = simulate_trades(tp_pct, sl_pct, trade_list)
    validator = MonteCarloValidator(n_simulations=n_sims)
    mc = validator.simulate_trade_sequences(sim_trades)
    mc_wr = np.percentile(mc["win_rate"], 50)
    mc_pf = np.percentile(mc["profit_factor"], 50)
    mc_dd = np.percentile(mc["max_dd"], 95)
    mc_ruin = np.mean(mc["ruin_prob"])
    actual_wr = sum(t.win for t in sim_trades) / len(sim_trades)
    score = abs(0.86 - mc_wr) + max(0, mc_ruin - 0.1) * 0.5
    if mc_pf < 0.8:
        score += (0.8 - mc_pf)
    return {
        "tp": tp_pct, "sl": sl_pct,
        "mc_wr": mc_wr, "mc_pf": mc_pf,
        "mc_dd": mc_dd, "mc_ruin": mc_ruin,
        "actual_wr": actual_wr, "score": score,
    }

# Search grid: wide range first, refine
print("\n=== Phase 1: Coarse grid search ===")
candidates = []
for tp in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
    for sl in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
        r = score_params(tp, sl, n_sims=1000)
        candidates.append(r)
        print(f"  TP={tp*100:.1f}% SL={sl*100:.1f}% → MC_WR={r['mc_wr']*100:.1f}% PF={r['mc_pf']:.2f} DD={r['mc_dd']*100:.0f}% Ruin={r['mc_ruin']*100:.0f}% Score={r['score']:.3f}")

candidates.sort(key=lambda x: x["score"])
print(f"\n=== Top 5 (lowest score) ===")
for r in candidates[:5]:
    print(f"  TP={r['tp']*100:.1f}% SL={r['sl']*100:.1f}% → MC_WR={r['mc_wr']*100:.1f}% PF={r['mc_pf']:.2f} Score={r['score']:.3f}")

# Phase 2: refine top candidates with full MC
print("\n=== Phase 2: Full MC (10k sims) for top 3 ===")
best = candidates[:3]
for r in best:
    r2 = score_params(r["tp"], r["sl"], n_sims=10000)
    print(f"  TP={r2['tp']*100:.1f}% SL={r2['sl']*100:.1f}% → MC_WR={r2['mc_wr']*100:.1f}% PF={r2['mc_pf']:.2f} DD={r2['mc_dd']*100:.0f}% Ruin={r2['mc_ruin']*100:.0f}% Score={r2['score']:.3f}")

# Phase 3: dense search around best region
if best:
    b = best[0]
    print(f"\n=== Phase 3: Dense search around TP={b['tp']*100:.1f}% SL={b['sl']*100:.1f}% ===")
    tps = np.linspace(max(0.002, b["tp"] - 0.005), b["tp"] + 0.005, 5)
    sls = np.linspace(max(0.002, b["sl"] - 0.01), b["sl"] + 0.01, 5)
    for tp, sl in itertools.product(tps, sls):
        r = score_params(round(tp, 4), round(sl, 4), n_sims=2000)
        print(f"  TP={r['tp']*100:.2f}% SL={r['sl']*100:.2f}% → MC_WR={r['mc_wr']*100:.1f}% PF={r['mc_pf']:.2f} DD={r['mc_dd']*100:.0f}% Ruin={r['mc_ruin']*100:.0f}% Score={r['score']:.3f}")
        candidates.append(r)

candidates.sort(key=lambda x: abs(0.86 - x["mc_wr"]) + max(0, x["mc_ruin"] - 0.1))
print(f"\n{'='*70}")
print(f"BEST PARAMETERS FOR 86% MC WIN RATE:")
final = candidates[0]
print(f"  TP target:    {final['tp']*100:.2f}%")
print(f"  SL width:     {final['sl']*100:.2f}%")
print(f"  MC win rate:  {final['mc_wr']*100:.1f}%")
print(f"  MC profit factor: {final['mc_pf']:.2f}")
print(f"  MC max DD (p95):  {final['mc_dd']*100:.0f}%")
print(f"  MC ruin prob:     {final['mc_ruin']*100:.0f}%")
print(f"  Actual WR:    {final['actual_wr']*100:.1f}%")
print(f"\n  → In config:")
print(f'  "tp_target": {final["tp"]},')
print(f'  "sl_min": {final["sl"]},')
print(f'  "sl_max": {final["sl"]},')
print(f'  "minimal_roi": {{"0": {final["tp"]*100}}},')
print(f'  "custom_stoploss": -{final["sl"]*100}')
print(f"{'='*70}")
