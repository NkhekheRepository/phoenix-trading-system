#!/usr/bin/env python3
"""Verify TP/SL on FLOW short_bear — checking price movement range + optimal params."""
import sys, os, json, zipfile, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.monte_carlo import MonteCarloValidator, TradeResult

BT_DIR = "/freqtrade/user_data/backtest_results"
zips = sorted([f for f in os.listdir(BT_DIR) if f.endswith(".zip")])
with zipfile.ZipFile(os.path.join(BT_DIR, zips[-1])) as z:
    for n in z.namelist():
        if n.endswith(".json") and "_config" not in n:
            raw = z.read(n)
            data = json.loads(re.search(rb'\{.*', raw, re.DOTALL).group())
td = data["strategy"]["PhoenixScalper"]["trades"]
trades = [t for t in td if t["pair"] == "FLOW/USDT:USDT" and t.get("enter_tag") == "short_bear_momentum"]

print("=== Price Movement Analysis (FLOW short_bear, 50x) ===", flush=True)
for i, t in enumerate(trades):
    lev = t.get("leverage", 50)
    price_move_up = (t["max_rate"] - t["open_rate"]) / t["open_rate"] * 100
    price_move_down = (t["open_rate"] - t["min_rate"]) / t["open_rate"] * 100
    pnl = t["profit_ratio"] * 100
    tag = "WIN " if t["profit_ratio"] > 0 else "LOSS"
    print(f"  {tag} | price up={price_move_up:>+6.2f}% dn={price_move_down:>+6.2f}% "
          f"→ wallet: {pnl:>+7.2f}% (50x ratio={pnl/(price_move_up if price_move_up>abs(price_move_down) else -price_move_down):.1f}x)", flush=True)

print(f"\n=== TP/SL at price % (50x, wallet P&L in parens) ===", flush=True)

def simulate(trades, tp_price_pct, sl_price_pct):
    """tp/sl as PRICE percentages (not wallet). For shorts, tp=price drop, sl=price rise."""
    results = []
    for t in trades:
        o, lo, hi = t["open_rate"], t["min_rate"], t["max_rate"]
        lev = t.get("leverage", 50)
        if t.get("is_short", False):
            tp_hit = lo <= o * (1 - tp_price_pct)
            sl_hit = hi >= o * (1 + sl_price_pct)
        else:
            tp_hit = hi >= o * (1 + tp_price_pct)
            sl_hit = lo <= o * (1 - sl_price_pct)
        if tp_hit and not sl_hit:
            wallet_pnl = tp_price_pct * lev * 100  # wallet % = price% * leverage
            results.append(TradeResult(wallet_pnl, True, t["trade_duration"]/60, o, 0, 0, 0, 0.5))
        elif sl_hit and not tp_hit:
            wallet_pnl = -sl_price_pct * lev * 100
            results.append(TradeResult(wallet_pnl, False, t["trade_duration"]/60, o, 0, 0, 0, 0.5))
        elif tp_hit and sl_hit:
            results.append(TradeResult(t["profit_ratio"]*100, t["profit_ratio"]>0, t["trade_duration"]/60, o, 0, 0, 0, 0.5))
        else:
            results.append(TradeResult(t["profit_ratio"]*100, t["profit_ratio"]>0, t["trade_duration"]/60, o, 0, 0, 0, 0.5))
    return results

validator = MonteCarloValidator(n_simulations=5000)
print(f"{'TP_price':>10s} {'SL_price':>10s} {'TP_wallet':>12s} {'SL_wallet':>12s} {'MC_WR':>8s} {'PF':>8s} {'Ruin':>8s} {'Wins':>6s}", flush=True)
print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*6}", flush=True)

results = []
for tp_bp in [1, 2, 3, 5, 7, 10, 14, 20, 25, 30, 40]:  # price basis points
    tp_pct = tp_bp / 10000  # convert bps to decimal
    for sl_bp in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]:
        sl_pct = sl_bp / 10000
        sim = simulate(trades, tp_pct, sl_pct)
        mc = validator.simulate_trade_sequences(sim)
        mw = np.percentile(mc["win_rate"], 50) * 100
        mp = np.percentile(mc["profit_factor"], 50)
        mr = np.mean(mc["ruin_prob"]) * 100
        nw = sum(1 for s in sim if s.win)
        tp_wallet = tp_pct * 50 * 100
        sl_wallet = sl_pct * 50 * 100
        results.append((mw, mp, mr, tp_pct, sl_pct, tp_wallet, sl_wallet, nw))

results.sort(key=lambda x: -x[0])
for mw, mp, mr, tp, sl, tp_w, sl_w, nw in results[:15]:
    print(f"{tp*100:>9.3f}% {sl*100:>9.3f}% {tp_w:>11.1f}% {sl_w:>11.1f}% {mw:>7.1f}% {mp:>7.2f} {mr:>7.0f}% {nw:>6d}", flush=True)

# Also score by target: 76% WR preferred, PF>1.5 required
print(f"\n=== Ranked by (76-WR) + PF<1.5 penalty ===", flush=True)
def score(mw, mp):
    return abs(76 - mw) + max(0, 1.5 - mp) * 5
results.sort(key=lambda x: score(x[0], x[1]))
for mw, mp, mr, tp, sl, tp_w, sl_w, nw in results[:10]:
    s = score(mw, mp)
    print(f"  TP={tp*100:.3f}% SL={sl*100:.3f}% → WR={tp_w:.0f}%/{sl_w:.0f}% wallet "
          f"MC_WR={mw:.1f}% PF={mp:.2f} Ruin={mr:.0f}% score={s:.3f}", flush=True)
