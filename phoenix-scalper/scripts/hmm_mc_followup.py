#!/usr/bin/env python3
"""Focused follow-up: stability gate + TP/SL + FLOW pair — can we hit 74%?"""
import sys, os, json, zipfile, re, logging
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.hmm_regime import RegimeDetector
from ml.monte_carlo import MonteCarloValidator, TradeResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
DATA_DIR = "/freqtrade/user_data/data/binance/futures"
BACKTEST_DIR = "/freqtrade/user_data/backtest_results"
np.random.seed(42)


def load_trades():
    zips = sorted([f for f in os.listdir(BACKTEST_DIR) if f.endswith(".zip")])
    with zipfile.ZipFile(os.path.join(BACKTEST_DIR, zips[-1])) as z:
        for name in z.namelist():
            if name.endswith(".json") and "_config" not in name:
                raw = z.read(name)
                return json.loads(re.search(rb'\{.*', raw, re.DOTALL).group())["strategy"]["PhoenixScalper"]["trades"]


def compute_hmm_for_pair(df):
    returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    vol = returns.rolling(10).std().fillna(0)
    vr = df['volume'] / (df['volume'].ewm(span=10).mean() + 1e-10)
    vc = vr.pct_change().fillna(0)
    obs = np.column_stack([returns.values, vol.values, vc.values])
    n = len(obs)
    obs_train = obs[np.linspace(0, n-1, 1000, dtype=int)] if n > 1000 else obs
    hmm = RegimeDetector(n_states=3, n_iter=5)
    hmm.fit(obs_train)
    probs = hmm.predict_regime_probs_fast(obs)
    stability = np.array([1.0 - sum(probs[i]**2) for i in range(n)])
    return {'regime': hmm.predict_regime_fast(obs), 'p_bull': probs[:, 0], 'p_bear': probs[:, 2],
            'stability': stability}


def make_records():
    td = load_trades()
    pairs = defaultdict(list)
    for t in td:
        pairs[t["pair"].replace("/", "_").replace(":", "_")].append(t)
    recs = []
    for pk, trades in pairs.items():
        fp = os.path.join(DATA_DIR, f"{pk}-5m-futures.feather")
        if not os.path.exists(fp):
            continue
        df = __import__('pandas').read_feather(fp)
        dates = df['date'].values
        hmm = compute_hmm_for_pair(df)
        for t in trades:
            idx = min(np.argmin(np.abs(dates - __import__('pandas').Timestamp(t["open_date"]).to_datetime64())),
                      len(hmm['regime']) - 1)
            recs.append({
                'profit_pct': t["profit_ratio"] * 100, 'win': t["profit_ratio"] > 0,
                'duration_h': t["trade_duration"] / 60.0,
                'is_short': t.get("is_short", False), 'enter_tag': t.get("enter_tag", ""),
                'pair': pk, 'open_rate': t["open_rate"],
                'min_rate': t.get("min_rate", t["open_rate"] * 0.99),
                'max_rate': t.get("max_rate", t["open_rate"] * 1.01),
                'regime': int(hmm['regime'][idx]),
                'p_bull': float(hmm['p_bull'][idx]), 'p_bear': float(hmm['p_bear'][idx]),
                'stability': float(hmm['stability'][idx]),
            })
    return recs


def do_tp_sl(subset, tp, sl):
    out = []
    for r in subset:
        o, lo, hi = r['open_rate'], r['min_rate'], r['max_rate']
        if r['is_short']:
            tp_hit, sl_hit = lo <= o * (1 - tp), hi >= o * (1 + sl)
        else:
            tp_hit, sl_hit = hi >= o * (1 + tp), lo <= o * (1 - sl)
        if tp_hit and not sl_hit:
            p, w = tp * 100, True
        elif sl_hit and not tp_hit:
            p, w = -sl * 100, False
        else:
            p, w = r['profit_pct'], r['profit_pct'] > 0
        out.append(TradeResult(p, w, r['duration_h'], o, 0, r['regime'], 0, 0.5))
    return out


recs = make_records()
print(f"Total: {len(recs)} trades\n", flush=True)

# 1) Deep stability analysis
print("=" * 70, flush=True)
print("DEEP DIVE: stability gate", flush=True)
print("=" * 70, flush=True)
stb_vals = sorted(set(r['stability'] for r in recs))
print(f"Stability values: {[round(v, 3) for v in stb_vals]}", flush=True)

for threshold in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50]:
    sub = [r for r in recs if r['stability'] < threshold and not r['is_short']]
    if len(sub) < 5:
        continue
    wr = sum(1 for r in sub if r['win']) / len(sub) * 100
    tags = defaultdict(int)
    for r in sub:
        tags[r['enter_tag']] += 1
    tag_info = ', '.join(f"{t}={c}" for t, c in sorted(tags.items()))
    print(f"  stability<{threshold:.2f} + long: N={len(sub):>3d} WR={wr:>5.1f}% tags: {tag_info}", flush=True)

# 2) TP/SL grid on best stability subset
st24 = [r for r in recs if r['stability'] < 0.2 and not r['is_short']]
print(f"\nstability<0.2+long has {len(st24)} trades, base WR={sum(1 for r in st24 if r['win'])/len(st24)*100:.1f}%", flush=True)

validator = MonteCarloValidator(n_simulations=5000)
print("\nTP/SL grid on stability<0.2+long:", flush=True)
best = None
best_s = float('inf')
results = []
for tp in [0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
    for sl in [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015]:
        sim = do_tp_sl(st24, tp, sl)
        mc = validator.simulate_trade_sequences(sim)
        mw = np.percentile(mc['win_rate'], 50) * 100
        mp = np.percentile(mc['profit_factor'], 50)
        mr = np.mean(mc['ruin_prob']) * 100
        s = abs(74 - mw) + max(0, mr - 10) * 0.3
        if s < best_s:
            best_s = s
            best = (tp, sl, mw, mp, mr)
        results.append((tp, sl, mw, mp, mr))
        if s < 30:  # Only show promising ones
            print(f"  TP={tp*100:.1f}% SL={sl*100:.1f}% → MC_WR={mw:.1f}% PF={mp:.2f} Ruin={mr:.0f}%", flush=True)

if best:
    print(f"\n  BEST: TP={best[0]*100:.1f}% SL={best[1]*100:.1f}% → MC_WR={best[2]:.1f}% PF={best[3]:.2f} Ruin={best[4]:.0f}%", flush=True)

# 3) FLOW pair deep dive
flow_trades = [r for r in recs if r['pair'] == 'FLOW_USDT_USDT']
print(f"\n{'='*70}", flush=True)
print(f"FLOW PAIR DEEP DIVE ({len(flow_trades)} trades)", flush=True)
print('=' * 70, flush=True)
for r in flow_trades:
    tag = f"{'SHORT' if r['is_short'] else 'LONG '} {r['enter_tag']:25s}"
    result = "WIN" if r['win'] else "LOSS"
    print(f"  {tag} profit={r['profit_pct']:>+7.2f}% stability={r['stability']:.3f} reg={r['regime']} {result}", flush=True)

# FLOW MC validation
print(f"\nFLOW MC (5000 sims):", flush=True)
flow_tr = [TradeResult(r['profit_pct'], r['win'], r['duration_h'], 0, 0, r['regime'], 0, 0.5) for r in flow_trades]
mc = MonteCarloValidator(n_simulations=5000).simulate_trade_sequences(flow_tr)
print(f"  MC_WR={np.percentile(mc['win_rate'], 50)*100:.1f}% PF={np.percentile(mc['profit_factor'], 50):.2f}", flush=True)
print(f"  MC_DD={np.percentile(mc['max_dd'], 95)*100:.1f}% Ruin={np.mean(mc['ruin_prob'])*100:.0f}%", flush=True)

# 4) Combined filter: stability<0.2 + FLOW
combined = [r for r in recs if r['stability'] < 0.2 and not r['is_short'] and r['pair'] == 'FLOW_USDT_USDT']
print(f"\n{'='*70}", flush=True)
print(f"COMBINED: stability<0.2+long+FLOW ({len(combined)} trades)", flush=True)
print('=' * 70, flush=True)
for r in combined:
    tag = f"{r['enter_tag']:25s}"
    print(f"  {tag} profit={r['profit_pct']:>+7.2f}% stability={r['stability']:.3f} {'WIN' if r['win'] else 'LOSS'}", flush=True)

# 5) Try reducing leverage via profit scaling
print(f"\n{'='*70}", flush=True)
print("LEVERAGE REDUCTION SIMULATION", flush=True)
print("=" * 70, flush=True)
print("At 50x: 5% wallet fee per round trip", flush=True)
print("At 10x: 1% wallet fee per round trip", flush=True)
print("At 5x:  0.5% wallet fee per round trip\n", flush=True)

for label, subset in [("all", recs), ("stability<0.2+long", st24)]:
    for lev in [5, 10, 20, 30, 50]:
        # Scale profit to lower leverage: profit_ratio * (lev/50)
        # Add back the extra fee: at 50x, fee = 5% = 0.05 * stake
        # At target lev, fee = 0.05 * 2 * lev / 100 = 0.001 * lev
        fee_50x = 0.05  # 5% wallet
        fee_target = 0.001 * lev  # 0.1% per leverage unit
        
        scaled = []
        for r in subset:
            # Original profit at 50x leverage (includes 5% fee drag)
            # At lower leverage, profit scales linearly but fee drag changes
            base_return = r['profit_pct'] / 100  # already includes 50x fee
            # Approximate: subtract 50x fee, add target fee, scale by lev/50
            adjusted = (base_return + fee_50x - fee_target) * (lev / 50)
            scaled.append(TradeResult(adjusted * 100, adjusted > 0, r['duration_h'],
                                      0, 0, r['regime'], 0, 0.5))
        
        mc = MonteCarloValidator(n_simulations=3000).simulate_trade_sequences(scaled)
        mw = np.percentile(mc['win_rate'], 50) * 100
        mp = np.percentile(mc['profit_factor'], 50)
        mr = np.mean(mc['ruin_prob']) * 100
        print(f"  {label:25s} {lev:2d}x: MC_WR={mw:>5.1f}% PF={mp:>5.2f} Ruin={mr:>3.0f}%", flush=True)
