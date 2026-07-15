#!/usr/bin/env python3
"""Combinatorial MC permutation search: find optimal {pair,signal,gate} portfolio for 8-10 trades/day at 74% daily WR, 50x."""
import sys, os, json, zipfile, re
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.hmm_regime import RegimeDetector
from ml.monte_carlo import MonteCarloValidator, TradeResult

DATA_DIR = "/freqtrade/user_data/data/binance/futures"
BT_DIR = "/freqtrade/user_data/backtest_results"
OUT = "/freqtrade/user_data/combinatorial_results.json"

np.random.seed(42)

def load_trades_and_hmm():
    zips = sorted([f for f in os.listdir(BT_DIR) if f.endswith(".zip")])
    # Use the 9-pair backtest (has 200+ trades from multiple pairs)
    for zf in reversed(zips):
        with zipfile.ZipFile(os.path.join(BT_DIR, zf)) as z:
            for n in z.namelist():
                if n.endswith(".json") and "_config" not in n:
                    raw = z.read(n)
                    trades = json.loads(re.search(rb'\{.*', raw, re.DOTALL).group())["strategy"]["PhoenixScalper"]["trades"]
                    # Check trades span multiple pairs (unique pair count > 1)
                    unique_pairs = set(t.get('pair','') for t in trades)
                    if len(trades) > 150 and len(unique_pairs) > 1:
                        print(f"Using: {zf} ({len(trades)} trades, {len(unique_pairs)} pairs)", flush=True)
                        return trades
    print(f"WARN: no multi-pair backtest found, falling back to latest", flush=True)

def compute_hmm(df):
    returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    vol = returns.rolling(10).std().fillna(0)
    vr = df['volume'] / (df['volume'].ewm(span=10).mean() + 1e-10)
    vc = vr.pct_change().fillna(0)
    obs = np.column_stack([returns.values, vol.values, vc.values])
    n = len(obs)
    obs_train = obs[np.linspace(0,n-1,1000,dtype=int)] if n>1000 else obs
    hmm = RegimeDetector(n_states=3, n_iter=5)
    hmm.fit(obs_train)
    probs = hmm.predict_regime_probs_fast(obs)
    stability = np.array([1.0 - sum(probs[i]**2) for i in range(n)])
    return {'regime': hmm.predict_regime_fast(obs), 'p_bull': probs[:,0], 'p_bear': probs[:,2], 'stability': stability}

def get_records():
    td = load_trades_and_hmm()
    pairs = defaultdict(list)
    for t in td:
        pairs[t["pair"].replace("/","_").replace(":","_")].append(t)
    recs = []
    for pk, trades in pairs.items():
        fp = os.path.join(DATA_DIR, f"{pk}-5m-futures.feather")
        if not os.path.exists(fp): continue
        df = __import__('pandas').read_feather(fp)
        dates = df['date'].values
        hmm = compute_hmm(df)
        for t in trades:
            idx = min(np.argmin(np.abs(dates - __import__('pandas').Timestamp(t["open_date"]).to_datetime64())), len(hmm['regime'])-1)
            recs.append({
                'pair': t['pair'], 'sig': t.get('enter_tag',''), 'short': t.get('is_short',False),
                'profit_pct': t['profit_ratio']*100, 'win': t['profit_ratio']>0,
                'duration_h': t['trade_duration']/60.0,
                'open_rate': t['open_rate'], 'min_rate': t.get('min_rate', t['open_rate']*0.99),
                'max_rate': t.get('max_rate', t['open_rate']*1.01),
                'regime': int(hmm['regime'][idx]),
                'p_bull': float(hmm['p_bull'][idx]), 'p_bear': float(hmm['p_bear'][idx]),
                'stability': float(hmm['stability'][idx]),
            })
    return recs

def mc_stats(subset, n_sims=5000):
    if len(subset) < 5: return None
    validator = MonteCarloValidator(n_simulations=n_sims)
    tr = [TradeResult(r['profit_pct'], r['win'], r['duration_h'], 0, 0, r['regime'], 0, 0.5) for r in subset]
    mc = validator.simulate_trade_sequences(tr)
    return {
        'n': len(subset),
        'awr': sum(1 for r in subset if r['win'])/len(subset)*100,
        'mc_wr': float(np.percentile(mc['win_rate'], 50)*100),
        'mc_pf': float(np.percentile(mc['profit_factor'], 50)),
        'mc_dd': float(np.percentile(mc['max_dd'], 95)*100),
        'mc_ruin': float(np.mean(mc['ruin_prob'])*100),
    }

# Stage 1: Load data
print("Loading data + HMM...", flush=True)
recs = get_records()
print(f"Total records: {len(recs)}", flush=True)

# Stage 2: All combos
print("\n=== STAGE 2: ALL COMBOS MC SCREEN ===", flush=True)
combo_groups = defaultdict(list)
for r in recs:
    ckey = (r['pair'], r['sig'], 'short' if r['short'] else 'long')
    combo_groups[ckey].append(r)

results = []
for ckey, sub in combo_groups.items():
    if len(sub) < 5: continue
    st = mc_stats(sub)
    if st:
        results.append((*ckey, st['n'], st['awr'], st['mc_wr'], st['mc_pf'], st['mc_dd'], st['mc_ruin']))

results.sort(key=lambda x: -x[4])  # sort by actual WR

print(f"{'Pair':<20s} {'Sig':<25s} {'Dir':<6s} {'N':>4s} {'AWR':>6s} {'MC_WR':>6s} {'PF':>5s} {'DD':>5s} {'Ruin':>5s}")
print("-"*85)
for p, sig, d, n, awr, mw, mp, dd, ru in results:
    print(f"{p:<20s} {sig:<25s} {d:<6s} {n:>4d} {awr:>5.1f}% {mw:>5.1f}% {mp:>4.2f} {dd:>4.0f}% {ru:>4.0f}%")

# Stage 3: HMM gate permutation on top combos (MC_WR > 37%)
print("\n=== STAGE 3: HMM GATE PERMUTATION ===", flush=True)
top_results = [r for r in results if r[5] >= 37]  # MC_WR >= 37%
gates = []

# Define gates as filter functions
gate_defs = []

# Stability gates
for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    gate_defs.append((f"stab<{thr}", lambda r, x=thr: r['stability'] < x))

# p_bear gates for shorts
for thr in [0.3, 0.5, 0.7]:
    gate_defs.append((f"p_bear>={thr}", lambda r, x=thr: r['p_bear'] >= x))

# p_bull gates for longs
for thr in [0.3, 0.5, 0.7]:
    gate_defs.append((f"p_bull>={thr}", lambda r, x=thr: r['p_bull'] >= x))

# Direction-consistency: short only if p_bear > p_bull, long only if p_bull > p_bear
gate_defs.append(("dir_consistent", lambda r: (r['short'] and r['p_bear'] > r['p_bull']) or (not r['short'] and r['p_bull'] > r['p_bear'])))

# Combined: stability + direction
gate_defs.append(("stab<0.4+dir", lambda r: r['stability'] < 0.4 and ((r['short'] and r['p_bear'] > r['p_bull']) or (not r['short'] and r['p_bull'] > r['p_bear']))))
gate_defs.append(("stab<0.3+dir", lambda r: r['stability'] < 0.3 and ((r['short'] and r['p_bear'] > r['p_bull']) or (not r['short'] and r['p_bull'] > r['p_bear']))))

# Trend-regime: regime=0 + long, regime=2 + short
gate_defs.append(("reg0_bull+long", lambda r: r['regime'] == 0 and not r['short']))
gate_defs.append(("reg2_bear+short", lambda r: r['regime'] == 2 and r['short']))

# No gate (baseline)
gate_defs.insert(0, ("none", lambda r: True))

gated_results = []
validator = MonteCarloValidator(n_simulations=5000)

# For each top combo, try all gates
for pair, sig, d, n, awr, mw, mp, dd, ru in top_results:
    base_sub = combo_groups[(pair, sig, d)]
    
    for gname, gfn in gate_defs:
        sub = [r for r in base_sub if gfn(r)]
        if len(sub) < 5: continue
        s = mc_stats(sub, 5000)
        if s:
            gated_results.append({
                'combo': f"{pair}/{sig}/{d}",
                'gate': gname,
                'n': s['n'],
                'awr': round(s['awr'], 1),
                'mc_wr': round(s['mc_wr'], 1),
                'mc_pf': round(s['mc_pf'], 2),
                'mc_dd': round(s['mc_dd'], 1),
                'mc_ruin': round(s['mc_ruin'], 1),
            })

gated_results.sort(key=lambda x: -x['mc_wr'])
print(f"{'Combo':<45s} {'Gate':<18s} {'N':>4s} {'AWR':>6s} {'MC_WR':>6s} {'PF':>5s} {'DD':>5s}", flush=True)
print("-"*95, flush=True)
for r in gated_results[:40]:
    print(f"{r['combo']:<45s} {r['gate']:<18s} {r['n']:>4d} {r['awr']:>5.1f}% {r['mc_wr']:>5.1f}% {r['mc_pf']:>4.2f} {r['mc_dd']:>4.0f}%", flush=True)

# Stage 4: Greedy portfolio builder
print("\n=== STAGE 4: GREEDY PORTFOLIO CONSTRUCTION (target: 8-10 trades/day, 50x) ===", flush=True)

# Sort by MC_WR descending
viable = [r for r in gated_results if r['mc_wr'] >= 35 and r['mc_pf'] >= 0.7 and r['mc_ruin'] < 80]
viable.sort(key=lambda x: -x['mc_wr'])

DAYS = 59
DAILY_TARGET_MIN = 8 * DAYS  # 472
DAILY_TARGET_MAX = 10 * DAYS  # 590

selected = []
cum_n = 0
for r in viable:
    if cum_n >= DAILY_TARGET_MAX:
        break
    selected.append(r)
    cum_n += r['n']
    # Prune: if adding this drops blended WR too low, skip
    # Blended WR = weighted avg of selected MC_WR
    blended_wr = sum(x['mc_wr'] * x['n'] for x in selected) / sum(x['n'] for x in selected)
    print(f"  Add {r['combo']:40s} gate={r['gate']:<18s} N={r['n']:>4d} WR={r['mc_wr']:>5.1f}% → cum={cum_n:>4d} blended_WR={blended_wr:.1f}%", flush=True)

print(f"\n  Final portfolio: {len(selected)} combos, {cum_n} trades ({cum_n/DAYS:.1f}/day), blended WR={sum(x['mc_wr']*x['n'] for x in selected)/sum(x['n'] for x in selected):.1f}%", flush=True)

# Stage 5: Daily WR via day-stratified MC
print("\n=== STAGE 5: DAILY WR SIMULATION ===", flush=True)
# For each day, simulate trades from each combo
# Use actual trade timestamps to sample
np.random.seed(42)
daily_results = []
for _ in range(5000):
    # Pick trades per combo proportionate to their daily freq
    daily_trades = []
    for r in selected:
        combo_sub = [x for x in recs if f"{x['pair']}/{x['sig']}/{'short' if x['short'] else 'long'}" == r['combo']]
        # Apply gate
        gfn = None
        for gn, gf in gate_defs:
            if gn == r['gate']:
                gfn = gf
                break
        if gfn:
            combo_sub = [x for x in combo_sub if gfn(x)]
        # Draw random trades
        if len(combo_sub) > 0:
            n_daily = max(1, round(r['n'] / DAYS))
            for _ in range(n_daily):
                idx = np.random.randint(0, len(combo_sub))
                daily_trades.append(combo_sub[idx])
    daily_wr = sum(1 for t in daily_trades if t['win']) / len(daily_trades) * 100 if daily_trades else 0
    daily_results.append(daily_wr)

print(f"  MC Daily WR (5000 sims): p10={np.percentile(daily_results,10):.1f}% p50={np.percentile(daily_results,50):.1f}% p90={np.percentile(daily_results,90):.1f}%", flush=True)

# Save (simple version)
out = {
    'n_trades': len(recs),
    'results_by_combo': [{'pair':p, 'sig':sig, 'dir':d, 'n':n, 'awr':awr, 'mc_wr':mw, 'mc_pf':mp, 'mc_dd':dd, 'mc_ruin':ru}
                          for p,sig,d,n,awr,mw,mp,dd,ru in results],
    'gated': gated_results,
    'selected': selected,
    'daily_mc': {
        'p10': round(float(np.percentile(daily_results,10)),1),
        'p50': round(float(np.percentile(daily_results,50)),1),
        'p90': round(float(np.percentile(daily_results,90)),1),
    }
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nFull results saved to {OUT}", flush=True)
