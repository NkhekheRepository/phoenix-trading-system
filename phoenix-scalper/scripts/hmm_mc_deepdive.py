#!/usr/bin/env python3
"""
HMM MC Deep-Dive: find ANY trade subset + TP/SL combo hitting 74% WR.
Explores regime×direction×tag×TP/SL using 5000-run MC validation.
"""
import sys, os, json, zipfile, re, logging, itertools
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.hmm_regime import RegimeDetector
from ml.monte_carlo import MonteCarloValidator, TradeResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
DATA_DIR = "/freqtrade/user_data/data/binance/futures"
BACKTEST_DIR = "/freqtrade/user_data/backtest_results"


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
    regime = hmm.predict_regime_fast(obs)
    probs = hmm.predict_regime_probs_fast(obs)
    stability = np.array([1.0 - sum(probs[i]**2) for i in range(n)])
    return {'regime': regime, 'p_bull': probs[:, 0], 'p_range': probs[:, 1], 'p_bear': probs[:, 2],
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
                'leverage': t.get("leverage", 50),
                'regime': int(hmm['regime'][idx]),
                'p_bull': float(hmm['p_bull'][idx]), 'p_range': float(hmm['p_range'][idx]),
                'p_bear': float(hmm['p_bear'][idx]), 'stability': float(hmm['stability'][idx]),
            })
    return recs


def mc_run(subset, n_sims=5000):
    if len(subset) < 5:
        return None
    tr = [TradeResult(profit_pct=r['profit_pct'], win=r['win'], duration_hours=r['duration_h'],
                      entry_price=0, exit_price=0, regime=r['regime'], kf_direction=0, kf_confidence=0.5)
          for r in subset]
    mc = MonteCarloValidator(n_simulations=n_sims).simulate_trade_sequences(tr)
    return {
        'n': len(subset),
        'awr': sum(r['win'] for r in subset) / len(subset) * 100,
        'mc_wr': np.percentile(mc['win_rate'], 50) * 100,
        'mc_pf': np.percentile(mc['profit_factor'], 50),
        'mc_dd': np.percentile(mc['max_dd'], 95) * 100,
        'mc_ruin': np.mean(mc['ruin_prob']) * 100,
    }


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


# ==============================================================
print("=" * 80, flush=True)
print("HMM × MC DEEP DIVE: finding ANY path to 74% WR", flush=True)
print("=" * 80, flush=True)

recs = make_records()
print(f"\nTotal records: {len(recs)}", flush=True)

# 1) Print ALL combos with sufficient trades
print("\n--- All regime×tag×direction combos (N>=5) ---", flush=True)
combos = defaultdict(list)
for r in recs:
    key = f"R{r['regime']}/{r['enter_tag']}/{'short' if r['is_short'] else 'long'}"
    combos[key].append(r)

for key, sub in sorted(combos.items(), key=lambda x: -len(x[1])):
    wr = sum(1 for s in sub if s['win']) / len(sub) * 100
    print(f"  {key:45s}: {len(sub):>4d} trades, WR={wr:>5.1f}%", flush=True)

# 2) MC grid: test every subset definition we can think of
print("\n--- MC Grid Search (5000 sims each) ---", flush=True)
validator = MonteCarloValidator(n_simulations=5000)
tested = 0

def test_subset(name, fn):
    global tested
    sub = [r for r in recs if fn(r)]
    if len(sub) < 5:
        return
    tested += 1
    r = mc_run(sub)
    if r:
        print(f"  [{tested:2d}] {name:<45s} N={r['n']:>4d} AWR={r['awr']:>5.1f}% "
              f"MC={r['mc_wr']:>5.1f}% PF={r['mc_pf']:.2f}", flush=True)

# Baseline
test_subset("all", lambda r: True)

# Regime only
for rid in [0, 2]:
    test_subset(f"regime={rid}", lambda r, x=rid: r['regime'] == x)

# Direction + regime
for rid in [0, 2]:
    test_subset(f"reg={rid}+long", lambda r, x=rid: r['regime'] == x and not r['is_short'])
    test_subset(f"reg={rid}+short", lambda r, x=rid: r['regime'] == x and r['is_short'])

# Tag only
for tag in ['rsi_momentum', 'momentum_breakout', 'short_bear_momentum', 'hmm_pullback']:
    test_subset(f"tag={tag}", lambda r, x=tag: r['enter_tag'] == x)

# Tag + regime
for rid in [0, 2]:
    for tag in ['rsi_momentum', 'momentum_breakout', 'short_bear_momentum']:
        test_subset(f"reg={rid}+{tag}", lambda r, x=rid, y=tag: r['regime'] == x and r['enter_tag'] == y)

# Tag + direction
for tag in ['rsi_momentum', 'momentum_breakout', 'short_bear_momentum']:
    test_subset(f"tag={tag}+long", lambda r, x=tag: r['enter_tag'] == x and not r['is_short'])
    test_subset(f"tag={tag}+short", lambda r, x=tag: r['enter_tag'] == x and r['is_short'])

# Tag + regime + direction
for rid in [0, 2]:
    for tag in ['rsi_momentum', 'short_bear_momentum']:
        for dr in ['long', 'short']:
            fn = lambda r, x=rid, y=tag, z=(dr == 'short'): r['regime'] == x and r['enter_tag'] == y and r['is_short'] == z
            test_subset(f"r{rid}+{tag}+{dr}", fn)

# Stability filters
for st in [0.2, 0.3, 0.4]:
    test_subset(f"stability<{st:.1f}", lambda r, x=st: r['stability'] < x)
    test_subset(f"stability<{st:.1f}+long", lambda r, x=st: r['stability'] < x and not r['is_short'])

# Price action filters (approximate with available data)
test_subset("open>close(up)", lambda r: r['open_rate'] < r['max_rate'] or True)  # always true as placeholder

# Pair-level filters
pair_wr = defaultdict(list)
for r in recs:
    pair_wr[r['pair']].append(r)
best_pairs = sorted(pair_wr.keys(), key=lambda p: sum(1 for r in pair_wr[p] if r['win']) / len(pair_wr[p]) * 100, reverse=True)
for p in best_pairs[:3]:
    sub = pair_wr[p]
    wr = sum(1 for r in sub if r['win']) / len(sub) * 100
    test_subset(f"pair={p}", lambda r, x=p: r['pair'] == x)

print(f"\nTotal subsets tested: {tested}", flush=True)

# 3) TP/SL optimization on best subsets
print("\n--- Phase 2: TP/SL optimization on promising subsets ---", flush=True)
validator2 = MonteCarloValidator(n_simulations=1000)

def score_tp_sl(subset, tp, sl):
    sim = do_tp_sl(subset, tp, sl)
    mc = validator2.simulate_trade_sequences(sim)
    mw = np.percentile(mc['win_rate'], 50) * 100
    mr = np.mean(mc['ruin_prob']) * 100
    return mw, mr

# Find subsets with actual WR > 40% and N >= 15
promising = []
for rid in [0, 2]:
    for tag in ['rsi_momentum', 'short_bear_momentum']:
        for dr in ['long', 'short']:
            dr_bool = (dr == 'short')
            sub = [r for r in recs if r['regime'] == rid and r['enter_tag'] == tag and r['is_short'] == dr_bool]
            if len(sub) >= 10:
                wr = sum(1 for r in sub if r['win']) / len(sub) * 100
                promising.append((f"R{rid}+{tag}+{dr}", sub, wr))

# Also check tag-only subsets
for tag in ['rsi_momentum', 'momentum_breakout', 'short_bear_momentum', 'hmm_pullback']:
    sub = [r for r in recs if r['enter_tag'] == tag and not r['is_short']]
    if len(sub) >= 10:
        wr = sum(1 for r in sub if r['win']) / len(sub) * 100
        promising.append((f"{tag}+long", sub, wr))
    sub = [r for r in recs if r['enter_tag'] == tag and r['is_short']]
    if len(sub) >= 10:
        wr = sum(1 for r in sub if r['win']) / len(sub) * 100
        promising.append((f"{tag}+short", sub, wr))

promising.sort(key=lambda x: -x[2])
print(f"  Testing {len(promising)} promising subsets with TP/SL grid...", flush=True)

for name, sub, base_wr in promising:
    best = None
    best_s = float('inf')
    for tp in [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]:
        for sl in [0.003, 0.005, 0.008, 0.01, 0.015]:
            mw, mr = score_tp_sl(sub, tp, sl)
            s = abs(74 - mw) + max(0, mr - 10) * 0.3
            if s < best_s:
                best_s = s
                best = (tp, sl, mw, mr)
    if best:
        print(f"  {name:<40s} N={len(sub):>3d} base={base_wr:.1f}% → "
              f"TP={best[0]*100:.1f}% SL={best[1]*100:.1f}% "
              f"MC_WR={best[2]:.1f}% Ruin={best[3]:.0f}%", flush=True)
