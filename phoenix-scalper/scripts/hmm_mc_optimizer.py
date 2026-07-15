#!/usr/bin/env python3
"""
HMM-Regime-Gated Monte Carlo WinRate Optimizer.
Target: daily 74% win rate via creative HMM regime gating + MC validation.
"""
import sys, os, json, zipfile, re, logging, itertools
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.hmm_regime import RegimeDetector
from ml.monte_carlo import MonteCarloValidator, TradeResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "/freqtrade/user_data/data/binance/futures"
BACKTEST_DIR = "/freqtrade/user_data/backtest_results"


def load_trades():
    zips = sorted([f for f in os.listdir(BACKTEST_DIR) if f.endswith(".zip")])
    zip_path = os.path.join(BACKTEST_DIR, zips[-1])
    logger.info(f"Loading backtest: {zips[-1]}")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".json") and "_config" not in name:
                raw = z.read(name)
                m = re.search(rb'\{.*', raw, re.DOTALL)
                data = json.loads(m.group())
    trades = data["strategy"]["PhoenixScalper"]["trades"]
    logger.info(f"Trades loaded: {len(trades)}")
    return trades


def compute_hmm_for_pair(df, n_states=3):
    returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    vol = returns.rolling(10).std().fillna(0)
    vol_ema = df['volume'].ewm(span=10).mean()
    vol_ratio = df['volume'] / (vol_ema + 1e-10)
    vol_change = vol_ratio.pct_change().fillna(0)

    obs = np.column_stack([returns.values, vol.values, vol_change.values])
    n = len(obs)
    max_train = 1000
    obs_train = obs[np.linspace(0, n - 1, max_train, dtype=int)] if n > max_train else obs

    hmm = RegimeDetector(n_states=n_states, n_iter=5)
    hmm.fit(obs_train)
    regime = hmm.predict_regime_fast(obs)
    probs = hmm.predict_regime_probs_fast(obs)
    stability = np.array([1.0 - np.sum(probs[i] ** 2) for i in range(n)])
    trend = np.array([abs(hmm.mu[regime[i], 0]) / (hmm.Sigma[regime[i], 0, 0] + 1e-10) for i in range(n)])
    trans_risk = np.array([
        (hmm.A[r, 1] + hmm.A[r, 2]) if r == 0 else
        (hmm.A[r, 0] + hmm.A[r, 2]) if r == 1 else
        (hmm.A[r, 0] + hmm.A[r, 1])
        for r in regime
    ])
    return {
        'hmm_regime': regime, 'hmm_p_bull': probs[:, 0],
        'hmm_p_range': probs[:, 1], 'hmm_p_bear': probs[:, 2],
        'hmm_regime_stability': stability, 'hmm_trend_strength': trend,
        'hmm_transition_risk': trans_risk,
    }


def build_trade_dataset(trades_data, pairs_dir, n_states=3):
    trades_by_pair = defaultdict(list)
    for t in trades_data:
        pk = t["pair"].replace("/", "_").replace(":", "_")
        trades_by_pair[pk].append(t)

    records = []
    for pair_key, trades in trades_by_pair.items():
        fpath = os.path.join(pairs_dir, f"{pair_key}-5m-futures.feather")
        if not os.path.exists(fpath):
            continue
        df = pd.read_feather(fpath)
        dates = df['date'].values
        hmm = compute_hmm_for_pair(df, n_states=n_states)
        for t in trades:
            idx = np.argmin(np.abs(dates - pd.Timestamp(t["open_date"]).to_datetime64()))
            idx = min(idx, len(hmm['hmm_regime']) - 1)
            records.append({
                'profit_pct': t["profit_ratio"] * 100,
                'win': t["profit_ratio"] > 0,
                'duration_h': t["trade_duration"] / 60.0,
                'is_short': t.get("is_short", False),
                'enter_tag': t.get("enter_tag", ""),
                'pair': pair_key,
                'open_rate': t["open_rate"],
                'min_rate': t.get("min_rate", t["open_rate"] * 0.99),
                'max_rate': t.get("max_rate", t["open_rate"] * 1.01),
                'leverage': t.get("leverage", 50),
                'regime': int(hmm['hmm_regime'][idx]),
                'p_bull': float(hmm['hmm_p_bull'][idx]),
                'p_range': float(hmm['hmm_p_range'][idx]),
                'p_bear': float(hmm['hmm_p_bear'][idx]),
                'stability': float(hmm['hmm_regime_stability'][idx]),
                'trend_strength': float(hmm['hmm_trend_strength'][idx]),
                'transition_risk': float(hmm['hmm_transition_risk'][idx]),
            })
    logger.info(f"Trade records with HMM: {len(records)}")
    return records


def phase1_per_regime(records):
    regimes = {0: 'Bull', 1: 'Range', 2: 'Bear'}
    print("\n" + "=" * 80, flush=True)
    print("PHASE 1: PER-REGIME WIN RATE ANALYSIS", flush=True)
    print("=" * 80, flush=True)
    for rid, name in sorted(regimes.items()):
        sub = [r for r in records if r['regime'] == rid]
        if sub:
            wr = sum(1 for r in sub if r['win']) / len(sub) * 100
            print(f"  {name} (reg {rid}): {len(sub):>4d} trades, WR={wr:>5.1f}%", flush=True)
    tags = defaultdict(list)
    for r in records:
        tags[r['enter_tag']].append(r)
    print("\n  Per enter_tag:", flush=True)
    for tag, sub in sorted(tags.items()):
        wr = sum(1 for r in sub if r['win']) / len(sub) * 100
        print(f"    {tag:30s}: {len(sub):>4d} trades, WR={wr:>5.1f}%", flush=True)


def to_trade_results(subset):
    return [TradeResult(profit_pct=r['profit_pct'], win=r['win'],
                        duration_hours=r['duration_h'], entry_price=0, exit_price=0,
                        regime=r['regime'], kf_direction=0, kf_confidence=0.5)
            for r in subset]


def phase2_mc_search(records):
    print("\n" + "=" * 80, flush=True)
    print("PHASE 2: HMM-GATED MC SEARCH (target: 74% MC WR)", flush=True)
    print("=" * 80, flush=True)

    validator = MonteCarloValidator(n_simulations=5000)
    results = []
    n_checked = 0

    def check(name, fn):
        nonlocal n_checked
        sub = [r for r in records if fn(r)]
        if len(sub) < 10:
            return
        n_checked += 1
        mc = validator.simulate_trade_sequences(to_trade_results(sub))
        mc_wr = np.percentile(mc['win_rate'], 50) * 100
        mc_pf = np.percentile(mc['profit_factor'], 50)
        mc_dd = np.percentile(mc['max_dd'], 95) * 100
        mc_ruin = np.mean(mc['ruin_prob']) * 100
        awr = sum(r['win'] for r in sub) / len(sub) * 100
        score = abs(74 - mc_wr) + max(0, mc_ruin - 10) * 0.3 + max(0, mc_dd - 30) * 0.2
        results.append({'filter': name, 'n': len(sub), 'actual_wr': round(awr, 1),
                        'mc_wr': round(mc_wr, 1), 'mc_pf': round(mc_pf, 2),
                        'mc_dd': round(mc_dd, 1), 'mc_ruin': round(mc_ruin, 1),
                        'score': round(score, 3)})
        print(f"  [{n_checked:2d}] {name:<40s} N={len(sub):>4d} AWR={awr:>5.1f}% "
              f"MC={mc_wr:>5.1f}% PF={mc_pf:.2f}", flush=True)

    check("all (baseline)", lambda r: True)
    for rid in [0, 1, 2]:
        check(f"regime={rid}", lambda r, x=rid: r['regime'] == x)
    for t in [0.2, 0.3, 0.4, 0.5]:
        check(f"stability<{t:.1f}", lambda r, x=t: r['stability'] < x)
    for t in [0.3, 0.5, 0.7, 0.9]:
        check(f"p_bull>={t:.1f}", lambda r, x=t: r['p_bull'] >= x)
        check(f"p_bear>={t:.1f}", lambda r, x=t: r['p_bear'] >= x)
    for t in [0.3, 0.5, 0.7]:
        check(f"dir:p>={t:.1f}", lambda r, x=t: (not r['is_short'] and r['p_bull'] >= x) or (r['is_short'] and r['p_bear'] >= x))
    for t in [1.0, 1.5, 2.0]:
        check(f"trend>={t:.1f}", lambda r, x=t: r['trend_strength'] >= x)
    for rid in [0, 2]:
        for st in [0.3, 0.5]:
            check(f"reg={rid}&stab<{st:.1f}", lambda r, x=rid, y=st: r['regime'] == x and r['stability'] < y)
    for pt in [0.5, 0.7]:
        for tt in [1.5, 2.0]:
            check(f"p_b>={pt}&tr>={tt}", lambda r, x=pt, y=tt: r['p_bull'] >= x and r['trend_strength'] >= y)
    for ct in [0.3, 0.5]:
        for st in [0.3, 0.5]:
            check(f"dir:p{ct}&st{st}", lambda r, x=ct, y=st: ((not r['is_short'] and r['p_bull'] >= x) or (r['is_short'] and r['p_bear'] >= x)) and r['stability'] < y)
    check("regime=0|2", lambda r: r['regime'] in (0, 2))
    check("dominant", lambda r: (r['p_bull'] > 2*r['p_range'] and r['p_bull'] > 2*r['p_bear']) or (r['p_bear'] > 2*r['p_range'] and r['p_bear'] > 2*r['p_bull']))

    results.sort(key=lambda x: -x['mc_wr'])
    print(f"\n  Top 10 by MC WR:", flush=True)
    print(f"  {'Filter':<40s} {'N':>5s} {'ActWR':>7s} {'MC_WR':>7s} {'PF':>6s} {'DD':>7s} {'Ruin':>7s} {'Score':>7s}", flush=True)
    print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7}", flush=True)
    for r in results[:10]:
        print(f"  {r['filter']:<40s} {r['n']:>5d} {r['actual_wr']:>6.1f}% {r['mc_wr']:>6.1f}% "
              f"{r['mc_pf']:>5.2f} {r['mc_dd']:>6.1f}% {r['mc_ruin']:>6.1f}% {r['score']:>6.3f}", flush=True)
    return results


def simulate_tp_sl(records, tp_pct, sl_pct):
    results = []
    for r in records:
        is_short = r['is_short']
        o, lo, hi = r['open_rate'], r['min_rate'], r['max_rate']
        if is_short:
            tp_hit, sl_hit = lo <= o * (1 - tp_pct), hi >= o * (1 + sl_pct)
        else:
            tp_hit, sl_hit = hi >= o * (1 + tp_pct), lo <= o * (1 - sl_pct)
        if tp_hit and not sl_hit:
            p, w = tp_pct * 100, True
        elif sl_hit and not tp_hit:
            p, w = -sl_pct * 100, False
        else:
            p, w = r['profit_pct'], r['profit_pct'] > 0
        results.append(TradeResult(p, w, r['duration_h'], o, 0, r['regime'], 0, 0.5))
    return results


def phase3_regime_tp_sl(records):
    print("\n" + "=" * 80, flush=True)
    print("PHASE 3: REGIME-ADAPTIVE TP/SL VIA MC", flush=True)
    print("=" * 80, flush=True)
    regimes = {0: 'Bull', 1: 'Range', 2: 'Bear'}
    validator = MonteCarloValidator(n_simulations=1000)
    all_best = []

    for rid, name in sorted(regimes.items()):
        sub = [r for r in records if r['regime'] == rid]
        if len(sub) < 20:
            print(f"  {name}: {len(sub)} trades, skip", flush=True)
            continue
        base_wr = sum(1 for r in sub if r['win']) / len(sub) * 100
        print(f"\n  {name} ({len(sub)} trades, base WR={base_wr:.1f}%):", flush=True)

        # Coarse grid first
        best = None
        best_s = float('inf')
        for tp in [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]:
            for sl in [0.003, 0.005, 0.008, 0.01, 0.015]:
                sim = simulate_tp_sl(sub, tp, sl)
                mc = validator.simulate_trade_sequences(sim)
                mw = np.percentile(mc['win_rate'], 50) * 100
                mr = np.mean(mc['ruin_prob']) * 100
                s = abs(74 - mw) + max(0, mr - 10) * 0.3
                if s < best_s:
                    best_s, best = s, {'tp': tp, 'sl': sl, 'mc_wr': round(mw, 1), 'mc_ruin': round(mr, 1)}

        if best:
            print(f"    Best: TP={best['tp']*100:.1f}% SL={best['sl']*100:.1f}% "
                  f"→ MC_WR={best['mc_wr']:.1f}% Ruin={best['mc_ruin']:.1f}%", flush=True)
            best['regime'] = rid
            best['regime_name'] = name
            all_best.append(best)

    return all_best


def phase4_multistate(records):
    print("\n" + "=" * 80, flush=True)
    print("PHASE 4: 4-STATE HMM EXPLORATION", flush=True)
    print("=" * 80, flush=True)
    td = load_trades()
    r4 = build_trade_dataset(td, DATA_DIR, n_states=4)
    if not r4:
        return
    for rid in sorted(set(r['regime'] for r in r4)):
        sub = [r for r in r4 if r['regime'] == rid]
        wr = sum(1 for r in sub if r['win']) / len(sub) * 100
        print(f"  State {rid}: {len(sub):>4d} trades, WR={wr:>5.1f}%", flush=True)
    mc4 = phase2_mc_search(r4)
    return r4, mc4


def main():
    td = load_trades()
    records = build_trade_dataset(td, DATA_DIR, n_states=3)
    phase1_per_regime(records)
    mc_results = phase2_mc_search(records)

    best_f = max(mc_results, key=lambda x: x['mc_wr']) if mc_results else None
    if best_f:
        print(f"\n  *** Best filter: '{best_f['filter']}' → MC_WR={best_f['mc_wr']:.1f}% "
              f"(need 74%) ***", flush=True)

    if best_f and best_f['mc_wr'] >= 50:
        phase3_regime_tp_sl(records)

    if not best_f or best_f['mc_wr'] < 60:
        print(f"\n  3-state HMM best only {best_f['mc_wr']:.1f}% — trying 4-state...", flush=True)
        r4 = phase4_multistate(records)
        if r4:
            b4 = max(r4[1], key=lambda x: x['mc_wr']) if r4[1] else None
            if b4:
                print(f"\n  4-state best: '{b4['filter']}' → MC_WR={b4['mc_wr']:.1f}%", flush=True)

    out_path = '/freqtrade/user_data/hmm_mc_results.json'
    with open(out_path, 'w') as f:
        json.dump({'n_trades': len(records), 'best_filter': best_f,
                    'all_filters': mc_results}, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
