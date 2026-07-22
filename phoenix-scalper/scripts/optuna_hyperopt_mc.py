#!/usr/bin/env python3
"""
Phase 1a: Optuna + MC Hyperopt using precomputed features.
Separate long/short searches with TPE sampler.
MC P50 Sharpe objective with DD/ruin penalties.

Usage (inside container):
  python /freqtrade/strategies/scripts/optuna_hyperopt_mc.py \\
    --side long --trials 300 --study-name long_v1

  python /freqtrade/strategies/scripts/optuna_hyperopt_mc.py \\
    --side short --trials 300 --study-name short_v1
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.monte_carlo import MonteCarloValidator, TradeResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('optuna_mc')

FEATHER_PATH = '/freqtrade/user_data/data/precomputed_features.feather'
STARTUP_BARS = 150


def load_data(path: str = FEATHER_PATH):
    import pandas as pd
    logger.info(f'Loading precomputed features from {path}')
    t0 = time.time()
    df = pd.read_feather(path)
    logger.info(f'Loaded {len(df)} rows x {len(df.columns)} cols in {time.time()-t0:.0f}s')
    return df


def simulate_long_trades(df, params: Dict) -> List[TradeResult]:
    adx_th = params['adx_threshold']
    vf = params['volume_factor']
    rsi_p = params['rsi_period']
    rsi_os = params['rsi_oversold']
    ema_f = params['ema_fast']
    ema_s = params['ema_slow']
    score_t = params['score_threshold']
    sl_min = params.get('sl_min', 0.0025)
    sl_max = params.get('sl_max', 0.0050)

    rsi_col = f'rsi_{rsi_p}'
    if rsi_col not in df.columns:
        rsi_col = 'rsi_14'
    ema_f_col = f'ema_{ema_f}'
    ema_s_col = f'ema_{ema_s}'

    mask = np.zeros(len(df), dtype=bool)

    for condition_name in ['pullback', 'rsi_momentum', 'momentum_breakout', 'kalman_cont']:
        if condition_name == 'pullback':
            cond = (
                (df['low'] <= df[ema_s_col] * 1.005)
                & (df['close'] > df[ema_s_col])
                & (df['close'] > df['open'])
                & (df[rsi_col] > 35) & (df[rsi_col] < 55)
                & (df['volume_ratio'] > vf)
                & (df['adx'] > adx_th)
                & (df['plus_di'] > df['minus_di'])
                & (df['volume'] > 0)
            )
        elif condition_name == 'rsi_momentum':
            cond = (
                (df[rsi_col] > 50)
                & (df['close'] > df['open'])
                & (df['close'] > df[ema_f_col])
                & (df['volume_ratio'] > vf)
                & (df['adx'] > adx_th)
                & (df['volume'] > 0)
            )
        elif condition_name == 'momentum_breakout':
            cond = (
                (df['close'] > df['high'].rolling(5).max().shift(1))
                & (df['volume_ratio'] > vf * 1.5)
                & (df['adx'] > adx_th * 1.2)
                & (df['close'] > df[ema_s_col])
                & (df['volume'] > 0)
            )
        elif condition_name == 'kalman_cont':
            cond = (
                (df['kf_trend'] > 0)
                & (df['kf_confidence'] > 0.6)
                & (df['close'] > df['kf_prediction'])
                & (df['kf_trend_acceleration'] > 0)
                & (df['close'] > df[ema_f_col])
                & (df['volume_ratio'] > vf)
                & (df['volume'] > 0)
            )
        mask = mask | cond.values

    score_ok = df['long_score'].values >= score_t
    mask = mask & score_ok & (df['long_target'].notna().values)

    indices = np.where(mask)[0]
    if len(indices) < 5:
        return []

    trade_results = []
    for i in indices:
        ret = df['long_return'].iloc[i]
        if np.isnan(ret):
            continue
        trade_results.append(TradeResult(
            profit_pct=float(ret) * 100,
            win=float(ret) > 0,
            duration_hours=float(df['long_bars'].iloc[i] * 5 / 60),
            entry_price=float(df['close'].iloc[i]),
            exit_price=float(df['close'].iloc[i]) * (1 + float(ret)),
            regime=int(df['hmm_regime'].iloc[i]) if 'hmm_regime' in df else 0,
            kf_direction=int(np.sign(df['kf_trend'].iloc[i])) if 'kf_trend' in df else 0,
            kf_confidence=float(df['kf_confidence'].iloc[i]) if 'kf_confidence' in df else 0.5,
        ))
    return trade_results


def simulate_short_trades(df, params: Dict) -> List[TradeResult]:
    adx_th = params['adx_threshold']
    vf = params['volume_factor']
    rsi_p = params['rsi_period']
    st = params.get('short_rsi_threshold', 46)
    lm = params.get('short_lookback', 13)
    vm = params.get('short_volume_mult', 1.931)
    am = params.get('short_adx_mult', 1.258)
    ema_f = params['ema_fast']
    ema_s = params['ema_slow']
    score_t = params['score_threshold']
    sl_min = params.get('sl_min', 0.0025)
    sl_max = params.get('sl_max', 0.0050)

    rsi_col = f'rsi_{rsi_p}'
    if rsi_col not in df.columns:
        rsi_col = 'rsi_14'
    ema_f_col = f'ema_{ema_f}'
    ema_s_col = f'ema_{ema_s}'

    mask = np.zeros(len(df), dtype=bool)

    for condition_name in ['short_breakdown', 'short_rally_fail', 'short_bear_momentum']:
        if condition_name == 'short_breakdown':
            cond = (
                (df['close'] < df['low'].rolling(lm).min().shift(1))
                & (df['volume_ratio'] > vf * vm)
                & (df['adx'] > adx_th * am)
                & (df['close'] < df[ema_s_col])
                & (df['close'] < df['open'])
                & (df['plus_di'] < df['minus_di'])
                & (df[rsi_col] < st)
                & (df['volume'] > 0)
            )
        elif condition_name == 'short_rally_fail':
            cond = (
                (df['high'] >= df[ema_s_col] * 0.995)
                & (df['close'] < df[ema_s_col])
                & (df['close'] < df['open'])
                & (df[rsi_col] > 55) & (df[rsi_col] < 75)
                & (df['volume_ratio'] > vf)
                & (df['adx'] > adx_th)
                & (df['plus_di'] < df['minus_di'])
                & (df['volume'] > 0)
            )
        elif condition_name == 'short_bear_momentum':
            cond = (
                (df['close'] < df['open'])
                & (df['volume_ratio'] > vf * 1.3)
                & (df['adx'] > adx_th * 1.1)
                & (df['plus_di'] < df['minus_di'])
                & (df[rsi_col] < 45)
                & (df['close'] < df[ema_s_col])
                & (df['volume'] > 0)
            )
        mask = mask | cond.values

    score_ok = df['short_score'].values >= score_t
    mask = mask & score_ok & (df['short_target'].notna().values)

    indices = np.where(mask)[0]
    if len(indices) < 5:
        return []

    trade_results = []
    for i in indices:
        ret = df['short_return'].iloc[i]
        if np.isnan(ret):
            continue
        trade_results.append(TradeResult(
            profit_pct=float(ret) * 100,
            win=float(ret) > 0,
            duration_hours=float(df['short_bars'].iloc[i] * 5 / 60),
            entry_price=float(df['close'].iloc[i]),
            exit_price=float(df['close'].iloc[i]) * (1 - float(ret)),
            regime=int(df['hmm_regime'].iloc[i]) if 'hmm_regime' in df else 0,
            kf_direction=int(np.sign(df['kf_trend'].iloc[i])) if 'kf_trend' in df else 0,
            kf_confidence=float(df['kf_confidence'].iloc[i]) if 'kf_confidence' in df else 0.5,
        ))
    return trade_results


def suggest_long_params(trial: optuna.Trial) -> Dict:
    return {
        'adx_threshold': trial.suggest_int('adx_threshold', 18, 35),
        'volume_factor': trial.suggest_float('volume_factor', 0.6, 3.0),
        'rsi_period': trial.suggest_int('rsi_period', 5, 12),
        'rsi_oversold': trial.suggest_int('rsi_oversold', 22, 40),
        'ema_fast': trial.suggest_int('ema_fast', 5, 12),
        'ema_slow': trial.suggest_int('ema_slow', 12, 24),
        'score_threshold': trial.suggest_int('score_threshold', 40, 80),
        'sl_min': trial.suggest_float('sl_min', 0.0015, 0.0035),
        'sl_max': trial.suggest_float('sl_max', 0.0035, 0.0060),
    }


def suggest_short_params(trial: optuna.Trial) -> Dict:
    return {
        'adx_threshold': trial.suggest_int('adx_threshold', 18, 35),
        'volume_factor': trial.suggest_float('volume_factor', 0.6, 3.0),
        'rsi_period': trial.suggest_int('rsi_period', 5, 12),
        'ema_fast': trial.suggest_int('ema_fast', 5, 12),
        'ema_slow': trial.suggest_int('ema_slow', 12, 24),
        'score_threshold': trial.suggest_int('score_threshold', 40, 80),
        'short_lookback': trial.suggest_int('short_lookback', 7, 15),
        'short_volume_mult': trial.suggest_float('short_volume_mult', 1.5, 3.0),
        'short_adx_mult': trial.suggest_float('short_adx_mult', 1.0, 1.5),
        'short_rsi_threshold': trial.suggest_int('short_rsi_threshold', 30, 50),
        'sl_min': trial.suggest_float('sl_min', 0.0015, 0.0035),
        'sl_max': trial.suggest_float('sl_max', 0.0035, 0.0060),
    }


def compute_mc_score(trades: List[TradeResult], validator: MonteCarloValidator) -> float:
    n = len(trades)
    if n < 20:
        return -999.0

    results = validator.simulate_trade_sequences(trades)
    sharpe_p50 = float(np.percentile(results['sharpe'], 50))
    sharpe_p10 = float(np.percentile(results['sharpe'], 10))
    max_dd_p95 = float(np.percentile(results['max_dd'], 95))
    ruin_prob = float(np.mean(results['ruin_prob']))
    wr_p50 = float(np.percentile(results['win_rate'], 50))
    pf_p50 = float(np.percentile(results['profit_factor'], 50))
    final_eq_p50 = float(np.percentile(results['final_equity'], 50))

    score = sharpe_p50
    score -= max(0, ruin_prob - 0.05) * 15
    score -= max(0, max_dd_p95 - 0.20) * 8
    score -= max(0, 3.0 - n / 60) * 5
    score -= max(0, 0.50 - wr_p50) * 8
    if sharpe_p10 < -1.0:
        score -= 5
    if pf_p50 < 1.0:
        score -= max(0, 1.0 - pf_p50) * 15
    if final_eq_p50 < 1000:
        score -= 5

    return score


def objective_long(df, validator: MonteCarloValidator, trial: optuna.Trial) -> float:
    params = suggest_long_params(trial)
    trades = simulate_long_trades(df, params)
    score = compute_mc_score(trades, validator)
    _store_attrs(trial, trades, params, score)
    return score


def objective_short(df, validator: MonteCarloValidator, trial: optuna.Trial) -> float:
    params = suggest_short_params(trial)
    trades = simulate_short_trades(df, params)
    score = compute_mc_score(trades, validator)
    _store_attrs(trial, trades, params, score)
    return score


def _store_attrs(trial, trades, params, score):
    profits = np.array([t.profit_pct for t in trades])
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    trial.set_user_attr('n_trades', len(trades))
    trial.set_user_attr('win_rate', round(float(np.mean(profits > 0)), 4) if len(profits) > 0 else 0)
    trial.set_user_attr('avg_win', round(float(np.mean(wins)), 4) if len(wins) > 0 else 0)
    trial.set_user_attr('avg_loss', round(float(np.mean(losses)), 4) if len(losses) > 0 else 0)
    trial.set_user_attr('total_pnl', round(float(np.sum(profits)), 4))
    trial.set_user_attr('params', params)


def print_best(study, side: str):
    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"  BEST {side.upper()}  |  MC Score: {best.value:.2f}")
    print(f"{'='*70}")
    print(f"  Params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:30s} = {v}")
    print(f"  Metrics:")
    for k in ['n_trades', 'win_rate', 'avg_win', 'avg_loss', 'total_pnl']:
        if k in best.user_attrs:
            print(f"    {k:30s} = {best.user_attrs[k]}")
    print()

    top = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)[:10]
    print(f"  Top-10 trials:")
    print(f"  {'Rank':<5} {'Score':<8} {'Trades':<8} {'WR':<8} {'AvgWin':<10} {'AvgLoss':<10} {'PnL':<10}")
    print(f"  {'-'*61}")
    for i, t in enumerate(top[:5]):
        v = t.value or 0
        nt = t.user_attrs.get('n_trades', 0)
        wr = t.user_attrs.get('win_rate', 0)
        aw = t.user_attrs.get('avg_win', 0)
        al = t.user_attrs.get('avg_loss', 0)
        pnl = t.user_attrs.get('total_pnl', 0)
        print(f"  {i:<5} {v:<8.2f} {nt:<8} {wr:<7.1%} {aw:<10.2f} {al:<10.2f} {pnl:<10.2f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', required=True, choices=['long', 'short'])
    parser.add_argument('--study-name', default=None)
    parser.add_argument('--trials', type=int, default=300)
    parser.add_argument('--mc-sims', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--timerange-start', default=None)
    parser.add_argument('--timerange-end', default=None)
    args = parser.parse_args()

    side = args.side
    study_name = args.study_name or f'{side}_v1'
    print(f"\n  {'='*60}")
    print(f"  Optuna + MC Hyperopt | Side: {side} | Trials: {args.trials}")
    print(f"  Study: {study_name} | MC Sims: {args.mc_sims}")
    print(f"  {'='*60}\n")

    np.random.seed(args.seed)
    df = load_data()

    if args.timerange_start:
        df = df[df['date'] >= args.timerange_start]
    if args.timerange_end:
        df = df[df['date'] <= args.timerange_end]
    df = df.iloc[STARTUP_BARS:].reset_index(drop=True)
    logger.info(f'After filter+startup: {len(df)} rows')

    # Baseline on current default params
    default_params = {
        'adx_threshold': 30, 'volume_factor': 1.003, 'rsi_period': 6,
        'rsi_oversold': 22, 'ema_fast': 7, 'ema_slow': 14,
        'score_threshold': 51, 'sl_min': 0.0025, 'sl_max': 0.0050,
        'short_lookback': 13, 'short_volume_mult': 1.931,
        'short_adx_mult': 1.258, 'short_rsi_threshold': 46,
    }
    if side == 'long':
        baseline_trades = simulate_long_trades(df, default_params)
    else:
        baseline_trades = simulate_short_trades(df, default_params)
    baseline_profits = [t.profit_pct for t in baseline_trades]
    print(f"  Baseline ({side}, default params):")
    print(f"    Trades: {len(baseline_trades)} | WR: {sum(1 for p in baseline_profits if p>0)/max(len(baseline_profits),1):.1%}")
    print(f"    Avg PnL: {np.mean(baseline_profits) if baseline_profits else 0:.2f}% | Total: {sum(baseline_profits):.2f}%")
    print()

    validator = MonteCarloValidator(n_simulations=args.mc_sims)
    out_dir = '/freqtrade/user_data'
    storage = f'sqlite:///{out_dir}/optuna_{study_name}.db'

    sampler = TPESampler(seed=args.seed, n_startup_trials=20, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=10, n_min_trials=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    objective_fn = objective_long if side == 'long' else objective_short

    study.optimize(
        lambda t: objective_fn(df, validator, t),
        n_trials=args.trials,
        n_jobs=1,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print_best(study, side)

    out = os.path.join(out_dir, f'optimal_params_{study_name}.json')
    best = study.best_trial
    with open(out, 'w') as f:
        json.dump({
            'side': side,
            'params': best.params,
            'metrics': {k: v for k, v in best.user_attrs.items() if k != 'params'},
            'mc_score': best.value,
            'n_trials': len(study.trials),
            'timestamp': datetime.utcnow().isoformat(),
        }, f, indent=2)
    print(f"  Saved: {out}\n")


if __name__ == '__main__':
    main()
