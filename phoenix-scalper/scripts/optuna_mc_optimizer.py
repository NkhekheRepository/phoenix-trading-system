#!/usr/bin/env python3
"""
PhoenixScalper — Honest Optuna + Monte Carlo Risk Optimizer

Evaluates parameters that MC can legitimately optimize on existing trade data:
  - Leverage (scales all profits linearly)
  - Stake fraction (position sizing)
  - Trade filters (which subsets to include/exclude by exit_reason, direction, etc.)

Does NOT optimize TP/SL targets (needs price paths — use Freqtrade hyperopt instead).

Usage:
  docker compose exec phoenix-scalper python scripts/optuna_mc_optimizer.py \\
      --db /freqtrade/user_data/backup_clean_v2.sqlite \\
      --study-name v2_risk --trials 200
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.monte_carlo import MonteCarloValidator, TradeResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optuna_mc")


def load_trades(db_path: str, min_trades: int = 10,
                days_back: Optional[int] = None) -> List[dict]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    query = """
        SELECT close_profit, pair, is_short, leverage, close_date, open_date,
               exit_reason, close_rate, open_rate, enter_tag, stake_amount
        FROM trades WHERE is_open = 0 AND close_profit IS NOT NULL
    """
    params = []
    if days_back:
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        query += f" AND close_date >= ?"
        params.append(cutoff)
    query += " ORDER BY close_date DESC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    if len(rows) < min_trades:
        logger.warning(f"Only {len(rows)} trades (need {min_trades})")
        return []

    trades = []
    for r in rows:
        profit_pct = float(r[0] or 0) * 100
        hours = 0.0
        if r[5] and r[4]:
            try:
                cd = datetime.fromisoformat(str(r[4]).replace("Z", "+00:00").replace(" ", "T"))
                od = datetime.fromisoformat(str(r[5]).replace("Z", "+00:00").replace(" ", "T"))
                hours = (cd - od).total_seconds() / 3600
            except Exception:
                hours = 1.0
        trades.append({
            "profit_pct": profit_pct,
            "win": profit_pct > 0,
            "duration_hours": hours,
            "entry_price": float(r[8] or 100.0),
            "exit_price": float(r[7] or 100.0),
            "pair": r[1],
            "is_short": bool(r[2]),
            "exit_reason": r[6] or "unknown",
            "enter_tag": r[9] or "unknown",
            "leverage": float(r[3] or 1),
        })
    logger.info(f"Loaded {len(trades)} closed trades from {db_path}")
    return trades


def to_trade_results(trades: List[dict], leverage: float,
                     stake_pct: float) -> List[TradeResult]:
    results = []
    scale = leverage / 50.0
    for t in trades:
        profit = t["profit_pct"] * scale
        results.append(TradeResult(
            profit_pct=profit,
            win=profit > 0,
            duration_hours=t["duration_hours"],
            entry_price=t["entry_price"],
            exit_price=t["exit_price"],
            regime=0, kf_direction=0, kf_confidence=0.5,
        ))
    return results


def suggest_params(trial: optuna.Trial) -> Dict:
    return {
        "leverage": trial.suggest_int("leverage", 5, 100, step=5),
        "stake_pct": trial.suggest_float("stake_pct", 0.02, 0.25),
        "short_only": trial.suggest_categorical("short_only", [True, False]),
        "long_only": trial.suggest_categorical("long_only", [True, False]),
    }


def stats(trades: List[TradeResult]) -> Dict:
    profits = np.array([t.profit_pct for t in trades])
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    avg = np.mean(profits) if len(profits) > 0 else 0
    std = np.std(profits) if len(profits) > 1 else 1
    total_hours = sum(t.duration_hours for t in trades) if trades else 1
    tpy = len(profits) / max(total_hours / 24 / 365, 1/365)
    sharpe = (avg / std) * math.sqrt(tpy) if std > 0 and tpy > 0 else 0
    return {
        "n": len(trades),
        "wr": len(wins) / len(profits) if profits.any() else 0,
        "avg_win": float(np.mean(wins)) if len(wins) > 0 else 0,
        "avg_loss": float(np.mean(losses)) if len(losses) > 0 else 0,
        "pf": float(abs(np.sum(wins) / np.sum(losses))) if len(losses) > 0 and abs(np.sum(losses)) > 1e-10 else 0,
        "sharpe": float(sharpe),
        "total_pnl": float(np.sum(profits)),
    }


def objective(trades: List[dict], validator: MonteCarloValidator,
              trial: optuna.Trial) -> float:
    p = suggest_params(trial)

    if p["short_only"] and p["long_only"]:
        return -999.0

    filtered = [t for t in trades
                if (not p["short_only"] or t["is_short"])
                and (not p["long_only"] or not t["is_short"])]

    modified = to_trade_results(filtered, p["leverage"], p["stake_pct"])
    if len(modified) < 5:
        return -999.0

    results = validator.simulate_trade_sequences(modified)

    sharpe_p50 = float(np.percentile(results["sharpe"], 50))
    sharpe_p10 = float(np.percentile(results["sharpe"], 10))
    max_dd_p95 = float(np.percentile(results["max_dd"], 95))
    ruin_prob = float(np.mean(results["ruin_prob"]))
    wr_p50 = float(np.percentile(results["win_rate"], 50))
    pf_p50 = float(np.percentile(results["profit_factor"], 50))
    trades_per_day = len(modified) / max(1, max(t.duration_hours for t in modified) / 24)

    score = sharpe_p50
    score -= max(0, ruin_prob - 0.05) * 15
    score -= max(0, max_dd_p95 - 0.30) * 5
    score -= max(0, 3.0 - trades_per_day) * 2
    if sharpe_p10 < -1.0:
        score -= 3

    trial.set_user_attr("sharpe_p50", round(sharpe_p50, 2))
    trial.set_user_attr("sharpe_p10", round(sharpe_p10, 2))
    trial.set_user_attr("max_dd_p95", round(max_dd_p95, 4))
    trial.set_user_attr("ruin_prob", round(ruin_prob, 4))
    trial.set_user_attr("win_rate_p50", round(wr_p50, 3))
    trial.set_user_attr("profit_factor_p50", round(pf_p50, 3))
    trial.set_user_attr("trades_per_day", round(trades_per_day, 1))
    trial.set_user_attr("n_trades", len(modified))
    st = stats(modified)
    trial.set_user_attr("naive_sharpe", round(st["sharpe"], 2))
    trial.set_user_attr("total_pnl_pct", round(st["total_pnl"], 2))
    return score


def print_results(study: optuna.Study):
    best = study.best_trial
    print(f"\n{'='*65}")
    print(f"  BEST TRIAL  |  Score: {best.value:.2f}")
    print(f"{'='*65}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k:30s} = {v}")
    print(f"  MC Metrics:")
    for k, v in best.user_attrs.items():
        print(f"    {k:30s} = {v}")
    print()

    top5 = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)[:5]
    print(f"  Top-5 trials:")
    print(f"  {'Rank':<5} {'Sharpe':<10} {'Ruin':<8} {'DD(p95)':<10} {'WR':<8} {'PF':<8} {'T/day':<8} {'Lev':<6} {'Stake':<8} {'Dir':<6}")
    print(f"  {'-'*79}")
    for i, t in enumerate(top5):
        v = t.value or 0
        sp = t.user_attrs.get("sharpe_p50", 0)
        rp = t.user_attrs.get("ruin_prob", 0)
        dd = t.user_attrs.get("max_dd_p95", 0)
        wr = t.user_attrs.get("win_rate_p50", 0)
        pf = t.user_attrs.get("profit_factor_p50", 0)
        tpd = t.user_attrs.get("trades_per_day", 0)
        lev = t.params.get("leverage", "?")
        stake = t.params.get("stake_pct", "?")
        short = t.params.get("short_only", False)
        long_ = t.params.get("long_only", False)
        direction = "short" if short else ("long" if long_ else "both")
        print(f"  {i:<5} {sp:<10.2f} {rp:<8.1%} {dd:<10.1%} {wr:<8.1%} {pf:<8.2f} {tpd:<8.1f} {lev:<6} {stake:<8.2f} {direction:<6}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--study-name", default="phoenix_mc")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--mc-sims", type=int, default=3000)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--days-back", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"\n  Optuna + MC Risk Optimizer")
    print(f"  DB: {args.db} | Study: {args.study_name} | Trials: {args.trials}")
    print()

    trades_raw = load_trades(args.db, min_trades=args.min_trades, days_back=args.days_back)
    if not trades_raw:
        print("Not enough trades — aborting")
        return

    baseline = to_trade_results(trades_raw, 50, 0.10)
    st = stats(baseline)
    print(f"  Baseline (50x, 10%% stake):")
    print(f"    {st['n']} trades | WR: {st['wr']:.1%} | Sharpe: {st['sharpe']:.2f}")
    print(f"    PnL: {st['total_pnl']:.2f}% | PF: {st['pf']:.2f}")
    print()

    validator = MonteCarloValidator(n_simulations=args.mc_sims)
    storage = f"sqlite:///{os.path.dirname(args.db)}/optuna_{args.study_name}.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=15),
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=10),
        load_if_exists=args.resume,
    )
    study.optimize(
        lambda t: objective(trades_raw, validator, t),
        n_trials=args.trials,
        n_jobs=1,
        show_progress_bar=True,
    )
    print_results(study)

    out = os.path.join(os.path.dirname(args.db), f"optimal_params_{args.study_name}.json")
    with open(out, "w") as f:
        json.dump({
            "params": study.best_trial.params,
            "metrics": {k: v for k, v in study.best_trial.user_attrs.items()},
            "timestamp": datetime.utcnow().isoformat(),
        }, f, indent=2)
    print(f"  Saved: {out}\n")


if __name__ == "__main__":
    main()
