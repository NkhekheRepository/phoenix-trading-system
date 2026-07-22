# PHOENIX TRADING SYSTEM — Research & Experiments

## 1. Research Framework

The system includes a dedicated research suite for quantitative analysis:

```
phoenix-scalper/research/
├── __init__.py
├── quant_agent.py                # Automated hypothesis generation
├── run_experiment.py             # Run parameter experiments
├── run_btc_only.py               # BTC-only backtests
├── run_v5.py                     # V5 backtests
├── run_v5_suite.py               # V5 multi-config suite
├── walk_forward_validation.py    # Walk-forward analysis
├── feature_importance.py         # Feature importance analysis
├── experiment_results.json       # V2/V3 experiment results
├── v5_experiment_results.json    # V5 experiment results
├── walk_forward_result.json      # Walk-forward results
├── feature_importance_result.json# Feature importance results
├── btc_only_results.json         # BTC-only results
```

## 2. Key Findings

### 2.1 Feature Importance

From `feature_importance_result.json` (LightGBM analysis):

**Top predictors of trade success:**
1. `kf_confidence` — Kalman Filter confidence score
2. `hmm_p_bull` — HMM bull regime probability
3. `hmm_regime_stability` — HMM classification certainty
4. `adx` — Trend strength
5. `volume_ratio` — Volume confirmation
6. `rsi_5` — Short-term RSI

**Low-impact features:**
- EMA crossovers (already captured by trend alignment score)
- BB width (redundant with ATR)

### 2.2 Optimal TP/SL Parameters

From `tp_sl_precision.py` analysis and hyperopt:

| Parameter | Range Tested | Optimal |
|-----------|-------------|---------|
| tp_target | 0.03 - 0.12 | **0.071** (7.1%) |
| bleed_loss | 0.02 - 0.05 | **0.033** (3.3%) |
| bleed_time | 120 - 360 min | **289 min** (~4.8h) |
| max_hold_min | 120 - 480 min | **303 min** (~5h) |
| trail_threshold | 0.05 - 0.12 | **0.078** (7.8%) |
| lock_ratio | 0.20 - 0.50 | **0.359** (35.9%)|

### 2.3 V2/V3 Experiment Results

From `experiment_results.json`:

```
V2.1 (pre-fix):
  Trades: ~87
  WR: 59.8%
  PF: 0.32
  Expectancy: -0.31

V3.1 (pre-fix):
  Trades: ~87 (same data)
  WR: ~55%
  PF: ~0.40
  Expectancy: -0.25
```

**Note**: Pre-fix results include the score ceiling bug (Fix #1) that nullified strong signals. Post-fix performance is expected to be significantly higher.

## 3. Walk-Forward Validation

From `walk_forward_result.json`:

```
Method: Expanding window, 6-month train / 1-month test
Years: 2025-2026

Results:
  Mean PF (validation): 1.2-1.8
  Mean WR (validation): 52-62%
  Max DD (validation): 12-18%
  Out-of-sample PF decay: 15-25%
```

## 4. Monte Carlo Simulation

```python
n_simulations = 10,000
trades_per_year = ~1,500 (estimated from 8 pairs × 5m)

Results distribution:
  PF 5th percentile: 0.85
  PF 50th percentile: 1.45
  PF 95th percentile: 2.10
  Ruin probability (< -50% DD): 3.2%
```

## 5. Research Scripts Usage

### 5.1 Walk-Forward Validation

```bash
docker exec phoenix-scalper-v2.1-bot python3 -m research.walk_forward_validation
```

### 5.2 Feature Importance

```bash
docker exec phoenix-scalper-v2.1-bot python3 -m research.feature_importance
```

### 5.3 Run Experiments

```bash
# Run V2/V3 experiment suite
docker exec phoenix-scalper-v2.1-bot python3 -m research.run_experiment

# Run V5 experiment suite
docker exec phoenix-scalper-v2.1-bot python3 -m research.run_v5_suite

# BTC-only experiment
docker exec phoenix-scalper-v2.1-bot python3 -m research.run_btc_only
```

### 5.4 TP/SL Optimization

```bash
docker exec phoenix-scalper-v2.1-bot python3 -m research.tp_sl_optimizer
```

## 6. Tools Directory

Additional research tools:

```
phoenix-scalper/tools/
├── ev_core.py          # Expected value calculation core
├── ev_tracker.py       # Real-time EV tracking
├── gated_backtest.py   # Gated backtesting with regime filter
├── observer.py         # Market observation tool
```

## 7. Hyperopt Results

`user_data/hyperopt_results/` contains 11+ `.fthypt` files from Optuna and scikit-optimize runs. Key parameter ranges explored:

| Parameter | Range | Best Found |
|-----------|-------|------------|
| score_threshold | 35-80 | 55 |
| adx_threshold | 15-25 | 20 |
| volume_factor | 1.0-2.0 | 1.2 |
| short_volume_mult | 1.5-3.0 | 2.0 |
| short_adx_mult | 1.0-1.5 | 1.3 |
| atr_sl_mult | 0.5-1.2 | 0.7 |
| hmm_target | 0.35-0.80 | 0.55 |
| hmm_bull_target | 0.50-1.20 | 0.80 |
| ema_fast | 5-10 | 7 |
| ema_slow | 12-22 | 15 |
| rsi_period | 5-10 | 7 |

Loss function: `-sharpe_ratio` (for hyperopt) and `-profit_factor` (for TP/SL search).
