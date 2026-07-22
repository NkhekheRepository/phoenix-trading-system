# PHOENIX TRADING SYSTEM — Core Framework Modules

## 1. RiskGovernor — `core/risk_governor.py`

### 1.1 Purpose

Independent risk management layer that monitors drawdowns, exposure, consecutive losses, and leverag e. Adjusts risk posture in real-time.

### 1.2 Risk Levels

| Level | Triggers | leverage_mult | max_trades | stake_mult |
|-------|----------|---------------|------------|------------|
| NORMAL | 0 triggers | 1.0× | 5 | 1.0× |
| REDUCED | 1 trigger | 0.5× | 2 | 0.5× |
| PAUSED | 2 triggers | 0.25× | 1 | 0.25× |
| EMERGENCY | 3+ triggers | 0.0× | 0 | 0.0× |

### 1.3 Trigger Thresholds

| Trigger | Default Threshold |
|---------|-------------------|
| Daily Drawdown | 5% |
| Weekly Drawdown | 10% |
| Consecutive Losses | 5 |
| Daily Loss | 8% |
| Exposure | 80% of wallet |
| Leverage | 90% of max leverage |

### 1.4 Integration

Called in `bot_loop_start()` every iteration:
```python
risk_state = self._risk_governor.update(balance, open_trades)
```

## 2. DataValidator — `core/data_quality.py`

### 2.1 Purpose

Validates OHLCV data quality before indicator computation.

### 2.2 Checks

| Check | Description | Threshold |
|-------|-------------|-----------|
| Missing candles | Gap detection | > 15 min gap |
| Duplicate timestamps | Exact timestamp duplicates | ≥ 1 |
| Stale data | Last candle age | > 10 min |
| NaN values | Percentage of NaN per column | > 10% |
| Infinity values | Inf in numeric columns | ≥ 1 |
| Price anomalies | Z-score on returns | > 5.0 |
| Volume anomalies | Modified Z-score on volume | > 5.0 |

### 2.3 Severity Levels

| Level | Issues | Action |
|-------|--------|--------|
| fatal | Empty/No date | Skip pair |
| critical | > 5 issues | Alert + mitigate |
| warning | 2-5 issues | Alert only |
| info | 1 issue | Log only |
| ok | 0 issues | No action |

### 2.4 Integration

```python
if pair_key not in self._hmm_cache:  # First call only
    self._data_validator.validate_candles(dataframe, pair_key)
```

### 2.5 DatasetLineage

Tracks data provenance:
- dataset_id, exchange, pair, timeframe
- start_date, end_date, candle_count
- feature_version, training_version, validation_version
- checksum (future use)

## 3. RegimeEngine — `core/regime_engine.py`

### 3.1 Purpose

Classifies market into 7 states using HMM probabilities + trend indicators.

### 3.2 States

| State | Description | Typical Action |
|-------|-------------|----------------|
| strong_bear | Bear trend, high confidence | Increase shorts |
| weak_bear | Bearish tendency | Prefer shorts |
| low_volatility | No clear direction | Reduce position size |
| neutral | Balanced | Wait for setup |
| weak_bull | Bullish tendency | Prefer longs |
| strong_bull | Bull trend, high confidence | Increase longs |

### 3.3 Input Features

- HMM regime probabilities (p_bull, p_bear, p_range)
- Regime stability
- ADX trend strength
- Kalman trend direction + confidence

### 3.4 Integration

Called in `populate_indicators`:
```python
regime_result = self._regime_engine.analyze(dataframe)
self._last_regime_str = regime_result.regime.value
```

## 4. MLEngine — `core/ml_engine.py`

### 4.1 Purpose

Manages ML retraining triggers. Monitors concept drift and performance degradation to decide when to retrain the winrate model.

### 4.2 Retrain Triggers

- **Performance degradation**: Win rate or avg loss exceeds baseline thresholds
- **Concept drift**: PSI/KL/Wasserstein exceed thresholds
- **Trade count threshold**: 20 trades needed for baseline

### 4.3 Baseline

Set after 20 closed trades:
```python
baseline = {
    "win_rate": actual_win_rate,
    "avg_loss_pct": abs(avg_loss_pct)
}
```

### 4.4 Integration

```python
if self._ml_baseline_set:
    triggers = self._ml_engine.check_retrain_triggers()
```

## 5. ConceptDriftDetector — `core/concept_drift.py`

### 5.1 Purpose

Statistical drift detection using Population Stability Index (PSI), KL Divergence, and Wasserstein Distance.

### 5.2 Metrics

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| PSI | 0.2 | Distribution shift magnitude |
| KL Divergence | 0.1 | Information-theoretic shift |
| Wasserstein Distance | 0.5 | Earth mover's distance |

### 5.3 Features Monitored

- `hmm_p_bull` — Bull regime probability distribution
- `hmm_p_bear` — Bear regime probability distribution
- `hmm_regime_stability` — Certainty distribution

### 5.4 Integration

```python
# Feed features every 10th iteration (V2.1) or every iteration (V3.1)
self._feed_concept_drift(dataframe)
```

## 6. TradeIntelligence — `core/trade_intel.py`

### 6.1 Purpose

Full trade attribution with win/loss pattern analysis. Tracks entry/exit conditions, regime, and risk level for each trade.

### 6.2 Key Methods

- `start_trade()`: Record trade entry with full context
- `close_trade()`: Record exit with P&L attribution
- `analyze_winning_patterns()`: Identify common factors in winners
- `analyze_losing_patterns()`: Identify common factors in losers
- `get_trade_count()`: Total trades tracked

## 7. Monitor — `core/monitoring.py`

### 7.1 Purpose

Telegram-based monitoring and alerting. Sends daily summaries, hourly health checks, and real-time alerts.

### 7.2 Notifications

| Notification | Frequency | Content |
|-------------|-----------|---------|
| Daily Summary | Once daily | Regime, risk, exposure, P&L, win/loss count |
| Hourly Health | Every hour | Uptime, active trades, memory, exchange status |
| Data Quality | On detection | Stale data, NaN anomalies, gaps |
| Risk Change | On level change | Risk level transition with triggers |
| Trade Alert | On entry/exit | Trade details with P&L |

### 7.3 Integration

```python
self._monitor = Monitor(
    dp=self.dp,
    bot_name="phoenix-scalper",
    chat_id=config["telegram"]["chat_id"],
    token=config["telegram"]["token"],
)
```

## 8. Telegram EV Command — `core/telegram_ev.py`

### 8.1 Purpose

Registers a custom `/ev` Telegram command for "Expected Value" analysis — shows current PF, WR, expectancy, and risk metrics on demand.

## 9. Deployment Manager — `core/deployment.py`

### 9.1 Purpose

Manages safe deployment lifecycle:

1. **Backtest** — Historical validation
2. **Walk-Forward** — Rolling validation
3. **Monte Carlo** — Robustness check
4. **Shadow Mode** — Paper trade alongside live
5. **Canary** — 10% capital allocation
6. **Full Deployment** — 100% capital
7. **Monitor** — Continuous drift + risk check
8. **Rollback** — Automatic on performance degradation

## 10. ChampionChallenger — `core/champion_challenger.py`

### 10.1 Purpose

A/B testing framework for strategy parameters. Runs champion (current best) against challenger (candidate) and promotes if challenger outperforms.

## 11. MarketMemory — `core/market_memory.py`

### 11.1 Purpose

Long-term knowledge storage of market patterns. Records regime transitions, volatility regimes, and performance by regime for pattern matching.

## 12. ExperimentDB — `core/experiment_db.py`

### 12.1 Purpose

Versioned experiment tracking for hyperopt and parameter optimization. Stores all experiment configurations and results for auditability.
