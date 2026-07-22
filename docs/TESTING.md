# PHOENIX TRADING SYSTEM — Test Suite & Validation

## 1. Test Infrastructure

Test directory: `phoenix-scalper/tests/`

```
tests/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── test_champion_challenger.py
│   ├── test_concept_drift.py
│   ├── test_data_quality.py
│   ├── test_ml_engine.py
│   ├── test_monitoring.py
│   ├── test_regime_engine.py
│   ├── test_risk_governor.py
│   ├── test_strategy_allocator.py
│   ├── test_validation_pipeline.py
│   └── test_validator_e2e.py
├── ml/
│   ├── __init__.py
│   ├── test_hmm_regime.py
│   ├── test_kalman_filter.py
│   └── test_monte_carlo.py
└── strategies/
```

## 2. Running Tests

```bash
# Inside the container
docker exec phoenix-scalper-v2.1-bot python3 -m pytest tests/ -v

# Run specific module
docker exec phoenix-scalper-v2.1-bot python3 -m pytest tests/core/test_risk_governor.py -v

# Run with coverage
docker exec phoenix-scalper-v2.1-bot python3 -m pytest tests/ --cov=core --cov=ml -v
```

## 3. Test Coverage by Module

### 3.1 Core Modules

| Module | File | Tests | Coverage |
|--------|------|-------|----------|
| DataValidator | `test_data_quality.py` | Validates candles, stale data, NaN detection, price/volume anomalies | High |
| RegimeEngine | `test_regime_engine.py` | 7-state classification, confidence scoring, regime transitions | High |
| RiskGovernor | `test_risk_governor.py` | Risk levels, drawdown thresholds, consecutive loss breaker, state transitions | High |
| ConceptDriftDetector | `test_concept_drift.py` | PSI computation, KL divergence, Wasserstein distance, reference/update cycle | High |
| MLEngine | `test_ml_engine.py` | Baseline setting, retrain triggers, drift integration | Medium |
| StrategyAllocator | `test_strategy_allocator.py` | Capital allocation by regime, max trades per regime | Medium |
| ChampionChallenger | `test_champion_challenger.py` | A/B test framework, promotion logic | Medium |
| ValidationPipeline | `test_validation_pipeline.py` | Full pipeline: backtest → walk-forward → MC → shadow | Medium |
| E2E Validator | `test_validator_e2e.py` | End-to-end validation flow | Medium |
| Monitor | `test_monitoring.py` | Telegram message formatting, notification dispatch | Low |

### 3.2 ML Modules

| Module | File | Tests | Coverage |
|--------|------|-------|----------|
| HMM Regime | `test_hmm_regime.py` | Baum-Welch EM, Viterbi decoding, forward-backward, regime probabilities | High |
| Kalman Filter | `test_kalman_filter.py` | State prediction, adaptive noise, confidence scoring, trend detection | High |
| Monte Carlo | `test_monte_carlo.py` | Simulation shuffling, equity curves, drawdown, Sharpe calculation | High |

## 4. Test Patterns

### 4.1 DataValidator Tests

```python
def test_validates_valid_dataframe():
    """Given a complete DataFrame, validation passes with no issues."""
    df = create_sample_dataframe(100)
    validator = DataValidator()
    result = validator.validate_candles(df, "BTC/USDT:USDT")
    assert result["valid"] == True
    assert len(result["issues"]) == 0

def test_detects_nan_columns():
    """DataFrame with >10% NaN triggers warning."""
    df = create_sample_dataframe(100)
    df.loc[df.index[:20], "close"] = np.nan
    validator = DataValidator()
    result = validator.validate_candles(df, "BTC/USDT:USDT")
    assert "High NaN" in str(result["issues"])
```

### 4.2 RiskGovernor Tests

```python
def test_normal_to_reduced_on_drawdown():
    """Daily drawdown exceeding 5% triggers REDUCED risk level."""
    governor = RiskGovernor(max_daily_drawdown=0.05)
    governor.update(100.0, [])
    governor.update(94.0, [])  # 6% drawdown
    assert governor.get_state().level == RiskLevel.REDUCED

def test_emergency_on_multiple_triggers():
    """3+ triggers simultaneously triggers EMERGENCY."""
    governor = RiskGovernor(max_consecutive_losses=3, max_daily_drawdown=0.05)
    for _ in range(3):
        governor.record_trade_result(-2.0, 100)
    result = governor.update(85.0, [])
    assert result.level == RiskLevel.EMERGENCY
```

### 4.3 HMM Tests

```python
def test_hmm_fit_converges():
    """HMM Baum-Welch EM converges within 100 iterations."""
    detector = RegimeDetector(n_states=3, n_iter=100)
    obs = np.random.randn(500, 3)
    obs[:200] += 0.1   # Bull-like
    obs[200:400] = 0.0 # Range-like
    obs[400:] -= 0.1   # Bear-like
    detector.fit(obs)
    regimes = detector.predict_regime(obs)
    assert len(np.unique(regimes)) <= 3
```

## 5. Research Experiments

Additional validation in `phoenix-scalper/research/`:

| Script | Purpose |
|--------|---------|
| `walk_forward_validation.py` | Rolling walk-forward with expanding window |
| `monte_carlo.py` | Trade sequence shuffling (10k sims) |
| `optuna_mc_optimizer.py` | Optuna hyperopt with MC validation |
| `tp_sl_precision.py` | TP/SL parameter precision analysis |
| `feature_importance.py` | LightGBM feature importance analysis |
| `quant_agent.py` | Research hypothesis generation agent |
| `stress_test.py` | Extreme market condition simulation |
