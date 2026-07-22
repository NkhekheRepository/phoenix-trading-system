# PHOENIX TRADING SYSTEM — Machine Learning Modules

## 1. Hidden Markov Model (HMM) — `ml/hmm_regime.py`

### 1.1 Overview

3-state Gaussian HMM for market regime detection.

**States:**
| State | Description | mu (log_return) | sigma (vol) | sigma (volume) |
|-------|-------------|-----------------|-------------|----------------|
| 0 | Trending Bull | +0.001 | 0.02 | 0.3 |
| 1 | Ranging | 0.0 | 0.05 | 0.5 |
| 2 | Trending Bear | -0.001 | 0.02 | 0.5 |

**Transition Matrix (initial):**
```
        Bull    Range   Bear
Bull    0.85    0.10    0.05
Range   0.20    0.60    0.20
Bear    0.05    0.10    0.85
```

### 1.2 Training

Uses Baum-Welch (EM) algorithm via `_forward_backward` + `_m_step`:
- Forward-backward pass computes log-likelihood, gamma (state probabilities), xi (transition probabilities)
- M-step re-estimates mu, Sigma, A, pi
- 100 iterations, log every 20 (visible in logs)

### 1.3 Inference

```python
compute_hmm_features(returns, vol, volume_change)
```
Returns dict with:
- `hmm_regime`: Most likely state (0/1/2)
- `hmm_p_bull`: Probability of bull state
- `hmm_p_range`: Probability of range state
- `hmm_p_bear`: Probability of bear state
- `hmm_regime_stability`: Max probability — certainty of classification
- `hmm_transition_risk`: Sum of off-diagonal transition probabilities
- `hmm_vol_regime`: 1.0 (low), 1.5 (medium), 2.0 (high) based on vol observation
- `hmm_trend_strength`: Signal-to-noise ratio (|mu|/sigma of current state)

### 1.4 Caching

- V2.1: Trained once per pair on first `populate_indicators` call, cached forever in `_hmm_cache`
- V3.1: Trained once per pair, retrained if 24h stale (`_hmm_last_train` check)
- V5-BTC: Retrained every 60th call via `do_heavy` flag

### 1.5 Look-Ahead Fix

HMM is trained on the FULL observation sequence then used to predict on the SAME sequence. This creates minor look-ahead (the EM algorithm uses all observations). Mitigated by:
- Caching: model only retrained on new data
- Startup period: `startup_candle_count = 100` ensures sufficient warmup

## 2. HMM Regime V3 — `ml/hmm_regime_v3.py`

### 2.1 Differences from V2

| Feature | hmm_regime | hmm_regime_v3 |
|---------|------------|----------------|
| Input type | numpy arrays | Accepts pandas Series |
| Forward-backward | Standard | Uses `logsumexp` stabilization |
| State count | 3 | 3 (same) |
| Class design | Instance methods | Class-level `n_states` attribute |
| Numerical stability | Standard | Enhanced with `np.clip` on B matrix |

### 2.2 Usage

```python
from ml.hmm_regime_v3 import compute_hmm_features
hmm_features = compute_hmm_features(
    returns.fillna(0),     # pandas Series
    vol.fillna(0),         # pandas Series
    volume_change.fillna(0)
)
```

### 2.3 Features

Same output schema as V2 HMM: regime, probabilities, stability, transition_risk, vol_regime, trend_strength.

## 3. Kalman Filter — `ml/kalman_filter.py`

### 3.1 Overview

1D Kalman filter for price with adaptive noise estimation.

**State Vector:**
```
x = [price_level, price_trend]
```

**State Transition:**
```
F = [[1, 1],
     [0, 1]]
```

**Observation:**
```
H = [[1, 0]]
```

### 3.2 Adaptive Noise

```python
R_base = ATR_pct^2                    # Measurement noise ~ volatility
vol_adj = max(0.5, 1/(volume_ratio + 0.1))  # Volume adjustment
R = R_base * vol_adj * measurement_noise_factor
```

When volume is low, measurement noise increases (less confidence in price). When volume is high, noise decreases.

### 3.3 Output Features

| Feature | Description |
|---------|-------------|
| `kf_price` | Smoothed/filtered price (state x[0]) |
| `kf_trend` | Estimated trend direction (state x[1]) |
| `kf_prediction` | Next-period price forecast (F@x)[0] |
| `kf_confidence` | Confidence score 0-1: `1/(1 + |S|/R)` |
| `kf_direction` | +1 (up), -1 (down), 0 (flat) based on trend threshold |
| `kf_innovation` | Prediction error (z - H@x_pred) |
| `kf_S` | Innovation covariance |
| `kf_price_momentum` | Normalized trend: `trend / price` |
| `kf_trend_acceleration` | Trend change rate (2nd derivative proxy) |
| `kf_prediction_error` | Accumulated prediction error |
| `kf_regime_score` | Regime from KF perspective |
| `kf_vol_of_trend` | Volatility of trend estimate |
| `kf_atr_ratio` | Current ATR-to-price ratio |

## 4. Feature Engine — `ml/feature_engine.py`

### 4.1 Purpose

Generates training data for LightGBM winrate model. Extracts features from historical OHLCV + indicators and labels them with forward returns.

### 4.2 Feature Extraction

For each entry point (candle index i):
- **Price features**: close, open, high, low, candle range, body ratio, close/open ratio
- **Technical**: EMA slopes, RSI level, BB position, volume ratio, ADX
- **HMM features**: regime probabilities, stability, transition risk
- **Kalman features**: trend, confidence, momentum, acceleration

### 4.3 Target Creation

```python
max_profit = (future_high_max - entry_price) / entry_price
max_drawdown = (entry_price - future_low_min) / entry_price
# Target is a function of max_profit and max_drawdown
```

Forward window: 24 bars (2 hours at 5m).

### 4.4 Usage

```python
from ml.feature_engine import FeatureEngine
engine = FeatureEngine(forward_bars=24, min_samples=50000)
training_df = engine.generate_training_data(dataframe)
```

## 5. Monte Carlo Validator — `ml/monte_carlo.py`

### 5.1 Purpose

Validates strategy robustness by simulating thousands of shuffled trade sequences. Measures probability of achieving key metrics by random chance.

### 5.2 Simulations

- Default: 10,000 permutations
- Each simulation shuffles trade profits and re-calculates equity curve
- Tracks: max drawdown, Sharpe, Calmar, final equity, ruin probability, max win/loss streaks

### 5.3 Output

```python
results = {
    'max_dd': [],         # List of max drawdowns across sims
    'sharpe': [],         # Sharpe ratios
    'calmar': [],         # Calmar ratios
    'final_equity': [],   # Final equity values
    'ruin_prob': [],      # Probability of ruin (>50% drawdown)
    'win_streak_max': [],
    'loss_streak_max': [],
    'win_rate': [],
    'avg_win': [], 'avg_loss': [],
    'profit_factor': [],
}
```

## 6. Model Training Scripts

### `train_winrate_model.py`
Trains a LightGBM classifier to predict trade win probability based on entry features. Output saved to `ml/models/winrate_model.pkl`.

### `train_strategy_model.py`
Trains a LightGBM regressor for TP/SL optimization based on entry features and regime state.

## 7. Dependencies

```
lightgbm==4.6.0
joblib==1.5.3
scikit-learn==1.9.0 (phoenix30 only)
numpy
pandas
scipy
```

Installed in Docker image via `pip install` in Dockerfile.
