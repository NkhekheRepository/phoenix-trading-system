# PhoenixScalperV3.1 — Strategy Whitepaper

## 1. Overview

Adaptive multi-asset futures scalper evolved from V2.1. Adds score gating, drift-aware threshold adaptation, and HMM v3 for regime detection. Same 8-pair 5m Binance Futures setup.

## 2. Key Differences from V2.1

| Feature | V2.1 | V3.1 |
|---------|------|------|
| Score system | 10-component composite score | Same structure, but gating instead of ceiling |
| Entry filtering | Score ceiling at 58 | Score gating + ADX/RVOL override |
| HMM variant | hmm_regime (V2) | hmm_regime_v3 (optimized) |
| HMM retrain | Cached per pair | Cached per pair + 24h stale check |
| Regime adaptation | Bot loop adjusts max_trades | Same + drift-adjusted thresholds |
| ML baseline | At 20 trades | Same |
| process_only_new_candles | True (config) | False (config) |
| strong_bull max_trades | 3 | **0** (no trades in strong bull) |
| confirm_trade_entry | No score ceiling check | **Score ceiling rejection** (scores > 59 rejected) |

## 3. Entry Logic

### 3.1 Entry Patterns

Identical 7-entry-pattern system as V2.1:

1. **Pullback** (LONG) — EMA bounce
2. **RSI Momentum** (LONG) — Momentum continuation
3. **Momentum Breakout** (LONG) — 5-bar high breakout
4. **Kalman Continuation** (LONG) — KF-filtered trend
5. **Breakdown** (SHORT) — Support break
6. **Rally Failure** (SHORT) — EMA rejection
7. **Bear Momentum** (SHORT) — Bearish continuation

### 3.2 Score Gating (V3.1 Only)

V3.1 removes the per-pattern score ceiling and adds a **global override gate**:

```python
has_entry = (dataframe["enter_long"] == 1) | (dataframe["enter_short"] == 1)
score_gate = (
    (dataframe["adx"] > 25) |
    (dataframe["volume_ratio"] > 0.5) |
    (dataframe["plus_di"].abs() - dataframe["minus_di"].abs() > 5)
)
dataframe.loc[has_entry & ~score_gate, ["enter_long", "enter_short"]] = 0
```

This rejects signals when:
- ADX ≤ 25 (no trend)
- Volume ratio ≤ 0.5 (no volume confirmation)
- |+DI - -DI| ≤ 5 (no directional conviction)

### 3.3 Score Ceiling Rejection (confirm_trade_entry)

V3.1 additionally rejects at trade confirmation time if score > 59:

```python
score_m = _re.search(r'\[(\d+)\]', entry_tag or "")
if score_m:
    sc = int(score_m.group(1))
    if sc > 59:
        logger.info(f"Score ceiling: {sc} > 59, rejecting {pair} {side}")
        return False
```

This acts as a final sanity check — scores > 59 are rejected entirely.

## 4. Regime Adaptation

### 4.1 Regime-Aware Max Trades

```python
regime_max = {
    "strong_bear": 10, "weak_bear": 10,
    "low_volatility": 7,
    "weak_bull": 3, "strong_bull": 0   # NOTE: 0 in strong bull
}
```

### 4.2 Drift-Aware Threshold Adjustment

| Drift Level | PSI | score_threshold | score_high_threshold |
|-------------|-----|-----------------|---------------------|
| Normal | < 0.5 | 55 | 60 |
| Warning | 0.5 - 2.0 | **58** | **62** |
| Critical | > 2.0 | **60** | **65** |

In high-drift regimes, entry thresholds tighten to avoid false signals.

## 5. Exit Logic

### 5.1 Exit Conditions

Identical to V2.1:

| Reason | Condition | Priority |
|--------|-----------|----------|
| tp_hit | profit ≥ 7.1% | 1 (best) |
| trailing_stop_loss | lock profit after 7.8% trigger | 2 |
| max_hold | elapsed > 303 min | 3 |
| bleed_exit | profit < -3.3% AND elapsed > 289 min | 4 (worst) |

### 5.2 Performance Analysis (13 trades)

| Metric | Value |
|--------|-------|
| Closed trades | 11 |
| Win Rate | 63.6% (7W/4L) |
| Profit Factor | 1.66 |
| Total PnL | +$1.30 |
| Avg Win | +5.32% |
| Avg Loss | -4.98% |
| Max Win | +7.57% (ADA long) |
| Max Loss | -9.70% (XRP trailing_stop_loss) |
| Best exit | tp_hit (4×, avg +7.37%) |
| Worst exit | trailing_stop_loss (1×, -9.70%) |

**Key insight**: A single trailing_stop_loss at -9.70% accounts for 49% of all gross losses. Removing it, PF jumps from 1.66 to 3.26.

## 6. ML Components

### 6.1 HMM Regime v3

V3 variant of HMM regime detection. Trained once per pair (cached, 24h stale check). Produces same 6 feature columns as V2 HMM.

### 6.2 Kalman Filter

Same as V2.1 with 24h stale retrain check (`kalman_stale`).

### 6.3 Concept Drift Detection

Feeds HMM probabilities (p_bull, p_bear, stability) into PSI/KL/Wasserstein detectors. Reference set after 50 samples.

## 7. Risk Management

- **Same as V2.1**: 10% Kelly position sizing, pair cooldown, loss breaker
- **Additional**: Regime-adaptive max_trades (0 in strong bull)
- **Additional**: Score ceiling rejection at confirm_trade_entry
- **Additional**: Drift-aware threshold tightening

## 8. Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| score_threshold | 55 | Increased to 58/60 in drift warning/critical |
| score_high_threshold | 60 | Increased to 62/65 in drift warning/critical |
| adx_threshold | 20 | Minimum for entry |
| short_adx_mult | 1.258 | ADX multiplier for short entries |
| short_volume_mult | 1.931 | Volume multiplier for short entries |
| tp_target | 0.071 (7.1%) | Take profit |
| max_hold_min | 303 | Maximum hold time |
