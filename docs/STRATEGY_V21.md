# PhoenixScalperV2.1 — Strategy Whitepaper

## 1. Overview

Multi-asset futures scalper using a 10-component composite score to filter entry signals. Designed for 5m timeframe on Binance Futures with 10× isolated margin.

**Target**: PF > 1.8, Win Rate > 55%, Max DD < 15%

## 2. Entry Logic

### 2.1 Entry Patterns (populate_entry_trend)

Seven distinct entry patterns set `enter_long` or `enter_short` flags:

| # | Pattern | Side | Key Conditions |
|---|---------|------|----------------|
| 1 | Pullback | LONG | Price touches EMA_slow, close > EMA_slow, bullish candle, RSI 35-55, volume spike, ADX > threshold, +DI > -DI |
| 2 | RSI Momentum | LONG | RSI > 50, bullish candle, close > EMA_fast, volume spike, ADX > threshold |
| 3 | Momentum Breakout | LONG | Close > 5-bar high, volume 1.5× spike, ADX > 1.2× threshold, close > EMA_slow |
| 4 | Kalman Continuation | LONG | KF trend > 0, KF confidence > 0.6, close > KF prediction, KF acceleration > 0 |
| 5 | Breakdown | SHORT | Close < lookback-period low, volume spike, ADX elevated, -DI > +DI, RSI < threshold |
| 6 | Rally Failure | SHORT | High touches EMA_slow, close < EMA_slow, RSI 55-75, -DI > +DI |
| 7 | Bear Momentum | SHORT | Bearish candle, volume 1.3× spike, -DI > +DI, RSI < 45, close < EMA_slow |

### 2.2 Score System (`_calculate_entry_score`)

Each candle receives two scores (0-100):

**Signal Score** (long entry quality):
| Component | Weight | Max Pts | Calculation |
|-----------|--------|---------|-------------|
| HMM Bull Confidence | 20 pts | 20 | `min(hmm_p_bull/0.6, 1.0) * 20` |
| Trend Strength (ADX) | 15 pts | 15 | `min(ADX/40, 1.0) * 15` |
| Kalman Confidence | 10 pts | 10 | `min(kf_confidence/0.8, 1.0) * 10` |
| Directional (+DI - -DI) | 10 pts | 10 | `clip(diff/20+0.5, 0, 1) * 10` |
| Price Momentum | 10 pts | 10 | `avg(kf_accel, kf_momentum) * 10` |
| Volume Ratio | 10 pts | 10 | `min(volume_ratio/3.0, 1.0) * 10` |
| Regime Stability | 10 pts | 10 | `(1 - min(stability/0.5, 1.0)) * 10` |
| RSI Position | 5 pts | 5 | `max(0, 1 - |RSI-45|/25) * 5` |
| Pullback from EMA | 5 pts | 5 | `max(0, 1 - |close/EMA_slow-1|/0.02) * 5` |
| Trend Alignment | 5 pts | 5 | `avg(close>EMA_fast, close>EMA_slow, close>EMA_50) * 5` |
| **Total** | | **100** | |

**Short Score**: Same structure mirrored for bear-side (hmm_p_bear, -DI > +DI, negative momentum, RSI near 70, breakdown from EMA).

### 2.3 Score Gating

```python
# Score ceiling: clamp scores > 58 to 58
# (strong signals survive the override gate but don't get nullified)
dataframe.loc[dataframe["signal_score"] > 58, "signal_score"] = 58.0
dataframe.loc[dataframe["short_score"] > 58, "short_score"] = 58.0

# Threshold: reject signals below score_threshold (default 55)
threshold = self.score_threshold.value
dataframe.loc[
    (dataframe["enter_long"] == 1) & (dataframe["signal_score"] < threshold),
    ["enter_long", "enter_tag"]
] = (0, None)
```

**Critical Fix**: Original code set scores > 58 to 0.0 (nullifying strong signals). Changed to `= 58.0` (clamp to ceiling). This was the root cause of zero trades when strong signals existed.

### 2.4 Entry Tag Format

`"{pattern_name} [{score}]"` — e.g., `"short_bear_momentum [58]"`

Score is appended to help debug signal quality in Telegram logs.

## 3. Exit Logic

### 3.1 Take Profit (`custom_exit`)

```
TP Target: 7.1% → return "tp_hit"
```

When profit reaches 7.1% of entry (pre-leverage), trade exits immediately.

### 3.2 Bleed Exit

```
Bleed Loss: -3.3%
Bleed Time: 289 minutes (4.8 hours)
→ return "bleed_exit"
```

If loss exceeds 3.3% AND trade has been open > 4.8 hours, exit.

### 3.3 Max Hold

```
Max Hold: 303 minutes (5.05 hours)
→ return "max_hold"
```

Forced exit regardless of P&L after 5 hours.

### 3.4 Trailing Stop Loss (`custom_stoploss`)

```python
trail_threshold = 0.078  # 7.8% profit
lock_ratio = 0.359       # lock 35.9% of profit
if current_profit > trail_threshold:
    lock_equity = max(current_profit * lock_ratio, trail_threshold * 0.5)
    return -(lock_equity / trade.leverage)
```

Hard stop: -0.99 (99% of position, i.e., no stop) during initial fill.

### 3.5 Exit Hierarchy

```
tp_hit (best) → trailing_stop_loss (good) → max_hold (forced)
                                         → bleed_exit (worst)
```

## 4. Risk Management

### 4.1 Position Sizing

```python
kelly_75_pct = 0.10  # 10% of wallet per trade
stake = max(min_stake, min(wallet * 0.10, max_stake))
```

### 4.2 Consecutive Loss Breaker

Resets daily. Rejects entries after MAX_CONSECUTIVE_LOSSES losses (set to 999 = disabled by default).

### 4.3 Pair Cooldown

After a losing exit, the same pair is blocked for 30 minutes.

### 4.4 Regime-Adaptive Max Trades

```python
regime_max = {
    "strong_bear": 10, "weak_bear": 10,
    "low_volatility": 7,
    "weak_bull": 3, "strong_bull": 3
}
new_max = regime_max.get(regime_str, 5)
```

In bear regimes the bot opens more trades (short-biased); in bull regimes it opens fewer.

### 4.5 Drift Mode

| Drift Level | PSI Threshold | Effect |
|-------------|---------------|--------|
| Normal | < 0.5 | Full operation |
| Warning | 0.5 - 2.0 | Reduced max_trades |
| Critical | > 2.0 | Minimal trading (max 3 trades) |

## 5. Leverage

Fixed at 10× for all trades.

## 6. Key Parameters

| Parameter | Value | Range | Description |
|-----------|-------|-------|-------------|
| timeframe | 5m | fixed | Candlestick period |
| startup_candle_count | 100 | — | Warmup for indicators |
| stoploss | -0.12 | — | Initial hard stop |
| tp_target | 0.071 | fixed | Take profit target |
| bleed_loss | 0.033 | fixed | Bleed loss threshold |
| bleed_time | 289 min | fixed | Bleed time threshold |
| max_hold_min | 303 min | fixed | Maximum hold time |
| trail_threshold | 0.078 | fixed | Trailing stop trigger |
| lock_ratio | 0.359 | fixed | Profit lock ratio |
| score_threshold | 55 | 35-80 | Minimum entry score |
| adx_threshold | 20 | 15-25 | Minimum ADX for entry |
| rsi_period | 7 | 5-10 | RSI calculation period |
| sl_min | 0.0025 | 0.0015-0.0035 | Min ATR stop multiplier |
| sl_max | 0.0050 | 0.0035-0.0060 | Max ATR stop multiplier |
