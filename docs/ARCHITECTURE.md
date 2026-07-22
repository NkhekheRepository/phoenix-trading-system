# PHOENIX TRADING SYSTEM — Architecture

## System Topology

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCKER HOST                                   │
│                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐ │
│  │  V2.1 Bot (8082)    │  │  V3.1 Bot (8083)    │  │ V5-BTC (8085)│ │
│  │  PhoenixScalperV2_1 │  │  PhoenixScalperV3_1  │  │PhoenixScalper│ │
│  │  8 pairs, 10x lev   │  │  8 pairs, 10x lev   │  │V5_BTC        │ │
│  │  GOD MODE           │  │  TSIMBI             │  │1 pair, 10x   │ │
│  └────────┬────────────┘  └────────┬────────────┘  └──────┬───────┘ │
│           │                        │                       │         │
│           └──────────┬─────────────┘───────────────────────┘         │
│                      │                                              │
│              ┌───────▼────────┐                                      │
│              │  Shared Volume  │                                      │
│              │  scalper_data   │                                      │
│              │  (feather + DB) │                                      │
│              └───────┬────────┘                                      │
│                      │                                              │
│              ┌───────▼────────┐                                      │
│              │  Binance API   │                                      │
│              │  (Futures)     │                                      │
│              └────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        CRON JOBS (host)                              │
│                                                                      │
│  */5 * * * *   health_check.py    — Bot liveness + Telegram alert   │
│  0 * * * *     regime_alert.py    — BTC market regime analysis      │
│  3 * * * *     refresh_data.sh    — Download latest 5m candles      │
└─────────────────────────────────────────────────────────────────────┘
```

## Per-Bot Component Architecture

Each bot container runs a Freqtrade instance with:

```
freqtrade trade
  --config /freqtrade/config.json      # Bind-mounted from host
  --strategy PhoenixScalperVx_x         # Python class in /freqtrade/strategies/
  --strategy-path /freqtrade/strategies/
  --db-url sqlite:///.../tradesv3_vx_x.sqlite
  --logfile .../logs/freqtrade_vx_x.log
```

### Strategy Class Hierarchy (V2.1 / V3.1)

```
IStrategy (freqtrade)
  └─ PhoenixScalperV2_1 / PhoenixScalperV3_1
       ├── populate_indicators()
       │    ├── Technical: RSI, EMA, ADX, MACD, BB, OBV, VWAP, ATR
       │    ├── KalmanFilter: price smoothing + trend + confidence
       │    ├── HMM Regime: 3-state market regime detection
       │    ├── RegimeEngine: 7-state regime classification
       │    ├── _calculate_entry_score()  (V2.1 only)
       │    └── ConceptDriftDetector: PSI/KL/Wasserstein drift
       │
       ├── populate_entry_trend()
       │    ├── Long: pullback, rsi_momentum, breakout, kalman_cont
       │    ├── Short: breakdown, rally_fail, bear_momentum
       │    └── Score filtering / gating
       │
       ├── custom_exit()
       │    ├── tp_hit       (≥7.1%)
       │    ├── bleed_exit   (<-3.3% + 289min)
       │    └── max_hold     (>303min)
       │
       ├── custom_stoploss()
       │    ├── Trail: lock profits at 7.8% threshold
       │    └── Default: -12% hard stop
       │
       ├── custom_stake_amount() — 10% Kelly
       ├── confirm_trade_entry() — Loss breaker + cooldown + trade_intel
       └── confirm_trade_exit()  — Risk recording + ML baseline
```

## Data Flow

```
Exchange
  │
  ▼
DataValidator ──── NaN/staleness/gap/volume checks
  │
  ▼
populate_indicators()
  ├── TA-Lib: RSI(5-10,14), EMA(5-22,50,200), ADX(14),
  │           MACD(8,17,5), BB(20,2), OBV, VWAP, ATR(14)
  ├── KalmanFilter: kf_price, kf_trend, kf_prediction, kf_confidence,
  │                 kf_direction, kf_innovation, kf_trend_acceleration
  ├── HMM: hmm_regime, hmm_p_bull/range/bear, hmm_regime_stability,
  │        hmm_transition_risk, hmm_vol_regime, hmm_trend_strength
  ├── _calculate_entry_score (V2.1): 10-component composite score
  └── _compute_hmm_target/sl: regime-adaptive TP/SL
  │
  ▼
RegimeEngine ──── 7-state regime (strong/weak bull/bear, low_vol, neutral)
  │
  ▼
populate_entry_trend ──── 7 entry patterns, score-filtered
  │
  ▼
bot_loop_start()
  ├── Regime-aware max_open_trades
  ├── Drift-aware threshold adjustment
  ├── Daily summary + hourly health → Telegram
  └── Risk governor update
  │
  ▼
create_trade / exit_positions
  ├── get_entry_signal() — checks latest candle only
  ├── custom_exit() — TP, bleed, max_hold
  └── custom_stoploss() — trailing lock
```

## Data Storage

| Path | Content |
|------|---------|
| `user_data/data/binance/futures/*.feather` | Cached OHLCV (5m, 1h) for all pairs |
| `user_data/tradesv3_v2_1.sqlite` | V2.1 trade DB |
| `user_data/tradesv3_v3_1.sqlite` | V3.1 trade DB |
| `user_data/tradesv3.sqlite` | V5-BTC trade DB |
| `user_data/logs/freqtrade_v2_1.log` | V2.1 logs |
| `user_data/logs/freqtrade_v3_1.log` | V3.1 logs |
| `user_data/logs/freqtrade.log` | V5-BTC logs |
| `user_data/backtest_results/` | Backtest output (JSON + ZIP) |
| `user_data/hyperopt_results/` | Hyperopt output (.fthypt) |

## ML Models Directory

| Path | Purpose |
|------|---------|
| `ml/hmm_regime.py` | HMM for V2.1 (3-state, EM-trained) |
| `ml/hmm_regime_v3.py` | HMM for V3.1 (optimized variant) |
| `ml/kalman_filter.py` | 1D Kalman with adaptive noise |
| `ml/feature_engine.py` | Training data generation for LightGBM |
| `ml/monte_carlo.py` | Monte Carlo simulation for validation |
| `ml/models/winrate_model.pkl` | Trained winrate prediction model |
| `core/` | Risk, monitoring, drift, trade intel, deployment |
| `tests/` | Unit tests for all core + ML modules |

## Networking

| Container | Host Port | Container Port | Purpose |
|-----------|-----------|----------------|---------|
| V2.1 | 127.0.0.1:8082 | 8082 | REST API |
| V3.1 | 127.0.0.1:8083 | 8082 | REST API |
| V5-BTC | 127.0.0.1:8085 | 8082 | REST API |

All containers share the Docker volume `phoenix-scalper_scalper_data` mapped to `/freqtrade/user_data` and the bind mount `./strategies/` → `/freqtrade/strategies/`.
