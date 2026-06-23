# Phoenix Trading System

Two independent Docker sandboxes for automated **Binance Futures** trading, each with ML-enhanced strategies built on [Freqtrade](https://www.freqtrade.io).

---

## 1. Installation

### 1.1 Docker (recommended — 30 seconds)

```bash
git clone --depth 1 https://github.com/NkhekheRepository/phoenix-trading-system
cd phoenix-trading-system

# Pick a sandbox:
cp phoenix30/.env.example phoenix30/.env           # nano phoenix30/.env
cp phoenix30/config.json.example phoenix30/config.json
cd phoenix30 && docker compose up -d

# Or for the scalper:
cp phoenix-scalper/.env.example phoenix-scalper/.env
cp phoenix-scalper/config.json.example phoenix-scalper/config.json
cd phoenix-scalper && docker compose up -d
```

### 1.2 Bare Metal

```bash
# System: Python 3.10+, TA-Lib
python3 -m venv venv && source venv/bin/activate
pip install freqtrade lightgbm scikit-learn joblib python-telegram-bot
freqtrade trade --config phoenix30/config.json --strategy Phoenix30
```

### 1.3 System Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 1 core | 2+ cores |
| RAM | 1 GB | 2 GB |
| Disk | 5 GB | 20 GB (data) |
| Docker | 24+ | latest |
| Python | 3.10+ | 3.10-3.12 |

---

## 2. Configuration

### 2.1 Environment Variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Telegram bot token for trade alerts |
| `TELEGRAM_CHAT_ID` | Yes | — | Telegram chat/user ID |
| `EXCHANGE_API_KEY` | No | (empty) | Binance API key (empty = dry-run) |
| `EXCHANGE_API_SECRET` | No | (empty) | Binance API secret |
| `EXCHANGE_PASSWORD` | No | (empty) | Binance API password |
| `API_USERNAME` | No | `freqtrader` | REST API username |
| `API_PASSWORD` | No | `freqtrader` | REST API password |
| `JWT_SECRET` | No | `change-me` | JWT signing secret |

### 2.2 Config JSON Reference

**Full schema:**

| Key | Type | phoenix30 | phoenix-scalper |
|---|---|---|---|
| `trading_mode` | string | `"futures"` | `"futures"` |
| `margin_mode` | string | `"isolated"` | `"isolated"` |
| `stake_currency` | string | `"USDT"` | `"USDT"` |
| `stake_amount` | number/string | `30` | `"unlimited"` |
| `max_open_trades` | number | `5` | `10` |
| `dry_run` | bool | `true` | `true` |
| `dry_run_wallet` | number | `1000` | `100` |
| `liquidation_buffer` | number | `0.0` | `0.15` |
| `entry_pricing.price_side` | string | `"same"` | `"same"` |
| `entry_pricing.order_book_top` | number | `1` | `1` |
| `exit_pricing.price_side` | string | `"same"` | `"same"` |
| `exit_pricing.order_book_top` | number | `1` | `1` |
| `telegram.enabled` | bool | `false` | `false` |
| `api_server.listen_port` | number | `8080` | `8082` |

### 2.3 Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) — get your token
2. Get your chat ID: message [@userinfobot](https://t.me/userinfobot)
3. Set in `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   TELEGRAM_CHAT_ID=123456789
   ```
4. Set `"telegram.enabled": true` in `config.json`

The bot sends trade open/close notifications, daily summaries, and health alerts.

### 2.4 REST API Setup

Each sandbox exposes a REST API:

```bash
# Phoenix30
curl -u freqtrader:freqtrader http://localhost:8080/api/v1/status

# PhoenixScalper
curl -u freqtrader:freqtrader http://localhost:8082/api/v1/status
```

Change credentials via `api_server.username` / `api_server.password` in config.

---

## 3. Sandboxes

### 3.1 Phoenix30

| Property | Value |
|---|---|
| Type | Long-only trend following |
| Timeframe | 5m |
| Leverage | 30x fixed |
| Max trades | 5 |
| Stake | 30 USDT per trade |
| Pairs | BTC, BNB, XRP, DOGE, NEAR |
| Entry signals | Trend pullback, RSI bounce, MACD reversal, Kalman breakout |
| ML filter | LightGBM winrate model (94 features) |
| Regime filter | Rejects entries when BTC daily < EMA50 |
| Exit | Regime-adaptive (HMM + Kalman) |
| Port | 8080 |
| Container | `phoenix30-bot` |

**Located at:** `phoenix30/`

### 3.2 PhoenixScalper

| Property | Value |
|---|---|
| Type | Long/Short scalper |
| Timeframe | 5m |
| Leverage | 100x long / 50x short (adaptive) |
| Max trades | 10 |
| Stake | Unlimited (Kelly 17.7% per trade) |
| Pairs | 40 altcoins |
| Entry signals | 4 long + 3 short (HMM-filtered) |
| SL | ATR-dynamic (0.15%-0.6%) |
| TP | HMM regime-adjusted targets (0.35%-1.2%) |
| Max hold | 1 hour |
| Port | 8082 |
| Container | `phoenix-scalper-bot` |

**Located at:** `phoenix-scalper/`

---

## 4. Strategy Customization

### 4.1 Phoenix30 Entry/Exit Logic

**Entry signals (any may fire per candle):**

| Signal | Conditions |
|---|---|
| `trend_pullback` | Bull trend, price pulls to EMA, RSI 39-55, ADX > 20, high volume, DI+ > DI- |
| `rsi_bounce` | RSI was oversold, now recovering, above EMA200, above BB lower |
| `macd_reversal` | MACD hist crosses zero, bull trend, RSI 40-60 |
| `kalman_breakout` | Price > Kalman prediction, trend accelerating, near BB upper, high volume |

**ML gate:** Entry proceeds only if LightGBM win probability >= `ml_threshold` (0.30).

**Market gate:** Entry blocked if BTC daily close < EMA50.

**Exit:** Delegated to `RegimeAdaptiveExit` — regime-specific: Bear = immediate, Range = TP at 0.5%, Bull = Kalman reversal detection.

**Parameters available for hyperopt:**
- `ema_fast`, `ema_slow`, `rsi_period`, `rsi_pullback_low`, `rsi_pullback_high`
- `rsi_bounce`, `rsi_exit`, `adx_threshold`, `volume_factor`, `ml_threshold`

### 4.2 PhoenixScalper Entry/Exit Logic

**Long signals:**

| Signal | Conditions |
|---|---|
| `hmm_pullback` | Price touches EMA, RSI 35-55, ADX > 20, DI+ > DI-, HMM stable |
| `rsi_momentum` | RSI > 50, above EMA fast, strong volume |
| `momentum_breakout` | 5-period high breakout, 1.5x volume |
| `kalman_cont` | Kalman trend > 0, confidence > 0.6, above prediction |

**Short signals:**

| Signal | Conditions |
|---|---|
| `short_breakdown` | Breakdown below support, massive volume, below VWAP/BB |
| `short_rally_fail` | Rally to EMA, rejected, DI- > DI+ |
| `short_bear_momentum` | Bearish candle, below BB lower, elevated volume |

**HMM gate:** Long only if `hmm_p_bull > hmm_p_bear`. Short only if `hmm_p_bear > hmm_p_bull`.

**Kelly sizing:** 17.7% of wallet per trade.

**Exit:** Profit >= 10% or hold > 1 hour. Dynamic ATR stoploss (0.24%-0.4%).

### 4.3 Customizing Parameters

Edit `buy_params` / `sell_params` dict in the strategy file, or run hyperopt to find optimal values.

```python
# phoenix30/strategies/Phoenix30.py
buy_params = {
    "adx_threshold": 20,     # lower = more signals, lower quality
    "ema_fast": 13,           # faster = more responsive
    "ema_slow": 25,           # slower = smoother trend
    "volume_factor": 1.086,   # higher = stricter volume confirmation
    "ml_threshold": 0.30,     # lower = ML allows more entries
}
```

---

## 5. ML Infrastructure

```
ml/
├── hmm_regime.py          # 3-state Gaussian HMM (Bull/Range/Bear)
├── kalman_filter.py       # 1D Kalman filter (level + trend)
├── regime_adaptive.py     # Regime-specific exit decisions
├── monte_carlo.py         # Strategy validation via permutation test
├── feature_engine.py      # Generate training data from 5m candles
├── train_30x_model.py     # LightGBM winrate classifier trainer
└── models/
    └── winrate_model.pkl  # Pre-trained LightGBM (12 KB)
```

### 5.1 Kalman Filter

**File:** `ml/kalman_filter.py`

1D Kalman filter with state `[price_level, price_trend]`.

- **Prediction:** `x_pred = F @ x` where `F = [[1,1],[0,1]]`
- **Adaptive noise:** `R = (ATR%)^2 * vol_adj` — trusts observations more when volume is high
- **13 output features:** filtered price, trend, prediction, confidence (0-1), direction (+1/0/-1), innovation, covariance, momentum, acceleration, prediction error, regime score, vol of trend, ATR ratio

**Usage:** Both strategies call `compute_kalman_features()` in `populate_indicators()`. Cached per pair, recomputed every 12th cycle.

### 5.2 HMM Regime Detector

**File:** `ml/hmm_regime.py`

3-state Gaussian Hidden Markov Model.

| State | Return | Volatility | Volume |
|---|---|---|---|
| Bull (0) | Positive | Low | High |
| Range (1) | ~Zero | Medium | Medium |
| Bear (2) | Negative | High | High |

- **Training:** Baum-Welch EM (5 iterations), subsampled to 1000 points max
- **Prediction:** Emission-only fast path (no forward-backward)
- **8 output features:** regime class (0/1/2), P(bull), P(range), P(bear), stability (1 - entropy), transition risk, vol regime, trend strength

**Usage:** Both strategies call `compute_hmm_features()` in `populate_indicators()`. Cached per pair.

### 5.3 LightGBM Winrate Model

**File:** `ml/train_30x_model.py`

**Purpose:** Binary classifier predicting trade win probability from candle features.

**Architecture:** LGBMClassifier, 500 estimators, max_depth=6, L1/L2 regularization, class-balanced.

**Input:** 94 features — all TA-Lib indicators + Kalman outputs + HMM outputs + time features.

**Output:** Win probability [0, 1]. Entries filtered at threshold >= 0.30.

**Pre-trained model:** `phoenix30/ml/models/winrate_model.pkl` (12 KB, ships with repo).

### 5.4 Regime-Adaptive Exit

**File:** `ml/regime_adaptive.py`

Used by Phoenix30 via `RegimeAdaptiveExit`:

| Regime | Action |
|---|---|
| Bear | Immediate exit |
| Range | Take profit at 0.5%, time exit at 60 min |
| Bull | Stall exit at 120 min, Kalman reversal detection |
| Any (high conf reversal) | Exit if Kalman turns bearish with high confidence |

### 5.5 Monte Carlo Validation

**File:** `ml/monte_carlo.py`

```bash
python phoenix30/ml/monte_carlo.py
```

Runs 10,000 random permutations of trade history to validate strategy robustness.

**Target gates:** Sharpe p10 > 2.0, Calmar p50 > 2.0, MaxDD p95 < 15%, Ruin < 10%.

### 5.6 Training Your Own Model

```bash
# 1. Generate training data
python phoenix30/ml/feature_engine.py

# 2. Train LightGBM
python phoenix30/ml/train_30x_model.py

# 3. Model saved to phoenix30/ml/models/winrate_model.pkl
#    (restart bot to pick it up)
```

Requires `training_data.pkl` (generated from downloaded OHLCV data).

---

## 6. Backtesting

```bash
cd phoenix30

docker compose run --rm phoenix30-bot freqtrade backtesting \
  --strategy Phoenix30 \
  --timerange 20260401-20260630 \
  --timeframe 5m

# Or bare-metal:
freqtrade backtesting \
  --config config.json \
  --strategy Phoenix30 \
  --timerange 20260401-20260630 \
  --timeframe 5m
```

Results saved to `data/backtest_results/`.

---

## 7. Hyperopt

Parameter optimization for any strategy:

```bash
cd phoenix-scalper

docker compose run --rm phoenix-scalper-bot freqtrade hyperopt \
  --strategy PhoenixScalper \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --timerange 20260401-20260630 \
  --spaces buy sell \
  --epochs 200

# With custom loss:
# --hyperopt-loss SortinoHyperOptLossDaily
# --hyperopt-loss CalmarHyperOptLoss
```

Optimized parameters are written to the strategy JSON file.

---

## 8. Data Download

```bash
cd phoenix30

# Download 60 days of 5m data for all whitelisted pairs
docker compose exec phoenix30-bot freqtrade download-data \
  --exchange binance \
  --trading-mode futures \
  --timeframes 5m 1h 4h \
  --days 60

# Or bare-metal:
freqtrade download-data \
  --exchange binance \
  --trading-mode futures \
  --timeframes 5m 1h 4h \
  --pairs BTC/USDT:USDT BNB/USDT:USDT
```

Data is stored in the Docker volume at `/freqtrade/user_data/data/`.

---

## 9. Production Operations

### 9.1 Docker Management

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Restart
docker compose restart

# Logs
docker compose logs -f              # all logs
docker compose logs -f --tail 100   # last 100 lines

# Rebuild after strategy changes
docker compose build --no-cache && docker compose up -d

# Execute commands inside container
docker compose exec phoenix30-bot freqtrade backtesting --strategy Phoenix30
```

### 9.2 systemd Integration

```bash
# Install service
sudo cp systemd/phoenix30.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable phoenix30.service
sudo systemctl start phoenix30.service

# Monitor
sudo systemctl status phoenix30.service
sudo journalctl -u phoenix30.service -f
```

Paths assume sandbox is at `/opt/phoenix30/`. Adjust `WorkingDirectory` in the `.service` file if different.

### 9.3 Cron Jobs

```bash
# Install via crontab -e:

# Phoenix30 health check (every 5 min)
*/5 * * * * cd /opt/phoenix30 && docker compose exec -T phoenix30-bot \
  python /freqtrade/scripts/health_check.py >> /var/log/phoenix30_health.log 2>&1

# Phoenix30 regime alert (hourly)
0 * * * * cd /opt/phoenix30 && docker compose exec -T phoenix30-bot \
  python /freqtrade/scripts/regime_alert.py >> /var/log/phoenix30_regime.log 2>&1

# Same for phoenix-scalper:
# */5 * * * * cd /opt/phoenix-scalper && docker compose exec -T phoenix-scalper-bot ...
```

**health_check.py:** Checks if the container is running, sends Telegram alert on failure/recovery.

**regime_alert.py:** Fetches BTC daily data, computes market regime (BULL/BEAR/RANGE), sends Telegram on regime change.

### 9.4 Logging

Logs go to the Docker volume at `/freqtrade/user_data/logs/`:

```bash
# View inside container
docker compose exec phoenix30-bot tail -f /freqtrade/user_data/logs/freqtrade.log

# Or access via the named volume (host path varies by Docker install)
```

Log rotation: freqtrade rotates at 10 MB per file, keeps 7 archives.

---

## 10. Troubleshooting

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 10.1 | Port 8080/8082 in use | Another process binds the port | Change `listen_port` in `config.json` or stop the other process |
| 10.2 | "No data for pair" | Historical OHLCV not downloaded | Run `freqtrade download-data` for your pairs and timeframes |
| 10.3 | ML model not loaded | Missing `winrate_model.pkl` | File should be at `phoenix30/ml/models/winrate_model.pkl`. Re-train if missing |
| 10.4 | Telegram not sending | Token/chat_id not configured | Check `.env` is present and has valid values. Set `telegram.enabled: true` |
| 10.5 | Container exits immediately | Config JSON error | `docker compose logs` to see parse errors |
| 10.6 | "CCXT Request timeout" | Binance rate limiting | Increase `ccxt_config.timeout` in config |
| 10.7 | HMM/Kalman errors | `ml/` package not importable | Ensure `sys.path.insert` in strategy points to parent of `ml/` |
| 10.8 | Low performance on backtest | Wrong timerange or stale data | Download fresh data, use recent timerange |
| 10.9 | "Exchange is not reachable" | No internet / API down | Check connectivity and exchange status |
| 10.10 | AIODNS or socket errors | IPv6 issues with Binance | `entrypoint.sh` sets `AIODNS_RESOLVE_USE_SOCKET=1` — confirm it runs |
