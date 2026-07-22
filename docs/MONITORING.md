# PHOENIX TRADING SYSTEM — Monitoring & Alerting

## 1. Architecture

```
┌──────────┐     ┌──────────────┐     ┌────────────┐
│  Bot     │────▶│  Telegram    │────▶│ You        │
│  (docker)│     │  Bot API     │     │ (chat_id)  │
└──────────┘     └──────────────┘     └────────────┘
     │
     │  ┌──────────────┐     ┌────────────┐
     ├──│ health_check │────▶│ Systemd    │
     │  │ (5min cron)  │     │ / Docker   │
     │  └──────────────┘     └────────────┘
     │
     │  ┌──────────────┐     ┌────────────┐
     └──│ regime_alert │────▶│ Telegram   │
        │ (hourly)     │     │ (regime)   │
        └──────────────┘     └────────────┘
```

## 2. Telegram Notifications

### 2.1 Strategy-Level (from within bot)

Sent by `Monitor` class in each bot:

| Notification | Cadence | Trigger |
|-------------|---------|---------|
| Daily Summary | Once daily | `bot_loop_start` |
| Hourly Health | Every hour | `bot_loop_start` |
| Trade Entry | Per trade | `confirm_trade_entry` |
| Trade Exit | Per trade | `confirm_trade_exit` |
| Risk Change | On change | `_evaluate_risk` |
| Data Quality Issue | On detection | `DataValidator` |

### 2.2 Daily Summary Format

```
━━━ DAILY SUMMARY ━━━
Date: 2026-07-22
Regime: weak_bear
Risk: NORMAL
Exposure: 0.0%
Leverage: 10x
Active: 2 trades
  - DOGE/USDT @ 0.07 | +3.2%
  - SOL/USDT @ 77.48 | -0.8%
Total Closed: 11 | +$1.30 | 7W / 4L
━━━━━━━━━━━━━━━━━━━
```

### 2.3 Hourly Health Format

```
━━━ HEALTH CHECK ━━━
Uptime: 2h 34m
Trades: 13 total / 2 active
Memory: 245 MB
Exchange: OK
P&L: +$1.30
Drift: normal
━━━━━━━━━━━━━━━━━
```

## 3. Host-Level Scripts

### 3.1 health_check.py

**Purpose**: Verifies bot container is running. Sends Telegram alert when DOWN and recovery notification when back up.

**Cron**: `*/5 * * * *`

**Behavior**:
1. Run `systemctl --user is-active freqtrade-bot.service` or check docker container
2. If DOWN → immediately send Telegram alert
3. If UP and previous state was DOWN → send recovery notification
4. Store state in `health_state.json`

### 3.2 regime_alert.py

**Purpose**: BTC daily market regime analysis. Detects regime changes and sends alerts.

**Cron**: `0 * * * *`

**Indicators**:
| Indicator | Period | Purpose |
|-----------|--------|---------|
| EMA50 | 50 days | Medium-term trend |
| EMA20 | 20 days | Short-term trend |
| RSI14 | 14 days | Momentum/overbought-oversold |
| ADX14 | 14 days | Trend strength |
| ATR14 | 14 days | Volatility |

**Regime Classification**:
| Regime | Conditions |
|--------|------------|
| STRONG_BULL | EMA50 < EMA20, ADX > 25, RSI > 60 |
| BULL | EMA50 < EMA20, RSI > 50 |
| BULL_WEAK | EMA50 < EMA20, ADX < 20 |
| STRONG_BEAR | EMA50 > EMA20, ADX > 25, RSI < 40 |
| BEAR | EMA50 > EMA20, RSI < 50 |
| BEAR_WEAK | EMA50 > EMA20, ADX < 20 |
| NEUTRAL | EMA50 ~ EMA20, ADX < 25 |
| EXTREME | RSI > 80 or RSI < 20 |

**State**: Stored in `regime_state.json` with last regime and timestamp.

### 3.3 refresh_data.sh

**Purpose**: Downloads fresh market data to prevent stale-data starvation.

**Cron**: `3 * * * *`

**Action**:
```bash
docker exec phoenix-scalper-v2.1-bot freqtrade download-data \
  --pairs BTC/USDT:USDT ETH/USDT:USDT ... \
  --days 2 --timeframes 5m
```

## 4. Cron Job Reference

| Expression | Script | Function |
|-----------|--------|----------|
| `*/5 * * * *` | `health_check.py` | Bot health monitoring |
| `0 * * * *` | `regime_alert.py` | BTC regime analysis |
| `3 * * * *` | `refresh_data.sh` | Market data refresh |

## 5. REST API Monitoring

Each bot exposes a REST API on its port:

```bash
# Ping
curl -u freqtrader:YOUR_PASS http://127.0.0.1:8082/api/v1/ping

# Status
curl -u freqtrader:YOUR_PASS http://127.0.0.1:8082/api/v1/status

# Balance
curl -u freqtrader:YOUR_PASS http://127.0.0.1:8082/api/v1/balance

# Whitelist
curl -u freqtrader:YOUR_PASS http://127.0.0.1:8082/api/v1/whitelist

# Trades
curl -u freqtrader:YOUR_PASS http://127.0.0.1:8082/api/v1/trades

# Performance
curl -u freqtrader:YOUR_PASS http://127.0.0.1:8082/api/v1/profit
```

## 6. Docker Health

```bash
# Check all containers
for b in phoenix-scalper-v2.1-bot phoenix-scalper-v3.1-bot phoenix-scalper-v5-btc-bot; do
  echo "$b: $(docker inspect $b --format '{{.State.Status}}' 2>&1)"
done

# Check resource usage
docker stats --no-stream $(docker ps -q)

# Check logs for errors
for b in phoenix-scalper-v2.1-bot phoenix-scalper-v3.1-bot phoenix-scalper-v5-btc-bot; do
  echo "=== $b ==="
  logfile=$(docker inspect "$b" --format '{{join .Config.Cmd " "}}' | tr ' ' '\n' | grep '\-\-logfile' -A1 | tail -1)
  docker exec "$b" grep -iE "error|exception|traceback|warning" "$logfile" 2>&1 | tail -5
done
```

## 7. Trade Database Queries

```bash
# V2.1 trades
docker exec phoenix-scalper-v2.1-bot sqlite3 \
  /freqtrade/user_data/tradesv3_v2_1.sqlite \
  "SELECT id, pair, open_date, close_date, close_profit_abs, close_profit FROM trades;"

# V3.1 trades
docker exec phoenix-scalper-v3.1-bot sqlite3 \
  /freqtrade/user_data/tradesv3_v3_1.sqlite \
  "SELECT id, pair, open_date, close_date, close_profit_abs, close_profit FROM trades;"

# V5-BTC trades
docker exec phoenix-scalper-v5-btc-bot sqlite3 \
  /freqtrade/user_data/tradesv3.sqlite \
  "SELECT id, pair, open_date, close_date, close_profit_abs, close_profit FROM trades;"

# Performance summary
docker exec phoenix-scalper-v3.1-bot sqlite3 \
  /freqtrade/user_data/tradesv3_v3_1.sqlite \
  "SELECT COUNT(*) as total, SUM(CASE WHEN is_open=0 AND close_profit_abs>0 THEN 1 ELSE 0 END) as wins, SUM(CASE WHEN is_open=0 AND close_profit_abs<=0 THEN 1 ELSE 0 END) as losses, ROUND(SUM(close_profit_abs),4) as total_pnl, ROUND(AVG(close_profit),4) as avg_roi FROM trades WHERE is_open=0;"
```
