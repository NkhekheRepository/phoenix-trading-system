# PHOENIX TRADING SYSTEM — Installation Manual

## Prerequisites

| Requirement | Version | Check |
|------------|---------|-------|
| Docker Engine | 24+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |
| Git | 2.40+ | `git --version` |
| RAM | 2 GB+ | `free -h` |
| Disk | 20 GB+ | `df -h` |
| OS | Linux (Ubuntu 22.04+ / Debian 12+) | |

## Step 1 — Provision Server

```bash
# Ubuntu 22.04 / 24.04 (DigitalOcean, Vultr, AWS EC2, etc.)
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | bash
```

## Step 2 — Clone Repository

```bash
git clone --depth 1 https://github.com/NkhekheRepository/phoenix-trading-system.git
cd phoenix-trading-system
```

## Step 3 — Configure Secrets

```bash
# Copy environment template
cp phoenix-scalper/.env.example phoenix-scalper/.env

# Edit .env with your credentials
nano phoenix-scalper/.env
```

Required credentials (`.env` file):

```
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyzABCdefGHI
TELEGRAM_CHAT_ID=123456789
EXCHANGE_API_KEY=your_binance_api_key
EXCHANGE_API_SECRET=your_binance_api_secret
EXCHANGE_PASSWORD=
API_USERNAME=freqtrader
API_PASSWORD=freqtrader
JWT_SECRET=generate-a-random-string-here
```

> **Security**: NEVER commit `.env` or config files with real secrets. The repository uses `{{VAR}}` placeholders and the `entrypoint.sh` script substitutes them at container start.

## Step 4 — Build and Start

```bash
# Option A: Start all 3 bots (recommended)
cd phoenix-scalper

docker compose -f docker-compose-v2.1.yml up -d --build
docker compose -f docker-compose-v3.1.yml up -d --build
docker compose -f docker-compose-v5-btc.yml up -d --build

# Option B: Start individual bots
docker compose -f docker-compose-v2.1.yml up -d --build   # V2.1 only
```

## Step 5 — Verify

```bash
# Check container status
docker ps

# Check logs
docker logs phoenix-scalper-v2.1-bot --tail 20

# Check trade database
docker exec phoenix-scalper-v2.1-bot sqlite3 \
  /freqtrade/user_data/tradesv3_v2_1.sqlite \
  "SELECT COUNT(*) FROM trades;"

# Check REST API
curl -u freqtrader:YOUR_PASSWORD http://127.0.0.1:8082/api/v1/ping
```

## Step 6 — Set Up Data Refresh (Cron)

```bash
# Install hourly data refresh cron job
crontab -e
# Add:
3 * * * * /home/your-user/phoenix-trading-system/scripts/refresh_data.sh
```

Or use the systemd timer (see `systemd/` directory).

## Step 7 — Download Initial Data

```bash
# Download 15 days of 5m data for all trading pairs
docker exec phoenix-scalper-v2.1-bot freqtrade download-data \
  --config /freqtrade/config.json \
  --exchange binance \
  --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT XRP/USDT:USDT \
           DOGE/USDT:USDT BNB/USDT:USDT ADA/USDT:USDT LINK/USDT:USDT \
  --days 15 \
  --timeframes 5m
```

## Bot Configuration Reference

### V2.1 — PhoenixScalperV2_1
- **Role**: Primary scalper, 8-pair diversified
- **Whitelist**: BTC, ETH, SOL, XRP, DOGE, BNB, ADA, LINK
- **Leverage**: 10×
- **Stake**: 10% Kelly of wallet
- **Entry**: Score-based (10-component composite), score threshold 55/58 ceiling
- **Exit**: TP 7.1%, bleed -3.3% after 289min, max hold 303min
- **Trailing stop**: Lock profits at 7.8%, lock ratio 0.359
- **Port**: 8082
- **Config**: `phoenix-scalper/config-v2.1.json`
- **Strategy**: `phoenix-scalper/strategies/PhoenixScalperV2.1.py`

### V3.1 — PhoenixScalperV3_1
- **Role**: Adaptive scalper, 8-pair diversified
- **Whitelist**: Same 8 pairs as V2.1
- **Leverage**: 10×
- **Entry**: Same 7 patterns as V2.1, with score gating + ADX/RVOL filter
- **Exit**: Same TP/bleed/max_hold as V2.1
- **Adaptation**: Regime-aware max_open_trades + drift-aware threshold adjustment
- **Port**: 8083
- **Config**: `phoenix-scalper/config-v3.1.json`
- **Strategy**: `phoenix-scalper/strategies/PhoenixScalperV3.1.py`

### V5-BTC — PhoenixScalperV5_BTC
- **Role**: BTC-only volatility scalper
- **Whitelist**: BTC/USDT:USDT only
- **Leverage**: 10×
- **Entry**: Volume spike + RSI extremes + MACD flip + OBV resistance
- **Exit**: Same TP/bleed/max_hold as V2.1
- **Loss controls**: 5 consecutive loss breaker, 10 trades/day limit, 30min cooldown
- **Port**: 8085
- **Config**: `phoenix-scalper/config-v5-btc.json`
- **Strategy**: `phoenix-scalper/strategies/PhoenixScalperV5_BTC.py`

## Updating

```bash
cd phoenix-trading-system
git pull
cd phoenix-scalper
docker compose -f docker-compose-v2.1.yml up -d --build --force-recreate
docker compose -f docker-compose-v3.1.yml up -d --build --force-recreate
docker compose -f docker-compose-v5-btc.yml up -d --build --force-recreate
```

## Uninstalling

```bash
cd phoenix-trading-system/phoenix-scalper
docker compose -f docker-compose-v2.1.yml down
docker compose -f docker-compose-v3.1.yml down
docker compose -f docker-compose-v5-btc.yml down
docker volume rm phoenix-scalper_scalper_data  # Removes ALL trade data
