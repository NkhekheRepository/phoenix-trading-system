# QIP — PhoenixScalper

ML-enhanced Binance Futures scalping bot built on [Freqtrade](https://www.freqtrade.io).

- Long/Short, 5m timeframe, 22 altcoin pairs
- 50-100x leverage, ATR-dynamic stoploss, HMM regime-adjusted targets
- Kalman filter trend detection + 3-state HMM regime classifier

---

## Quick Start (AWS EC2)

### 1. Launch EC2

| Setting | Value |
|---------|-------|
| AMI | Ubuntu 22.04 LTS |
| Instance | **t3.medium** (2 vCPU, 4 GB RAM) or larger |
| Storage | 20 GB gp3 |
| Security Group | Allow SSH (22) from your IP |

### 2. Install Docker

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu
# Log out & back in (or: newgrp docker)
```

### 3. Clone & Configure

```bash
git clone https://github.com/<YOUR_USERNAME>/QIP
cd QIP

cp config.json.example config.json   # nano config.json     → paste your API keys
cp .env.example .env                 # nano .env            → paste Telegram token
```

### 4. Start

```bash
docker compose build
docker compose up -d
```

### 5. Monitor

```bash
docker compose logs -f --tail 50
```

---

## Configuration

### `.env` — Secrets & API Keys

| Variable | Required | Description |
|----------|----------|-------------|
| `EXCHANGE_API_KEY` | Yes* | Binance API key |
| `EXCHANGE_API_SECRET` | Yes* | Binance API secret |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | No | Telegram chat/user ID |

\* Leave empty for dry-run mode (simulated trading with $100).

### `config.json` — Bot Settings

Key settings you may want to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `dry_run` | `true` | `true` = simulated, `false` = real money |
| `dry_run_wallet` | `100` | Starting wallet in dry-run mode |
| `max_open_trades` | `10` | Maximum concurrent positions |

---

## Operations

```bash
# View logs
docker compose logs -f

# Stop
docker compose down

# Update after code changes
docker compose build && docker compose up -d

# Download fresh market data
docker compose exec phoenix-scalper-bot freqtrade download-data \
  --exchange binance --trading-mode futures --timeframes 5m --days 60

# Backtest
docker compose exec phoenix-scalper-bot freqtrade backtesting \
  --strategy PhoenixScalper --timerange 20260601-20260706 --timeframe 5m

# Hyperopt
docker compose run --rm phoenix-scalper-bot freqtrade hyperopt \
  --strategy PhoenixScalper --hyperopt-loss SharpeHyperOptLossDaily \
  --timerange 20260601-20260706 --spaces buy sell --epochs 200
```

---

## Architecture

```
QIP/
├── strategies/PhoenixScalper.py   # Main trading strategy
├── ml/
│   ├── hmm_regime.py              # 3-state HMM (Bull/Range/Bear)
│   ├── kalman_filter.py           # 1D Kalman filter (level + trend)
│   ├── feature_engine.py          # Feature generation
│   └── monte_carlo.py             # Strategy validation
├── scripts/                       # Health checks, analysis tools
├── Dockerfile                     # Container build
├── docker-compose.yml             # Orchestration
├── config.json.example            # Config template
└── .env.example                   # Secrets template
```
