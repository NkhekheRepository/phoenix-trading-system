# Phoenix Trading System

**Two independent Docker sandboxes** — each self-contained, deployable to any server in under 3 minutes.

## Sandboxes

| Sandbox | Strategy | Type | Pairs | Port | Container |
|---|---|---|---|---|---|
| `phoenix30/` | Phoenix30 | Long-only, 30x, ML-enhanced | 5 | 8080 | phoenix30-bot |
| `phoenix-scalper/` | PhoenixScalper | Long/Short, 50-100x, HMM-regime | 40 | 8082 | phoenix-scalper-bot |

## Quick Start

```bash
# === Deploy one sandbox to ANY server ===

git clone --depth 1 <repo-url> phoenix
cd phoenix/phoenix30          # or phoenix-scalper

# Initialize (one-time):
cp .env.example .env          # nano .env with Telegram token
cp config.json.example config.json  # edit if needed
docker compose up -d          # LIVE in 30 seconds

# Monitor:
docker compose logs -f
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    NEW SERVER                         │
│                                                       │
│  /opt/phoenix30/         /opt/phoenix-scalper/        │
│  ┌──────────────────┐   ┌──────────────────────┐      │
│  │ docker-compose   │   │ docker-compose       │      │
│  │   → port 8080    │   │   → port 8082        │      │
│  │   → config.json  │   │   → config.json      │      │
│  │   → strategies/  │   │   → strategies/      │      │
│  │   → ml/          │   │   → ml/              │      │
│  │   → scripts/     │   │   → scripts/         │      │
│  │   → .env         │   │   → .env             │      │
│  │                  │   │                      │      │
│  │ Container:       │   │ Container:           │      │
│  │ phoenix30-bot    │   │ phoenix-scalper-bot  │      │
│  └──────────────────┘   └──────────────────────┘      │
│                                                       │
│  Host Cron:                                            │
│  ├── */5 * * * * docker exec phoenix30-bot ...health  │
│  └── 0  * * * * docker exec phoenix-scalper ...regime  │
└──────────────────────────────────────────────────────┘
```

## ML Pipeline

```
5m candle ──► populate_indicators()
                ├── TA-Lib: EMA, RSI, ADX, ATR, MACD, BB, OBV
                ├── Informative pairs: 1h, 4h, 1d BTC
                ├── KalmanFilter: 13 features (cached, 1/12 cycles)
                └── HMM: 8 features (3-state Gaussian, cached 1/12)
                    │
                    ▼
              populate_entry_trend()
                ├── Phoenix30: 4 long signals + LightGBM ML filter
                └── PhoenixScalper: 4 long + 3 short, HMM-gated
                    │
                    ▼
              confirm_trade_entry() → Binance Futures order
                    │
                    ▼
              custom_exit() → RegimeAdaptiveExit / time-based / ATR-SL
```

## File Structure

```
phoenix-trading-system/
├── phoenix30/                          ← COPY THIS to any server
│   ├── docker-compose.yml              ← docker compose up -d
│   ├── Dockerfile                      ← freqtrade + lightgbm
│   ├── .env.example                    ← Telegram token
│   ├── config.json.example             ← port 8080, BTC/BNB/XRP/DOGE/NEAR
│   ├── strategies/Phoenix30.py         ← Long-only, ML-enhanced
│   ├── strategies/Phoenix30.json       ← Hyperopt params
│   ├── ml/                             ← Kalman, HMM, RegimeAdaptive
│   │   └── models/winrate_model.pkl    ← Pre-trained LightGBM (12KB)
│   └── scripts/{entrypoint,health_check,regime_alert}
│
├── phoenix-scalper/                    ← COPY THIS to any server
│   ├── docker-compose.yml              ← docker compose up -d
│   ├── Dockerfile                      ← freqtrade (no lightgbm)
│   ├── .env.example                    ← Telegram token
│   ├── config.json.example             ← port 8082, 40 pairs
│   ├── strategies/PhoenixScalper.py    ← Long/Short, HMM-scalper
│   ├── ml/                             ← Kalman, HMM, MonteCarlo, FeatureEngine
│   └── scripts/{entrypoint,health_check,regime_alert}
│
├── systemd/                            ← systemctl enable phoenix30
│   ├── phoenix30.service
│   └── phoenix-scalper.service
│
├── cron/crontab.example                ← docker exec cron entries
├── setup.sh                            ← ./setup.sh phoenix30
├── README.md
└── .gitignore
```

## Configuration

Each sandbox has its own `config.json.example` and `.env.example`. Minimum setup:

```bash
cp .env.example .env                # nano .env → add Telegram token
cp config.json.example config.json  # optional: change pairs, wallet
```

### Config keys

| Key | phoenix30 | phoenix-scalper |
|---|---|---|
| stake_amount | 30 USDT | unlimited |
| max_open_trades | 5 | 10 |
| dry_run_wallet | 1000 USDT | 100 USDT |
| listen_port | 8080 | 8082 |
| trading_mode | futures | futures |
| margin_mode | isolated | isolated |
| liquidation_buffer | 0.0 | 0.15 |

## Testing

```bash
# Backtest (from inside sandbox dir)
docker compose run --rm phoenix30-bot freqtrade backtesting \
  --strategy Phoenix30 --timerange 20260401-20260630 --timeframe 5m

# Hyperopt
docker compose run --rm phoenix-scalper-bot freqtrade hyperopt \
  --strategy PhoenixScalper --hyperopt-loss SharpeHyperOptLossDaily \
  --timerange 20260401-20260630 --epochs 200
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Port conflict | Another process on 8080/8082 | Edit `listen_port` in config.json |
| "No data for pair" | Missing historical data | `docker compose exec <container> freqtrade download-data` |
| ML model not loaded | Missing `ml/models/winrate_model.pkl` | This file ships with the repo, check it exists |
| Telegram not sending | Token not set | Check `.env` has valid TELEGRAM_BOT_TOKEN |
| Container exits immediately | Config error | `docker compose logs` to see the error |
