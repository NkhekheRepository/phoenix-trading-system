# Phoenix Trading System

**ML-enhanced automated trading platform** for Binance Futures. Three independent bots running the PhoenixScalper strategy family with HMM regime detection, Kalman filter price smoothing, and adaptive risk management.

## System Overview

| Bot | Strategy | Pairs | Leverage | Role |
|-----|----------|-------|----------|------|
| **V2.1** | PhoenixScalperV2_1 | 8 diversified | 10× | Score-based scalper |
| **V3.1** | PhoenixScalperV3_1 | 8 diversified | 10× | Adaptive scalper |
| **V5-BTC** | PhoenixScalperV5_BTC | BTC-only | 10× | Volatility scalper |

**Architecture**: `docs/ARCHITECTURE.md`

## Quick Start (60 seconds)

```bash
git clone --depth 1 https://github.com/NkhekheRepository/phoenix-trading-system.git
cd phoenix-trading-system
cp phoenix-scalper/.env.example phoenix-scalper/.env
nano phoenix-scalper/.env        # Add Telegram + Exchange API keys
cd phoenix-scalper
docker compose -f docker-compose-v2.1.yml up -d --build
docker compose -f docker-compose-v3.1.yml up -d --build
docker compose -f docker-compose-v5-btc.yml up -d --build
```

**Full installation**: `docs/INSTALLATION.md`

## Documentation

| Document | Description |
|----------|-------------|
| `docs/QUICKSTART.md` | 60-second deploy guide |
| `docs/INSTALLATION.md` | Fresh server installation manual |
| `docs/ARCHITECTURE.md` | System architecture and data flow |
| `docs/STRATEGY_V21.md` | V2.1 strategy whitepaper |
| `docs/STRATEGY_V31.md` | V3.1 strategy whitepaper |
| `docs/ML_MODULES.md` | HMM, Kalman Filter, ML Engine |
| `docs/CORE_MODULES.md` | Risk, Data Quality, Monitoring |
| `docs/PERFORMANCE.md` | Live performance analysis |
| `docs/CRITICAL_FIXES.md` | Forensic analysis of critical bugs |
| `docs/TROUBLESHOOTING.md` | Common issues and solutions |
| `docs/MONITORING.md` | Alerting, health checks, cron jobs |
| `docs/TESTING.md` | Test suite and validation |
| `docs/RESEARCH.md` | Research findings and experiments |
| `docs/API.md` | REST API and Telegram commands |

## Performance Snapshot

### V3.1 (11 closed trades)

| Metric | Value |
|--------|-------|
| Win Rate | 63.6% |
| Profit Factor | 1.66 |
| Avg ROI | +1.58% |
| Best Trade | +7.57% (ADA) |
| Worst Trade | -9.70% (XRP trailing_stop) |

**Details**: `docs/PERFORMANCE.md`

## Tech Stack

| Component | Technology |
|-----------|------------|
| Trading Engine | Freqtrade 2026.5.1 |
| Exchange | Binance Futures (isolated) |
| Container | Docker + Docker Compose |
| ML Models | LightGBM, HMM (custom EM), Kalman Filter |
| Monitoring | Telegram Bot API |
| Data Store | SQLite + Feather files |
| Language | Python 3.14 |

## License

Private — NkhekheRepository
