# QIP — PhoenixScalper Hedge Fund System

ML-enhanced Binance Futures scalping system. 3 parallel trading bots with Kalman Filter trend detection, 3-state HMM regime classification, and hyperopt-tuned parameters targeting PF > 1.5 and Sharpe > 2.0.

---

## System Architecture

```
phoenix-scalper/
├── strategies/                    # Freqtrade strategy modules
│   ├── PhoenixScalperV2.1.py      # Main 8-pair bot (GodModeQuantBot)
│   ├── PhoenixScalperV3.1.py      # Secondary 8-pair bot (LettaAIbot)
│   ├── PhoenixScalperV5_BTC.py    # BTC-only bot (PhoenixEdgeBot)
│   ├── PhoenixScalperV2_1_hyperopt.py  # Hyperopt strategy (10 exit params)
│   ├── bt_config.json             # Backtesting config (isolated futures)
│   ├── backups/                   # Pre-tuning backups
│   └── _legacy/                   # Archived strategy versions
├── ml/                            # Machine learning modules
│   ├── kalman_filter.py           # 1D Kalman filter (level + trend + adaptive noise)
│   ├── hmm_regime.py              # 3-state HMM (Bull/Range/Bear) with EM training
│   ├── hmm_regime_v3.py           # V3 variant with StandardScaler
│   ├── feature_engine.py          # Training data generation
│   ├── monte_carlo.py             # Monte Carlo bootstrap validation
│   ├── train_strategy_model.py    # Supervised win-rate model
│   └── train_winrate_model.py     # ML-based exit prediction
├── core/                          # Central framework
│   ├── data_quality.py            # Candle validation + outlier detection
│   ├── regime_engine.py           # Multi-timeframe regime analysis
│   ├── strategy_allocator.py      # Dynamic capital allocation
│   ├── risk_governor.py           # Independent risk layer
│   ├── market_memory.py           # Long-term market knowledge
│   ├── concept_drift.py           # Statistical drift detection
│   ├── trade_intel.py             # Trade attribution analysis
│   ├── experiment_db.py           # Experiment tracking
│   ├── deployment.py              # Safe deployment + rollback
│   ├── champion_challenger.py     # Strategy competition framework
│   ├── ml_engine.py               # ML model orchestration
│   ├── monitoring.py              # System health monitoring
│   ├── ev_core.py                 # Expected value computations
│   ├── telegram_ev.py             # Telegram event dispatcher
│   └── validation_pipeline.py     # End-to-end validation
├── tests/                         # Test suite
│   ├── core/                      # Core module tests (9 test files)
│   └── ml/                        # ML module tests (3 test files)
├── research/                      # Quant research tools
│   ├── quant_agent.py             # Hypothesis generation agent
│   ├── walk_forward_validation.py # Walk-forward analysis
│   ├── feature_importance.py      # Feature importance analysis
│   └── run_experiment.py          # Batch experiment runner
├── tools/                         # Utility tools
│   ├── gated_backtest.py          # Sequential backtesting gate
│   ├── ev_core.py                 # Expected value calculator
│   └── observer.py                # Live trade observer
├── scripts/                       # Automation scripts
│   └── optuna_mc_optimizer.py     # Optuna + Monte Carlo optimizer
├── docs/
│   └── architecture.md            # Full architecture documentation
├── config-v2.1.json               # V2.1 bot config (8 pairs)
├── config-v3.1.json               # V3.1 bot config (8 pairs)
├── config-v5-btc.json             # V5_BTC bot config (BTC only)
├── Dockerfile                     # Container build
├── docker-compose.yml             # Base orchestration
├── docker-compose-v2.1.yml        # V2.1 standalone compose
├── docker-compose-v3.1.yml        # V3.1 standalone compose
├── docker-compose-v5-btc.yml      # V5_BTC standalone compose
└── .env.example                   # Secrets template
```

---

## 3-Bot Hedge Fund

| Bot | Telegram Handle | Strategy | Pairs | Port | Config |
|-----|----------------|----------|-------|------|--------|
| V2.1 | @GodModeQuantBot | PhoenixScalperV2.1 | 8 (BTC, ETH, SOL, XRP, DOGE, BNB, ADA, LINK) | 8082 | config-v2.1.json |
| V3.1 | @LettaAIbot | PhoenixScalperV3.1 | 8 (same universe) | 8083 | config-v3.1.json |
| V5_BTC | @PhoenixEdgeBot | PhoenixScalperV5_BTC | 1 (BTC only) | 8085 | config-v5-btc.json |

Each bot runs as an independent Docker container with its own SQLite database and isolated margin mode. Shared strategy bind-mount allows hot-reloading.

---

## ML Pipeline

### Kalman Filter (`ml/kalman_filter.py`)
- 1D state-space model: `[price_level, price_trend]`
- Adaptive measurement noise from ATR + volume ratio
- Outputs: filtered price, trend direction/acceleration, prediction confidence, innovation

### HMM Regime Detector (`ml/hmm_regime.py`)
- 3-state Hidden Markov Model: Bull (mu>0, low sigma), Range (mu=0), Bear (mu<0, high sigma)
- EM training with Baum-Welch (5 iterations, 1000-point subsample)
- Feature scaling prevents volume_change from dominating log_returns
- Outputs: regime probabilities, stability, transition risk, vol regime

### Entry Score (10-feature composite)
- HMM bull/bear probability (20pts)
- ADX trend strength (15pts)
- Kalman filter confidence (10pts)
- DI directional differential (10pts)
- Kalman momentum + acceleration (10pts)
- Volume ratio (10pts)
- Regime stability (10pts)
- RSI positioning (5pts)
- EMA pullback distance (5pts)
- Multi-EMA trend alignment (5pts)
- Max score: 58 (capped)

---

## Hyperopt Results

### Best Parameters (SortinoHyperOptLossDaily, 500 epochs)

**Entry:**
| Param | Value | Description |
|-------|-------|-------------|
| score_threshold | 51 | Minimum composite score to enter |
| adx_threshold | 30 | Minimum ADX for trend strength |
| ema_fast/slow | 7/14 | Fast/slow EMA periods |
| volume_factor | 1.0 | Minimum volume ratio |
| rsi_period | 6 | RSI calculation period |

**Exit:**
| Param | Value | Description |
|-------|-------|-------------|
| tp_target | 0.071 | Take profit at 7.1% equity (0.71% price at 10x) |
| trail_threshold | 0.078 | Trailing stop activates at 7.8% profit |
| lock_ratio | 0.359 | Trail locks 36% of profit above threshold |
| bleed_loss | 0.033 | Bleed exit at -3.3% equity |
| max_hold_min | 303 | Force exit after 5 hours |

**In-sample (2026-07-15 to 2026-07-17, 8 pairs):**
- 10 trades | 6W / 4L | 60% win rate
- Total profit: +1.05% | Profit Factor: 1.48
- Sharpe: 15.43 | Sortino: 23.26
- Max drawdown: 1.89%

**Out-of-sample (2026-07-18 to 2026-07-20):**
- 8 trades | 4W / 4L | 50% win rate
- Total profit: -0.06% (breakeven)
- Max drawdown: 1.10%

---

## Quick Start (AWS EC2 / bare metal)

### Prerequisites
- Ubuntu 22.04+ with Docker and Docker Compose
- 3.2 GB+ RAM (minimum for 3 containers)
- Binance API keys (paper or real)

### Setup

```bash
# Clone
git clone https://github.com/NkhekheRepository/QIP
cd phoenix-scalper

# Configure
cp .env.example .env          # Edit with Telegram + Exchange keys
cp config.json.example config.json

# Build
docker compose build

# Start all 3 bots
docker compose -f docker-compose-v2.1.yml up -d
docker compose -f docker-compose-v3.1.yml up -d
docker compose -f docker-compose-v5-btc.yml up -d
```

### Commands

```bash
# View logs
docker logs phoenix-scalper-v2.1-bot -f --tail 50

# Backtest
docker exec phoenix-scalper-v2.1-bot freqtrade backtesting \
  --config /freqtrade/strategies/bt_config.json \
  --strategy PhoenixScalperV2_1_Hyperopt \
  --strategy-path /freqtrade/strategies \
  --timerange 20260715-20260717 --timeframe 5m

# Hyperopt
docker exec phoenix-scalper-v2.1-bot freqtrade hyperopt \
  --config /freqtrade/strategies/bt_config.json \
  --strategy PhoenixScalperV2_1_Hyperopt \
  --strategy-path /freqtrade/strategies \
  --hyperopt-loss SortinoHyperOptLossDaily \
  --timerange 20260715-20260717 \
  --spaces buy sell --epochs 500 -j 1

# Download data
docker exec phoenix-scalper-v2.1-bot freqtrade download-data \
  --exchange binance --trading-mode futures --timeframes 5m 15m 1h --days 200
```

---

## Docker Deployment

Each bot runs as a standalone container:

```bash
# V2.1 — 8 pairs, port 8082
docker run -d --name phoenix-scalper-v2.1-bot \
  --restart unless-stopped \
  -v $(pwd)/config-v2.1.json:/freqtrade/config.json:ro \
  -v $(pwd)/strategies:/freqtrade/strategies \
  -v scalper_data:/freqtrade/user_data \
  -p 127.0.0.1:8082:8082 \
  phoenix-scalper-phoenix-scalper:latest \
  trade --config /freqtrade/config.json \
  --strategy PhoenixScalperV2_1 \
  --strategy-path /freqtrade/strategies/ \
  --db-url sqlite:////freqtrade/user_data/tradesv3_v2_1.sqlite \
  --logfile /freqtrade/user_data/logs/freqtrade_v2.1.log
```

Repeat for V3.1 (port 8083) and V5_BTC (port 8085) with respective configs.

---

## Performance Targets

| Metric | Target | Current (IS) | Current (OOS) |
|--------|--------|-------------|---------------|
| Profit Factor | > 1.5 | 1.48 | 1.0 |
| Sharpe Ratio | > 2.0 | 15.43 | — |
| Win Rate | 40-60% | 60% | 50% |
| Max Drawdown | < 5% | 1.89% | 1.10% |
| Avg Trade Duration | — | 2h 46m | 4h 26m |

---

## Risk Controls

- **Hard stop**: -12% price (-120% equity at 10x, can never be hit due to liquidation)
- **Trailing stop**: Activates at +7.8% equity, locks 36% of peak profit
- **Bleed exit**: -3.3% equity after 4.8 hours (prevents death by a thousand cuts)
- **Max hold**: 5 hours forced exit
- **All trades**: Isolated margin, paper mode (`dry_run: true`)
- **Independent DBs**: Each bot has its own SQLite database (no cross-contamination)

---

## Dependencies

| Package | Version |
|---------|---------|
| freqtrade | 2026.5.1 |
| ccxt | 4.5.55 |
| pandas | 3.0.3 |
| numpy | 2.4.5 |
| scipy | 1.17.1 |
| joblib | 1.5.3 |
| lightgbm | 4.6.0 |
| Python | 3.14.5 |

---

## License

MIT — see LICENSE file.
