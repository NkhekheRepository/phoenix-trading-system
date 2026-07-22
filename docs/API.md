# PHOENIX TRADING SYSTEM — API & Command Reference

## 1. REST API

Each bot exposes a REST API on its respective port:

| Bot | Port | URL Base |
|-----|------|----------|
| V2.1 | 8082 | `http://127.0.0.1:8082/api/v1` |
| V3.1 | 8083 | `http://127.0.0.1:8083/api/v1` |
| V5-BTC | 8085 | `http://127.0.0.1:8085/api/v1` |

**Authentication**: HTTP Basic Auth (username/password from config).

### 1.1 System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check → `{"status":"pong"}` |
| `/version` | GET | Freqtrade version |
| `/status` | GET | Bot status (running/stopped) |
| `/show_config` | GET | Current configuration |
| `/logs` | GET | Recent log entries |

### 1.2 Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forcebuy` | POST | Force long entry: `{"pair":"BTC/USDT:USDT"}` |
| `/forceentry` | POST | Force entry with side: `{"pair":"BTC/USDT:USDT","side":"short"}` |
| `/forceexit` | POST | Force exit: `{"tradeid":5}` |
| `/delete` | DELETE | Delete trade: `{"tradeid":5}` |

### 1.3 Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trades` | GET | List all trades |
| `/profit` | GET | P&L summary |
| `/performance` | GET | Performance by pair |
| `/daily` | GET | Daily profit |
| `/balance` | GET | Wallet balance |
| `/count` | GET | Trade counts |
| `/whitelist` | GET | Active whitelist |
| `/blacklist` | GET | Current blacklist |
| `/locks` | GET | Active pair locks |
| `/stats` | GET | Trade statistics |
| `/marketdir` | GET | Market direction |
| `/order` | GET | Order details: `?order_id=xxx` |

### 1.4 Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/start` | POST | Start bot |
| `/stop` | POST | Stop bot |
| `/stopbuy` | POST | Stop new entries |
| `/reload_config` | POST | Reload configuration |
| `/reload_trade` | POST | Reload trade from exchange |
| `/cancel_open_order` | POST | Cancel open order |

### 1.5 Usage Examples

```bash
# Check health
curl -u freqtrader:freqtrader http://127.0.0.1:8082/api/v1/ping

# Get all trades
curl -u freqtrader:freqtrader http://127.0.0.1:8082/api/v1/trades

# Get profit summary
curl -u freqtrader:freqtrader http://127.0.0.1:8082/api/v1/profit | python3 -m json.tool

# Force short
curl -u freqtrader:freqtrader -X POST "http://127.0.0.1:8082/api/v1/forceentry" \
  -H "Content-Type: application/json" \
  -d '{"pair":"BTC/USDT:USDT","side":"short"}'

# Force exit trade #5
curl -u freqtrader:freqtrader -X POST "http://127.0.0.1:8082/api/v1/forceexit" \
  -H "Content-Type: application/json" \
  -d '{"tradeid":5}'
```

## 2. Telegram Commands

Available through Telegram bot (enabled in config):

| Command | Description |
|---------|-------------|
| `/start` | Start bot |
| `/stop` | Stop bot |
| `/status` | Current trades status |
| `/profit` | P&L summary |
| `/balance` | Wallet balance |
| `/daily` | Daily profit breakdown |
| `/weekly` | Weekly profit |
| `/monthly` | Monthly profit |
| `/performance` | Best/worst performing pairs |
| `/count` | Trade count by side |
| `/forcebuy <pair>` | Force long (if enabled) |
| `/forceshort <pair>` | Force short (if enabled) |
| `/forceexit <trade_id>` | Force close a trade |
| `/reload_trade <trade_id>` | Reload trade state |
| `/delete <trade_id>` | Delete trade record |
| `/cancel_open_order <trade_id>` | Cancel open order |
| `/whitelist` | Show whitelist |
| `/blacklist` | Show blacklist |
| `/blacklist_delete <pair>` | Remove pair from blacklist |
| `/locks` | Show pair locks |
| `/reload_config` | Reload config without restart |
| `/show_config` | Show current configuration |
| `/stopbuy` | Stop new entries |
| `/pause` | Pause trading |
| `/logs <n>` | Last N log entries |
| `/health` | Bot health status |
| `/version` | Bot version |
| `/marketdir` | Market direction/regime |
| `/ev` | Expected value analysis |
| `/help` | Show all commands |

## 3. Database Schema

```sql
-- Trades
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    pair TEXT,
    exchange TEXT,
    is_open INTEGER,
    is_short INTEGER,
    leverage REAL,
    open_rate REAL,
    open_date DATETIME,
    close_rate REAL,
    close_date DATETIME,
    close_profit REAL,
    close_profit_abs REAL,
    stop_loss REAL,
    initial_stop_loss REAL,
    max_rate REAL,
    min_rate REAL,
    enter_tag TEXT,
    exit_reason TEXT,
    stake_amount REAL,
    fee_open REAL,
    fee_close REAL,
    open_trade_value REAL,
    interest_rate REAL,
    funding_fees REAL,
    liquidation_price REAL,
    trading_mode TEXT,
    amount REAL,
    open_order_id TEXT,
    stoploss_order_id TEXT,
    stoploss_last_update DATETIME,
    max_stake_amount REAL,
    realized_profit REAL
);

-- Orders
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    trade_id INTEGER,
    order_id TEXT,
    ft_order_side TEXT,
    pair TEXT,
    order_type TEXT,
    status TEXT,
    amount REAL,
    filled REAL,
    remaining REAL,
    cost REAL,
    rate REAL,
    average REAL,
    order_date DATETIME,
    order_filled_date DATETIME,
    ft_fee_base REAL,
    ft_order_tag TEXT,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);

-- PairLocks
CREATE TABLE pairlocks (
    id INTEGER PRIMARY KEY,
    pair TEXT,
    side TEXT,
    reason TEXT,
    lock_end_time DATETIME,
    active INTEGER
);

-- CustomData
CREATE TABLE custom_data (
    id INTEGER PRIMARY KEY,
    trade_id INTEGER,
    key TEXT,
    value TEXT,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);
```

## 4. Performance Metrics

```bash
# Quick PF calculation
docker exec phoenix-scalper-v3.1-bot sqlite3 \
  /freqtrade/user_data/tradesv3_v3_1.sqlite \
  "SELECT ROUND(SUM(CASE WHEN close_profit_abs > 0 THEN close_profit_abs ELSE 0 END) / 
          ABS(SUM(CASE WHEN close_profit_abs < 0 THEN close_profit_abs ELSE 0 END)), 2) as pf
   FROM trades WHERE is_open=0;"

# Win rate
docker exec phoenix-scalper-v3.1-bot sqlite3 \
  /freqtrade/user_data/tradesv3_v3_1.sqlite \
  "SELECT ROUND(100.0 * SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) || '%' as wr
   FROM trades WHERE is_open=0;"

# Expectancy
docker exec phoenix-scalper-v3.1-bot sqlite3 \
  /freqtrade/user_data/tradesv3_v3_1.sqlite \
  "SELECT ROUND(AVG(close_profit), 4) as expectancy FROM trades WHERE is_open=0;"

# Max drawdown (from equity curve)
# Requires running backtest or EV tracker
```
