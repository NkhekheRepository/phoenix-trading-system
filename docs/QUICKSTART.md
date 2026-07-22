# PHOENIX TRADING SYSTEM — Quickstart (60 seconds)

```bash
# 1. Clone
git clone --depth 1 https://github.com/NkhekheRepository/phoenix-trading-system.git
cd phoenix-trading-system

# 2. Configure
cp phoenix-scalper/.env.example phoenix-scalper/.env
nano phoenix-scalper/.env     # Fill in Telegram + Exchange keys

# 3. Build & start all 3 bots
cd phoenix-scalper
docker compose -f docker-compose-v2.1.yml up -d --build && \
docker compose -f docker-compose-v3.1.yml up -d --build && \
docker compose -f docker-compose-v5-btc.yml up -d --build

# 4. Download market data
docker exec phoenix-scalper-v2.1-bot freqtrade download-data \
  --config /freqtrade/config.json \
  --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT \
         XRP/USDT:USDT DOGE/USDT:USDT BNB/USDT:USDT \
         ADA/USDT:USDT LINK/USDT:USDT \
  --days 15 --timeframes 5m

# 5. Verify
docker ps && echo "---" && \
curl -s -u freqtrader:freqtrader http://127.0.0.1:8082/api/v1/ping && \
echo " System ready!"
```

## Bot Ports

| Bot | API | Check |
|-----|-----|-------|
| V2.1 | `http://127.0.0.1:8082` | `curl -u freqtrader:freqtrader http://127.0.0.1:8082/api/v1/ping` |
| V3.1 | `http://127.0.0.1:8083` | `curl -u freqtrader:freqtrader http://127.0.0.1:8083/api/v1/ping` |
| V5-BTC | `http://127.0.0.1:8085` | `curl -u freqtrader:freqtrader http://127.0.0.1:8085/api/v1/ping` |

## Telegram Commands

- `/start` — Start bot
- `/stop` — Stop bot
- `/status` — Current trades
- `/profit` — P&L summary
- `/forcebuy BTC/USDT:USDT` — Force long (if enabled)
- `/forceshort BTC/USDT:USDT` — Force short (if enabled)
- `/forceexit 5` — Force close trade #5
- `/daily` — Daily profit
- `/whitelist` — Show whitelist
- `/blacklist` — Show blacklist
- `/reload_config` — Reload config without restart

## Health Check

```bash
# Check all 3 bots in one line
for b in phoenix-scalper-v2.1-bot phoenix-scalper-v3.1-bot phoenix-scalper-v5-btc-bot; do
  echo "$b: $(docker inspect $b --format '{{.State.Status}}')"
done
```
