#!/usr/bin/env bash
set -euo pipefail

LOGFILE="/home/nkhekhe/phoenix-trading-system/scripts/refresh_data.log"
PAIRS_JSON='["BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","XRP/USDT:USDT","DOGE/USDT:USDT","BNB/USDT:USDT","ADA/USDT:USDT","LINK/USDT:USDT"]'

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting data refresh..." >> "$LOGFILE"

docker exec phoenix-scalper-v2.1-bot sh -c "echo '$PAIRS_JSON' > /tmp/pairs8.json"

docker exec phoenix-scalper-v2.1-bot freqtrade download-data \
  --config /freqtrade/config.json \
  --pairs-file /tmp/pairs8.json \
  --days 2 \
  --timeframes 5m \
  >> "$LOGFILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Data refresh complete." >> "$LOGFILE"
