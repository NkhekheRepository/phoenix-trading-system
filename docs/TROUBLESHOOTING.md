# PHOENIX TRADING SYSTEM — Troubleshooting Guide

## 1. Bot Not Generating Trades

### Symptom: No new trades created for hours

**Check 1: Data Freshness**
```bash
docker exec phoenix-scalper-v2.1-bot python3 -c "
import pandas as pd
from pathlib import Path
import glob
for f in sorted(glob.glob('/freqtrade/user_data/data/binance/futures/*5m-futures.feather')):
    name = f.split('/')[-1].split('_')[0]
    if name in ['BTC','ETH','SOL','XRP','DOGE','BNB','ADA','LINK']:
        df = pd.read_feather(f)
        print(f'{name}: last candle {df[\"date\"].max()}')
"
```

**Fix**: If last candle is > 10 minutes old, run data refresh:
```bash
docker exec phoenix-scalper-v2.1-bot freqtrade download-data \
  --config /freqtrade/config.json \
  --pairs BTC/USDT:USDT ETH/USDT:USDT ... \
  --days 2 --timeframes 5m
```

**Check 2: Entry Signals on Latest Candle**

The bot only checks the LATEST candle for entry signals. If no candle in the last 5 minutes has an entry condition met, no trade is created. This is **normal behavior**.

**Check 3: process_only_new_candles**

In dry-run mode, this should be `false`:
```bash
docker exec phoenix-scalper-v2.1-bot grep process_only_new_candles /freqtrade/config.json
```

**Check 4: Force a trade to verify pipeline**
```bash
curl -u freqtrader:freqtrader -X POST "http://127.0.0.1:8082/api/v1/forcebuy" \
  -H "Content-Type: application/json" \
  -d '{"pair":"BTC/USDT:USDT","side":"short"}'
```

---

## 2. Telegram Not Working

### Symptom: No Telegram messages

**Check 1: Bot token**
```bash
docker exec phoenix-scalper-v2.1-bot grep "token" /freqtrade/config.json
```
Should be a valid token format: `1234567890:ABCdef...`

**Check 2: Chat ID**
```bash
docker exec phoenix-scalper-v2.1-bot grep "chat_id" /freqtrade/config.json
```

**Check 3: Restart bot** (fixes stale event loop)
```bash
docker restart phoenix-scalper-v2.1-bot
```

**Check 4: Test Telegram API**
```bash
curl -s "https://api.telegram.org/bot${YOUR_TOKEN}/getMe"
```

---

## 3. Score Ceiling — Trades Not Created

### Symptom: Log shows scores > 58 but no trades

**Check**: The fix (clamp to 58 instead of zero) must be in place:
```bash
grep -n "score.*58" phoenix-scalper/strategies/PhoenixScalperV2.1.py
```
Expected:
```
624: dataframe.loc[dataframe["signal_score"] > 58, "signal_score"] = 58.0
625: dataframe.loc[dataframe["short_score"] > 58, "short_score"] = 58.0
```

If it shows `= 0.0` instead, apply the fix.

---

## 4. HMM Training Loop — High CPU

### Symptom: CPU 80-100%, constant HMM training in logs

**Check 1**: How often HMM trains per pair:
```bash
docker exec phoenix-scalper-v2.1-bot grep "HMM iter" /freqtrade/user_data/logs/freqtrade_v2_1.log | tail -20
```

**Fix**: The cache check should be:
```python
if pair_key not in self._hmm_cache:
    # train once
    self._hmm_cache[pair_key] = hmm_features
else:
    # use cached
    hmm_features = self._hmm_cache[pair_key]
```

---

## 5. Container Not Starting

### Symptom: Container exits immediately

**Check 1: Logs**
```bash
docker logs phoenix-scalper-v2.1-bot --tail 50
```

**Common causes**:
- Invalid config JSON → fix syntax errors
- Missing strategy file → check `strategy_path` and `strategy` name match
- Port conflict → change `listen_port` in config
- Missing DB directory → ensure `./user_data/` exists

**Check 2: Config syntax**
```bash
docker run --rm -v $(pwd)/config-v2.1.json:/config.json \
  phoenix-scalper-phoenix-scalper:latest \
  python3 -c "import json; json.load(open('/config.json')); print('OK')"
```

---

## 6. Database Issues

### Symptom: Trade count mismatch

**Check**: Each bot has its own DB:
- V2.1 → `tradesv3_v2_1.sqlite`
- V3.1 → `tradesv3_v3_1.sqlite`
- V5-BTC → `tradesv3.sqlite`

**Repair**: If DB is corrupted, stop the bot, delete the .sqlite-wal/-shm files, and restart:
```bash
docker stop phoenix-scalper-v2.1-bot
rm -f /freqtrade/user_data/tradesv3_v2_1.sqlite*
docker start phoenix-scalper-v2.1-bot
```

---

## 7. Login Issues

### Symptom: "Invalid API key" or exchange errors

**Check 1**: API key permissions (Binance Futures must be enabled)
**Check 2**: API key has not expired (valid on Binance account)
**Check 3**: Correct pair format (`BTC/USDT:USDT` for futures, not `BTC/USDT`)
**Check 4**: Trading mode matches (`trading_mode: futures`)

---

## 8. Force Entry Not Working

### Symptom: `Force_entry not enabled.` error

**Fix**: Add to config:
```json
"force_entry_enable": true
```
Then restart.

---

## 9. Container Has No Strategy File

### Symptom: "Strategy PhoenixScalperV2_1 not found"

**Fix**: The strategies directory must be bind-mounted:
```yaml
volumes:
  - ./strategies:/freqtrade/strategies
```

Verify inside container:
```bash
docker exec phoenix-scalper-v2.1-bot ls /freqtrade/strategies/PhoenixScalperV2.1.py
```

---

## 10. Cron Jobs Not Running

### Symptom: No market data refresh

**Check**:
```bash
crontab -l
grep refresh_data /var/log/syslog | tail -5
```

**Fix**: Ensure the script is executable:
```bash
chmod +x /home/nkhekhe/phoenix-trading-system/scripts/refresh_data.sh
```

## 11. Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| Telegram asyncio stall after 12h+ uptime | Open | Weekly restart via cron |
| Dry-run only evaluates last candle | By design | Refresh data hourly via cron for new candles |
| HMM look-ahead (trained on full sequence) | Accepted | Mitigated by caching + startup period |
| V5-BTC insufficient sample | Open | Let it accumulate more trades |
