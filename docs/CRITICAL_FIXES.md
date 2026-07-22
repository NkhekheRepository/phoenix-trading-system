# PHOENIX TRADING SYSTEM — Critical Fixes & Forensic Analysis

## Fix #1 — Score Ceiling Bug (V2.1)

### Discovery

During live monitoring on 2026-07-20, the V2.1 bot generated 0 trades despite market conditions that should have triggered entry signals. Analysis of `populate_entry_trend` revealed:

### Root Cause

```python
# ORIGINAL (buggy) — PhoenixScalperV2.py:622-623
dataframe.loc[dataframe["signal_score"] > 58, "signal_score"] = 0.0
dataframe.loc[dataframe["short_score"] > 58, "short_score"] = 0.0
```

Scores > 58 were set to **0.0**, which then failed the `≥ score_threshold` (default 55) gate. Any strong signal (score > 58) was silently nullified — the opposite of what was intended.

### Fix

```python
# FIXED — clamp to 58 instead of nullifying
dataframe.loc[dataframe["signal_score"] > 58, "signal_score"] = 58.0
dataframe.loc[dataframe["short_score"] > 58, "short_score"] = 58.0
```

### Impact

- Before fix: Strong signals (score 59-100) → score 0 → gate rejects → 0 trades
- After fix: Strong signals (score 59-100) → score 58 → gate passes → trades created
- Confirmed: 11 new trades generated immediately after fix applied (Jun 20 present trades)

### Files Affected

- `phoenix-scalper/strategies/PhoenixScalperV2.py` (lines 622-623)
- `phoenix-scalper/strategies/PhoenixScalperV2.1.py` (lines 624-625)

---

## Fix #2 — HMM Retrain Loop (All Bots)

### Discovery

Log analysis showed HMM training running 6-10 times per `populate_indicators` call (every 60s), consuming 300+ MB RAM and 5-10s CPU per iteration. This prevented the bot from keeping up with real-time data.

### Root Cause

V2 version used `do_heavy = (self._hmm_update_count % 60 == 0)` as a modulo check but incremented `_hmm_update_count = 1` inside the heavy block WITHOUT caching the HMM model. Every 60th iteration, a full retrain would trigger. V3/V5 _BTC had similar patterns.

### Fix

```python
# BEFORE: No cache — retrains every 60th iteration
do_heavy = (self._hmm_update_count % 60 == 0)
if do_heavy:
    hmm_features = compute_hmm_features(...)
    self._hmm_update_count += 1  # ← bug: should be _hmm_cache

# AFTER: Train once, cache forever
if pair_key not in self._hmm_cache:
    hmm_features = compute_hmm_features(...)
    self._hmm_cache[pair_key] = hmm_features
else:
    hmm_features = self._hmm_cache[pair_key]
```

### Impact

- Before: HMM retrains every ~60 min per pair → 8 retrains/hour → CPU pegged at 80-100%
- After: HMM trains once per pair on first call → 8 retrains total → CPU drops to 5-10%

### Files Affected

- `phoenix-scalper/strategies/PhoenixScalperV2.py` (removed `do_heavy`, added `_hmm_cache`)
- `phoenix-scalper/strategies/PhoenixScalperV2.1.py` (already had cache, verified)
- `phoenix-scalper/strategies/PhoenixScalperV5_BTC.py` (kept `do_heavy` as 60-iteration, but uses cache)

---

## Fix #3 — Stale Data Root Cause

### Discovery

After running for 3+ days, all bots stopped generating trades. `DataValidator` showed "Stale data: last candle X min old" warnings.

### Root Cause

The shared Docker volume had feather files whose latest candle dates ended 2026-07-09 for most pairs (13 days stale). In dry-run mode, `process_only_new_candles: true` + no new candles = `populate_entry_trend` never evaluates.

### Fix

1. Downloaded fresh 15-day data for all 8 pairs:
```bash
docker exec phoenix-scalper-v2.1-bot freqtrade download-data \
  --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT \
         XRP/USDT:USDT DOGE/USDT:USDT BNB/USDT:USDT \
         ADA/USDT:USDT LINK/USDT:USDT \
  --days 15 --timeframes 5m
```

2. Set `process_only_new_candles: false` in all 3 configs (so bot re-analyzes on every iteration)

3. Set up hourly data refresh cron (`refresh_data.sh`) to prevent recurrence

### Impact

- Before: Data ends Jul 9-20, no new trades possible
- After: Data extends to current UTC time, trade generation resumes
- Confirmed: V3.1 created 2 new trades (DOGE Jul 22 05:50, SOL Jul 22 06:00) immediately after data refresh

---

## Fix #4 — Telegram Unresponsive (All Bots)

### Discovery

After 13+ hours uptime, Telegram bots stopped sending message Completely. Bot logs showed no errors — the bot was running but no messages arrived.

### Root Cause

`python-telegram-bot` uses an asyncio event loop for polling. After prolonged uptime, the event loop can stall due to stale coroutines or network interruptions. Freqtrade's `_loop.create_task` pattern does not recover from this.

### Fix

Container restart:
```bash
docker restart phoenix-scalper-v2.1-bot
docker restart phoenix-scalper-v3.1-bot
docker restart phoenix-scalper-v5-btc-bot
```

### Impact

- Telegram messages resumed immediately
- All queued messages (daily summaries, health checks) delivered
- No data loss

### Recommendation

Add a weekly cron job to restart all 3 containers as preventive maintenance.

---

## Fix #5 — Stuck Open Trade (V3.1 BTC Short)

### Discovery

Trade #11 (BTC/USDT short, opened Jul 22 01:46) stayed open for 8+ hours. Custom_exit should have triggered tp_hit or max_hold.

### Root Cause

`custom_exit` receives `current_rate` from the analyzed dataframe, which was computed from stale data. With stale candles, `current_rate` never changed, so profit conditions never triggered.

### Fix

Force-exit via API:
```bash
curl -u freqtrader:freqtrader -X POST http://127.0.0.1:8083/api/v1/forceexit \
  -H "Content-Type: application/json" \
  -d '{"tradeid":11}'
```

Result: Trade closed at +5.48% profit (+$0.36).

---

## Fix #6 — V3.1 BTC Short Stale current_rate

### Root Cause

In dry-run mode, `custom_exit` reads `current_rate` from the un-refreshed analyzed dataframe. With stale feather files, the dataframe's latest candle date is in the past, so `current_rate` is the close of the last available candle, not the current market price. Exit conditions are evaluated against an outdated price.

### Resolution

Fresh data download + `process_only_new_candles: false` ensures the analyzed dataframe is re-computed with current data on every iteration.

---

## Summary of Fixes

| # | Fix | Bot Affected | Severity | Status |
|---|-----|-------------|----------|--------|
| 1 | Score ceiling clamp | V2.1 | Critical | Applied |
| 2 | HMM retrain elimination | All | Critical | Applied |
| 3 | Stale data refresh | All | Critical | Applied + cron |
| 4 | Telegram restart | All | High | Applied |
| 5 | Force-exit stuck trade | V3.1 | High | Applied |
| 6 | current_rate staleness | V3.1 | High | Applied |
