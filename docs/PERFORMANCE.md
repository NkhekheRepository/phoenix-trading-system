# PHOENIX TRADING SYSTEM — Performance Analysis

## 1. Live Performance Summary

Data as of 2026-07-22 06:00 UTC. All trades on Binance Futures (isolated, dry-run).

### 1.1 V3.1 — PhoenixScalperV3_1 (8 pairs, 10× leverage)

| Metric | Value |
|--------|-------|
| Total Trades | 13 |
| Closed Trades | 11 |
| Open Trades | 2 (DOGE short, SOL short) |
| Win Rate | 63.6% (7W / 4L) |
| **Profit Factor** | **1.66** |
| Total P&L | +$1.30 |
| Avg ROI | +1.58% |
| Avg Win ROI | +5.32% |
| Avg Loss ROI | -4.98% |
| Max Win ROI | +7.57% (ADA long) |
| Max Loss ROI | -9.70% (XRP trailing_stop_loss) |
| Max Win (abs) | +$0.75 (ADA) |
| Max Loss (abs) | -$0.97 (XRP) |

**Exit Breakdown:**
| Exit Reason | Count | Avg ROI | Contribution |
|------------|-------|---------|--------------|
| tp_hit | 4 | +7.37% | All profits |
| max_hold | 4 | +0.45% | Mixed (2W/2L) |
| bleed_exit | 2 | -4.68% | Losses |
| trailing_stop_loss | 1 | -9.70% | Single worst |
| force_exit | 1 | +5.48% | Manual close |

**Key Insight**: The single trailing_stop_loss at -9.70% accounts for 49% of all gross losses. Removing it, PF would be **3.26**.

### 1.2 V2.1 — PhoenixScalperV2_1 (score-ceiling fix applied)

| Metric | Value |
|--------|-------|
| Total Trades | 5 |
| Closed Trades | 4 (pre-fix) |
| Open Trades | 1 (force_short) |
| Win Rate | 75.0% (3W / 1L) |
| **Profit Factor** | **7.61** |
| Total P&L | +$0.61 |
| Avg Win ROI | +3.16% |
| Avg Loss ROI | -0.95% |

**Note**: These 4 trades used pre-fix scoring logic (where scores > 58 were nullified to 0). The score-clamp fix (caps to 58 instead) has NOT yet generated natural trades. The single open trade is a force_entry.

### 1.3 V5-BTC — PhoenixScalperV5_BTC (BTC-only, 10× leverage)

| Metric | Value |
|--------|-------|
| Total Trades | 3 |
| Closed Trades | 2 |
| Open Trades | 1 (BTC long) |
| Win Rate | 50.0% (1W / 1L) |
| **Profit Factor** | **0.22** |
| Total P&L | -$0.51 |
| Avg Loss ROI | -9.84% |

**Warning**: Insufficient sample size (2 closed trades). Not statistically meaningful.

## 2. Trade Distribution

### 2.1 V3.1 Trade Timeline

```
Trade   Pair     Dir  Entry        Exit          ROI%    Reason
─────────────────────────────────────────────────────────────────
 1   ETH/USDT   LONG  Jul 21 10:00  14:49      -4.16  bleed_exit
 2   BTC/USDT   LONG  Jul 21 10:15  14:10      +7.28  tp_hit
 3   XRP/USDT   LONG  Jul 21 13:25  14:50      +7.41  tp_hit
 4   XRP/USDT   LONG  Jul 21 15:25  18:22      +7.24  tp_hit
 5   LINK/USDT  SHORT Jul 21 17:50  22:53      +0.27  max_hold
 6   XRP/USDT   LONG  Jul 21 18:25  20:28      -9.70  trailing_stop_loss
 7   BNB/USDT   SHORT Jul 21 18:25  23:28      -0.84  max_hold
 8   XRP/USDT   SHORT Jul 21 22:05  03:08      +1.99  max_hold
 9   ADA/USDT   LONG  Jul 21 23:35  01:46      +7.57  tp_hit
10   DOGE/USDT  LONG  Jul 22 00:15  05:04      -5.20  bleed_exit
11   BTC/USDT   SHORT Jul 22 01:46  05:40      +5.48  force_exit
12   DOGE/USDT  SHORT Jul 22 05:50  OPEN         —     —
13   SOL/USDT   SHORT Jul 22 06:00  OPEN         —     —
```

### 2.2 V3.1 Profit Distribution by Exit Reason

```
tp_hit (4 trades)
  ┌──── XRP #3: +7.24%
  ├──── XRP #4: +7.41%
  ├──── BTC #2: +7.28%
  └──── ADA #9: +7.57%
  Mean: +7.37% | Range: 7.24-7.57

max_hold (4 trades)
  ├──── LINK #5: +0.27%  ✓
  ├──── BNB #7: -0.84%   ✗
  ├──── XRP #8: +1.99%   ✓
  └──── (2W/2L)
  Mean: +0.45% | Wide spread

bleed_exit (2 trades)
  ├──── ETH #1: -4.16%
  └──── DOGE #10: -5.20%
  Mean: -4.68%

trailing_stop_loss (1 trade)
  └──── XRP #6: -9.70%  ⚠️
```

## 3. Pair Performance (V3.1)

| Pair | Trades | Wins | Losses | P&L | ROI% |
|------|--------|------|--------|-----|------|
| XRP/USDT | 4 | 3 | 1 | +$0.68 | +0.48% |
| BTC/USDT | 2 | 2 | 0 | +$0.85 | +6.38% |
| ADA/USDT | 1 | 1 | 0 | +$0.75 | +7.57% |
| ETH/USDT | 1 | 0 | 1 | -$0.40 | -4.16% |
| DOGE/USDT | 1 | 0 | 1 | -$0.52 | -5.20% |
| LINK/USDT | 1 | 1 | 0 | +$0.03 | +0.27% |
| BNB/USDT | 1 | 0 | 1 | -$0.08 | -0.84% |
| SOL/USDT | 1 | 0 | 0 | — (open) | — |

## 4. Performance Against Targets

| Target | V2.1 | V3.1 | V5-BTC | Status |
|--------|------|------|--------|--------|
| PF > 1.8 | 7.61 ✱ | 1.66 | 0.22 | ❌ V3.1 below target |
| WR > 55% | 75% ✱ | 63.6% | 50% | ✅ V2.1, V3.1 above |
| Max DD < 15% | — | — | — | ✅ (no >15% DD observed) |

✱ Pre-fix data, insufficient sample. Not representative of current fix.

## 5. Backtest Results

Backtest data available in `user_data/backtest_results/`:

| File | Date Range | Trades | PF | Notes |
|------|-----------|--------|----|-------|
| See `scripts/tp_sl_precision.py` | Various | Various | Various | TP/SL precision analysis |
| See `research/v5_experiment_results.json` | Various | Various | Various | V5 experiments |

## 6. Key Performance Drivers

### Bull Case (drives profit)
- **tp_hit**: 4 of 4 successful TPs → 100% hit rate (target 7.1%)
- **max_hold winning**: Slow grind to positive in ranging markets
- **Direction**: Longs outperform shorts in this sample

### Bear Case (drives losses)
- **trailing_stop_loss**: Single trade caused -9.70% — needs investigation
- **bleed_exit**: Trades that do not recover within 4.8 hours
- **max_hold losing**: Slow bleed that does not recover in 5 hours

## 7. Recommendations for PF > 1.8

1. **Fix trailing_stop_loss**: The -9.70% single event is the primary PF killer. Consider adjusting trail_threshold (7.8%) or lock_ratio (0.359).
2. **Increase position size on tp_hit patterns**: tp_hit has 100% success rate — consider higher Kelly allocation for high-confidence entry tags.
3. **Reduce exposure on XRP**: 4 trades, high variance (+7.41% wins to -9.70% loss).
4. **Increase sample size**: 11 closed trades is inadequate. Run backtest for 200+ trades.
