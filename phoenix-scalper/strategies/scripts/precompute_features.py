#!/usr/bin/env python3
"""
Phase 1: Pre-compute all features + forward labels for 8 pairs over 3-month range.
Output: single feather file ~170 MB, used by Optuna for fast trial evaluation.

Saves: /freqtrade/user_data/data/precomputed_features.feather

Run inside container:
  docker exec phoenix-scalper-v2.1-bot python /freqtrade/strategies/scripts/precompute_features.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib.abstract as ta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # /freqtrade/
from ml.kalman_filter import compute_kalman_features
from ml.hmm_regime import compute_hmm_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger('precompute')

DATA_DIR = '/freqtrade/user_data/data/binance/futures'
PAIRS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'BNB', 'ADA', 'LINK']
FORWARD_BARS = 48
STARTUP_BARS = 150


def _align(values, target_len):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == target_len:
        return v
    if len(v) < target_len:
        p = np.full(target_len, np.nan, dtype=np.float64)
        p[:len(v)] = v
        p[len(v):] = v[-1] if len(v) > 0 else 0.0
        return p
    return v[-target_len:]


def process_pair(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f'{symbol}_USDT_USDT-5m-futures.feather')
    if not os.path.exists(path):
        logger.warning(f'Missing: {path}')
        return None
    df = pd.read_feather(path)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pair'] = f'{symbol}/USDT:USDT'
    logger.info(f'{symbol}: {len(df)} candles | {df["date"].min()} to {df["date"].max()}')

    n = len(df)

    # --- Technical indicators ---
    df['rsi_14'] = ta.RSI(df, timeperiod=14)
    df['adx'] = ta.ADX(df, timeperiod=14)
    df['plus_di'] = ta.PLUS_DI(df, timeperiod=14)
    df['minus_di'] = ta.MINUS_DI(df, timeperiod=14)

    bb = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    df['bb_upper'] = bb['upperband']
    df['bb_middle'] = bb['middleband']
    df['bb_lower'] = bb['lowerband']

    macd = ta.MACD(df, fastperiod=8, slowperiod=17, signalperiod=5)
    df['macd'] = macd['macd']
    df['macdsignal'] = macd['macdsignal']
    df['macdhist'] = macd['macdhist']

    df['volume_ema'] = ta.EMA(df['volume'].values, timeperiod=10)
    df['volume_ratio'] = df['volume'] / (df['volume_ema'] + 1e-10)
    df['atr'] = ta.ATR(df, timeperiod=14)
    df['atr_pct'] = df['atr'] / (df['close'] + 1e-10)
    df['obv'] = ta.OBV(df)

    for p in range(5, 26):
        df[f'ema_{p}'] = ta.EMA(df['close'].values, timeperiod=p)
    for p in [50, 200]:
        df[f'ema_{p}'] = ta.EMA(df['close'].values, timeperiod=p)
    for p in range(5, 13):
        df[f'rsi_{p}'] = ta.RSI(df, timeperiod=p)

    # --- Kalman Filter ---
    try:
        kf = compute_kalman_features(
            df['close'].values, df['atr_pct'].values, df['volume_ratio'].values
        )
        for col, vals in kf.items():
            df[col] = _align(vals, n)
    except Exception as e:
        logger.warning(f'{symbol} KF failed: {e}')
        for col in ['kf_price','kf_trend','kf_prediction','kf_confidence','kf_direction',
                    'kf_innovation','kf_S','kf_price_momentum','kf_trend_acceleration',
                    'kf_prediction_error','kf_regime_score','kf_vol_of_trend','kf_atr_ratio']:
            df[col] = 0.0

    # --- HMM ---
    try:
        returns = np.log(df['close'] / df['close'].shift(1))
        vol = returns.rolling(10).std()
        volume_change = df['volume_ratio'].pct_change()
        hmm = compute_hmm_features(
            returns.fillna(0).values, vol.fillna(0).values, volume_change.fillna(0).values
        )
        for col, vals in hmm.items():
            df[col] = _align(vals, n)
    except Exception as e:
        logger.warning(f'{symbol} HMM failed: {e}')
        for col, val in {'hmm_regime':1,'hmm_p_bull':0.33,'hmm_p_range':0.34,
                         'hmm_p_bear':0.33,'hmm_regime_stability':0.5,
                         'hmm_transition_risk':0.3,'hmm_vol_regime':1.0,
                         'hmm_trend_strength':1.0}.items():
            df[col] = val

    # --- Entry scores (same as strategy) ---
    rsi_v = df['rsi_14'].values
    ema_s = df['ema_14'].values
    ema_f = df['ema_7'].values
    ema_50 = df['ema_50'].values

    hmm_bull = np.minimum(df['hmm_p_bull'].values / 0.6, 1.0) * 20
    trend_str = np.minimum(df['adx'].values / 40.0, 1.0) * 15
    kalman_c = np.minimum(df['kf_confidence'].values / 0.8, 1.0) * 10
    di_d = df['plus_di'].values - df['minus_di'].values
    directional = np.clip(di_d / 20.0 + 0.5, 0, 1) * 10
    mom_a = np.minimum(np.maximum(df['kf_trend_acceleration'].values, 0), 1)
    mom_p = np.minimum(np.maximum(df['kf_price_momentum'].values, 0), 1)
    momentum = (mom_a + mom_p) / 2.0 * 10
    volume = np.minimum(df['volume_ratio'].values / 3.0, 1.0) * 10
    stability = (1.0 - np.minimum(df['hmm_regime_stability'].values / 0.5, 1.0)) * 10
    rsi_s = np.maximum(0, 1.0 - np.abs(rsi_v - 45) / 25.0) * 5
    ema_dist = np.abs(df['close'].values / ema_s - 1.0)
    pullback = np.maximum(0, 1.0 - ema_dist / 0.02) * 5
    trend_al = ((df['close'].values > ema_f).astype(float) +
                (df['close'].values > ema_s).astype(float) +
                (df['close'].values > ema_50).astype(float)) / 3.0 * 5

    df['long_score'] = hmm_bull + trend_str + kalman_c + directional + momentum + volume + stability + rsi_s + pullback + trend_al
    df.loc[df['long_score'] > 58, 'long_score'] = 58.0

    hmm_bear = np.minimum(df['hmm_p_bear'].values / 0.6, 1.0) * 20
    di_s = df['minus_di'].values - df['plus_di'].values
    directional_s = np.clip(di_s / 20.0 + 0.5, 0, 1) * 10
    mom_a_s = np.minimum(np.maximum(-df['kf_trend_acceleration'].values, 0), 1)
    mom_p_s = np.minimum(np.maximum(-df['kf_price_momentum'].values, 0), 1)
    momentum_s = (mom_a_s + mom_p_s) / 2.0 * 10
    rsi_s_s = np.maximum(0, 1.0 - np.abs(rsi_v - 70) / 20.0) * 5
    bk_dist = np.abs(df['close'].values / ema_s - 1.0)
    breakdown = np.maximum(0, 1.0 - bk_dist / 0.02) * 5
    bear_al = ((df['close'].values < ema_f).astype(float) +
               (df['close'].values < ema_s).astype(float) +
               (df['close'].values < ema_50).astype(float)) / 3.0 * 5

    df['short_score'] = hmm_bear + trend_str + kalman_c + directional_s + momentum_s + volume + stability + rsi_s_s + breakdown + bear_al
    df.loc[df['short_score'] > 58, 'short_score'] = 58.0

    # --- Forward labels ---
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n_candles = n

    long_targets = np.full(n_candles, np.nan)
    short_targets = np.full(n_candles, np.nan)
    long_return = np.full(n_candles, np.nan)
    short_return = np.full(n_candles, np.nan)
    long_bars = np.full(n_candles, np.nan)
    short_bars = np.full(n_candles, np.nan)

    for i in range(STARTUP_BARS, n_candles - FORWARD_BARS):
        entry = close[i]
        forward_high = high[i+1:i+FORWARD_BARS+1]
        forward_low = low[i+1:i+FORWARD_BARS+1]
        forward_close = close[i+1:i+FORWARD_BARS+1]

        max_runup = (forward_high.max() - entry) / entry
        max_dd = (entry - forward_low.min()) / entry
        final_ret = (forward_close[-1] - entry) / entry

        # Long: take-profit at +3%, stop-loss at -2%, else final return
        first_tp = np.where(forward_high >= entry * 1.03)[0]
        first_sl = np.where(forward_low <= entry * 0.98)[0]
        if len(first_tp) > 0 and (len(first_sl) == 0 or first_tp[0] < first_sl[0]):
            long_targets[i] = 1
            long_return[i] = 0.03
            long_bars[i] = first_tp[0] + 1
        elif len(first_sl) > 0:
            long_targets[i] = 0
            long_return[i] = -0.02
            long_bars[i] = first_sl[0] + 1
        else:
            long_targets[i] = 1 if final_ret > 0 else 0
            long_return[i] = final_ret
            long_bars[i] = FORWARD_BARS

        # Short: take-profit at +3%, stop-loss at -2%, else final return
        first_tp_s = np.where(forward_low <= entry * 0.97)[0]
        first_sl_s = np.where(forward_high >= entry * 1.02)[0]
        if len(first_tp_s) > 0 and (len(first_sl_s) == 0 or first_tp_s[0] < first_sl_s[0]):
            short_targets[i] = 1
            short_return[i] = 0.03
            short_bars[i] = first_tp_s[0] + 1
        elif len(first_sl_s) > 0:
            short_targets[i] = 0
            short_return[i] = -0.02
            short_bars[i] = first_sl_s[0] + 1
        else:
            short_targets[i] = 1 if final_ret < 0 else 0
            short_return[i] = -final_ret
            short_bars[i] = FORWARD_BARS

    df['long_target'] = long_targets
    df['short_target'] = short_targets
    df['long_return'] = long_return
    df['short_return'] = short_return
    df['long_bars'] = long_bars
    df['short_bars'] = short_bars

    return df


def main():
    t0 = time.time()
    all_frames = []
    for sym in PAIRS:
        df = process_pair(sym)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        logger.error('No data processed')
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values(['date', 'pair'], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    out_path = '/freqtrade/user_data/data/precomputed_features.feather'
    combined.to_feather(out_path)
    elapsed = time.time() - t0
    logger.info(f'Saved {len(combined)} rows x {len(combined.columns)} cols to {out_path}')
    logger.info(f'Elapsed: {elapsed:.0f}s')
    logger.info(f'Long targets: 0={int((combined["long_target"]==0).sum())}  1={int((combined["long_target"]==1).sum())}  nan={int(combined["long_target"].isna().sum())}')
    logger.info(f'Short targets: 0={int((combined["short_target"]==0).sum())}  1={int((combined["short_target"]==1).sum())}  nan={int(combined["short_target"].isna().sum())}')


if __name__ == '__main__':
    main()
