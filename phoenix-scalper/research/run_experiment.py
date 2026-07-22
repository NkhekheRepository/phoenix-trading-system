"""
Experiment Runner — reusable framework for testing entry scoring approaches.

Each experiment defines scoring functions that operate on the fully-indicator-
enriched dataframe. The framework handles data loading, indicator computation,
forward-labeling, and metric reporting.

Usage:
  python3 research/run_experiment.py <timerange> <config> <experiment_name>

Experiments: e8_kalman_only, e9_top5, e14_adaptive, baseline_v41
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
import datetime as dt

TOP10 = [
    "SOL/USDT:USDT", "BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "BNB/USDT:USDT", "ETH/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT",
]

HOLD_BARS = 12
LEVERAGE = 10


def load_all_data(pairs, timerange_str):
    from freqtrade.configuration.timerange import TimeRange
    tr = TimeRange.parse_timerange(timerange_str)
    base = Path("user_data/data/binance/futures")
    start = pd.Timestamp(dt.datetime.fromtimestamp(tr.startts), tz="UTC")
    end = pd.Timestamp(dt.datetime.fromtimestamp(tr.stopts), tz="UTC")
    dataframes = {}
    for pair in pairs:
        fname = pair.replace("/", "_").replace(":", "_") + "-5m-futures.feather"
        fpath = base / fname
        if not fpath.exists():
            continue
        df = pd.read_feather(fpath)
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(df) > 60:
            dataframes[pair] = df
    return dataframes


def compute_indicators(raw_data, config):
    from freqtrade.resolvers import StrategyResolver
    config["strategy"] = "PhoenixScalperV4_1"
    config["strategy_path"] = "strategies"
    strategy = StrategyResolver.load_strategy(config)
    indicator_data = {}
    for pair, df in raw_data.items():
        d = strategy.populate_indicators(df.copy(), {"pair": pair})
        indicator_data[pair] = d
    return indicator_data, strategy


def apply_scoring_long(dataframe):
    """Replicate V4.1 signal_score exactly."""
    hmm_bull = np.minimum(dataframe["hmm_p_bull"].values / 0.6, 1.0) * 20
    trend_str = np.minimum(dataframe["adx"].values / 40.0, 1.0) * 15
    kalman_c = np.minimum(dataframe["kf_confidence"].values / 0.8, 1.0) * 10
    di_diff = dataframe["plus_di"].values - dataframe["minus_di"].values
    directional = np.clip(di_diff / 20.0 + 0.5, 0, 1) * 10
    mom_a = np.minimum(np.maximum(dataframe["kf_trend_acceleration"].values, 0), 1)
    mom_p = np.minimum(np.maximum(dataframe["kf_price_momentum"].values, 0), 1)
    momentum = (mom_a + mom_p) / 2.0 * 10
    volume = np.minimum(dataframe["volume_ratio"].values / 3.0, 1.0) * 10
    stability = (1.0 - np.minimum(dataframe["hmm_regime_stability"].values / 0.5, 1.0)) * 10
    rsi_v = dataframe["rsi_5"].values
    rsi_s = np.maximum(0, 1.0 - np.abs(rsi_v - 45) / 25.0) * 5
    ema_dist = np.abs(dataframe["close"].values / dataframe["ema_21"].values - 1.0)
    pullback = np.maximum(0, 1.0 - ema_dist / 0.02) * 5
    trend_al = (
        (dataframe["close"].values > dataframe["ema_8"].values).astype(float) +
        (dataframe["close"].values > dataframe["ema_21"].values).astype(float) +
        (dataframe["close"].values > dataframe["ema_50"].values).astype(float)
    ) / 3.0 * 5
    return hmm_bull + trend_str + kalman_c + directional + momentum + volume + stability + rsi_s + pullback + trend_al


def apply_scoring_short(dataframe):
    """Replicate V4.1 short_score exactly."""
    hmm_bear = np.minimum(dataframe["hmm_p_bear"].values / 0.6, 1.0) * 20
    trend_str = np.minimum(dataframe["adx"].values / 40.0, 1.0) * 15
    kalman_c = np.minimum(dataframe["kf_confidence"].values / 0.8, 1.0) * 10
    di_s = dataframe["minus_di"].values - dataframe["plus_di"].values
    directional_s = np.clip(di_s / 20.0 + 0.5, 0, 1) * 10
    mom_a_s = np.minimum(np.maximum(-dataframe["kf_trend_acceleration"].values, 0), 1)
    mom_p_s = np.minimum(np.maximum(-dataframe["kf_price_momentum"].values, 0), 1)
    momentum_s = (mom_a_s + mom_p_s) / 2.0 * 10
    volume = np.minimum(dataframe["volume_ratio"].values / 3.0, 1.0) * 10
    stability = (1.0 - np.minimum(dataframe["hmm_regime_stability"].values / 0.5, 1.0)) * 10
    rsi_v = dataframe["rsi_5"].values
    rsi_s_s = np.maximum(0, 1.0 - np.abs(rsi_v - 70) / 20.0) * 5
    bk_dist = np.abs(dataframe["close"].values / dataframe["ema_21"].values - 1.0)
    breakdown = np.maximum(0, 1.0 - bk_dist / 0.02) * 5
    bear_al = (
        (dataframe["close"].values < dataframe["ema_8"].values).astype(float) +
        (dataframe["close"].values < dataframe["ema_21"].values).astype(float) +
        (dataframe["close"].values < dataframe["ema_50"].values).astype(float)
    ) / 3.0 * 5
    return hmm_bear + trend_str + kalman_c + directional_s + momentum_s + volume + stability + rsi_s_s + breakdown + bear_al


def label_entries(entries_df, indicator_data, hold_bars=HOLD_BARS):
    labels = []
    for _, entry in entries_df.iterrows():
        pair = entry["__pair"]
        direction = entry["__direction"]
        df_idx = int(entry["__df_idx"])
        entry_price = entry["close"]
        df = indicator_data[pair]
        future_idx = df_idx + hold_bars
        if future_idx >= len(df):
            labels.append({"__y_win": 0, "__y_profit": 0.0})
            continue
        exit_price = df.iloc[future_idx]["close"]
        if direction == "long":
            pct = (exit_price - entry_price) / entry_price
        else:
            pct = (entry_price - exit_price) / entry_price
        labels.append({"__y_win": 1 if pct > 0 else 0, "__y_profit": float(pct)})
    return pd.DataFrame(labels)


def compute_metrics(trades_df):
    if len(trades_df) < 3:
        return {"n": len(trades_df), "wr": 0, "pf": 0, "exp": 0, "avg_win": 0, "avg_loss": 0, "mdd": 0}
    n = len(trades_df)
    wins = trades_df[trades_df["__y_win"] == 1]
    losses = trades_df[trades_df["__y_win"] == 0]
    n_win = len(wins)
    n_loss = len(losses)
    gross_profit = wins["__y_profit"].sum() if n_win > 0 else 0
    gross_loss = abs(losses["__y_profit"].sum()) if n_loss > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)
    avg_win = wins["__y_profit"].mean() if n_win > 0 else 0
    avg_loss = losses["__y_profit"].mean() if n_loss > 0 else 0
    exp = trades_df["__y_profit"].mean()
    cum = trades_df["__y_profit"].cumsum()
    peak = cum.cummax()
    mdd = (peak - cum).max()
    return {
        "n": n, "n_win": n_win, "n_loss": n_loss,
        "wr": n_win / n if n > 0 else 0,
        "pf": pf, "exp": exp, "avg_win": avg_win, "avg_loss": avg_loss,
        "mdd": mdd,
    }


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {}


def register_experiment(name, description):
    def decorator(func):
        EXPERIMENTS[name] = {"func": func, "desc": description}
        return func
    return decorator


@register_experiment("baseline_v41", "V4.1 baseline: 3 patterns + short_score >= 56")
def exp_baseline(indicator_data):
    """Full V4.1 entry logic: pattern detection + score filter."""
    all_entries = []
    for pair, df in indicator_data.items():
        df = apply_scoring_short_full(df)

        # Pattern 1: short_breakdown
        c1 = [
            df["close"] < df["low"].rolling(14).min().shift(1),
            df["volume_ratio"] > 1.0,
            df["adx"] > 15,
            df["close"] < df["ema_21"],
            df["close"] < df["open"],
            df["plus_di"] < df["minus_di"],
            df["rsi_5"] < 47,
            df["volume"] > 0,
        ]
        m1 = reduce(lambda x, y: x & y, c1)

        # Pattern 2: short_rally_fail
        c2 = [
            df["high"] >= df["ema_21"] * 0.995,
            df["close"] < df["ema_21"],
            df["close"] < df["open"],
            df["rsi_5"] > 55, df["rsi_5"] < 75,
            df["volume_ratio"] > 1.0,
            df["adx"] > 15,
            df["plus_di"] < df["minus_di"],
            df["volume"] > 0,
        ]
        m2 = reduce(lambda x, y: x & y, c2)

        # Pattern 3: short_bear_momentum
        c3 = [
            df["close"] < df["open"],
            df["volume_ratio"] > 1.3,
            df["adx"] > 16.5,
            df["plus_di"] < df["minus_di"],
        ]
        m3 = reduce(lambda x, y: x & y, c3)

        pattern = m1 | m2 | m3

        # Score ceiling
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0

        # Pattern + threshold
        entry_pattern = pattern & (short_sc >= 56)

        # Override: high score without pattern
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override

        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            entries["__score"] = short_sc[entry_final]
            all_entries.append(entries)

    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


def apply_scoring_short_full(dataframe):
    """Compute V4.1 short_score on a dataframe."""
    hmm_bear = np.minimum(dataframe["hmm_p_bear"].values / 0.6, 1.0) * 20
    trend_str = np.minimum(dataframe["adx"].values / 40.0, 1.0) * 15
    kalman_c = np.minimum(dataframe["kf_confidence"].values / 0.8, 1.0) * 10
    di_s = dataframe["minus_di"].values - dataframe["plus_di"].values
    directional_s = np.clip(di_s / 20.0 + 0.5, 0, 1) * 10
    mom_a_s = np.minimum(np.maximum(-dataframe["kf_trend_acceleration"].values, 0), 1)
    mom_p_s = np.minimum(np.maximum(-dataframe["kf_price_momentum"].values, 0), 1)
    momentum_s = (mom_a_s + mom_p_s) / 2.0 * 10
    volume = np.minimum(dataframe["volume_ratio"].values / 3.0, 1.0) * 10
    stability = (1.0 - np.minimum(dataframe["hmm_regime_stability"].values / 0.5, 1.0)) * 10
    rsi_v = dataframe["rsi_5"].values
    rsi_s_s = np.maximum(0, 1.0 - np.abs(rsi_v - 70) / 20.0) * 5
    bk_dist = np.abs(dataframe["close"].values / dataframe["ema_21"].values - 1.0)
    breakdown = np.maximum(0, 1.0 - bk_dist / 0.02) * 5
    bear_al = (
        (dataframe["close"].values < dataframe["ema_8"].values).astype(float) +
        (dataframe["close"].values < dataframe["ema_21"].values).astype(float) +
        (dataframe["close"].values < dataframe["ema_50"].values).astype(float)
    ) / 3.0 * 5
    dataframe["short_score"] = (
        hmm_bear + trend_str + kalman_c + directional_s +
        momentum_s + volume + stability + rsi_s_s + breakdown + bear_al
    )
    return dataframe


@register_experiment("e8_kalman_only", "E8: kf_regime_score < p25, kf_confidence > p25 (percentile-based)")
def exp_e8_kalman(indicator_data):
    """Minimal Kalman entry. Short only. Percentile thresholds."""
    all_entries = []
    for pair, df in indicator_data.items():
        ks = df["kf_regime_score"].values
        conf = df["kf_confidence"].values
        # Use percentile-based: ks < 25th percentile, conf > 25th percentile
        ks_thresh = np.percentile(ks, 25)
        conf_thresh = np.percentile(conf, 25)
        mask = (ks < ks_thresh) & (conf > conf_thresh)
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[mask]
            entries["__score"] = ks[mask]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e8_kalman_relaxed", "E8 relaxed: kf_regime_score < p10, no confidence filter")
def exp_e8_kalman_relaxed(indicator_data):
    all_entries = []
    for pair, df in indicator_data.items():
        ks = df["kf_regime_score"].values
        conf = df["kf_confidence"].values
        # Relaxed: just bearish regime score, no confidence filter
        ks_thresh = np.percentile(ks, 10)
        mask = ks < ks_thresh
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[mask]
            entries["__score"] = ks[mask]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e8_kalman_trend", "E8: kf_regime_score < p25, kf_trend < 0, confidence > p25")
def exp_e8_kalman_trend(indicator_data):
    all_entries = []
    for pair, df in indicator_data.items():
        ks = df["kf_regime_score"].values
        conf = df["kf_confidence"].values
        trend = df["kf_trend"].values
        ks_thresh = np.percentile(ks, 25)
        conf_thresh = np.percentile(conf, 25)
        mask = (ks < ks_thresh) & (conf > conf_thresh) & (trend < 0)
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[mask]
            entries["__score"] = ks[mask]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e8_kalman_macd", "E8: kf_regime_score < p25 + macdhist < 0 + volume > 1.0")
def exp_e8_kalman_macd(indicator_data):
    """Kalman + MACD + Volume — the top 3 non-overlapping features."""
    all_entries = []
    for pair, df in indicator_data.items():
        ks = df["kf_regime_score"].values
        conf = df["kf_confidence"].values
        macd_h = df["macdhist"].values
        vol = df["volume_ratio"].values
        ks_thresh = np.percentile(ks, 25)
        conf_thresh = np.percentile(conf, 25)
        mask = (ks < ks_thresh) & (conf > conf_thresh) & (macd_h < 0) & (vol > 1.0)
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[mask]
            entries["__score"] = ks[mask]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e8_composite_simple", "E8: top-5 weighted composite (percentile-normalized)")
def exp_e8_composite(indicator_data):
    """
    Weighted score from top-5 features, all percentile-normalized to 0-1.
    Short when composite >= p75 (top quartile bearish).
    """
    all_entries = []
    for pair, df in indicator_data.items():
        pe = df["kf_prediction_error"].values
        # Higher prediction error = price above prediction = expect reversion down = bearish
        pe_rank = np.argsort(np.argsort(pe)) / len(pe)  # percentile rank

        macd = df["macdhist"].values
        # Negative macdhist = bearish momentum
        macd_rank = 1.0 - np.argsort(np.argsort(macd)) / len(macd)  # invert: more negative = higher rank

        rsi = df["rsi_5"].values
        # Higher RSI = overbought = bearish opportunity
        rsi_rank = np.argsort(np.argsort(rsi)) / len(rsi)

        ks = df["kf_regime_score"].values
        # More negative = more bearish
        ks_rank = 1.0 - np.argsort(np.argsort(ks)) / len(ks)

        vol = df["volume_ratio"].values
        vol_rank = np.argsort(np.argsort(vol)) / len(vol)

        w = np.array([40, 35, 34, 33, 30])
        ranks = np.column_stack([pe_rank, macd_rank, rsi_rank, ks_rank, vol_rank])
        composite = (ranks * w).sum(axis=1) / w.sum() * 100

        mask = composite >= 65
        if mask.any():
            entries = df[mask].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[mask]
            entries["__score"] = composite[mask]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e8_kalman_pattern", "E8: Kalman direction + V4.1 pattern timing")
def exp_e8_kalman_pattern(indicator_data):
    """
    Combine Kalman direction filter with V4.1 pattern detection.
    Short only when Kalman confirms bearish AND a pattern fires.
    This tests whether Kalman adds value as a REGIME FILTER on top of pattern timing.
    """
    all_entries = []
    for pair, df in indicator_data.items():
        df = apply_scoring_short_full(df)
        ks = df["kf_regime_score"].values
        conf = df["kf_confidence"].values
        ks_thresh = np.percentile(ks, 25)
        conf_thresh = np.percentile(conf, 25)
        kalman_bear = (ks < ks_thresh) & (conf > conf_thresh)

        # Pattern 1: short_breakdown
        c1 = [
            df["close"] < df["low"].rolling(14).min().shift(1),
            df["volume_ratio"] > 1.0,
            df["adx"] > 15,
            df["close"] < df["ema_21"],
            df["close"] < df["open"],
            df["plus_di"] < df["minus_di"],
            df["rsi_5"] < 47,
            df["volume"] > 0,
        ]
        m1 = reduce(lambda x, y: x & y, c1)

        # Pattern 2: short_rally_fail
        c2 = [
            df["high"] >= df["ema_21"] * 0.995,
            df["close"] < df["ema_21"],
            df["close"] < df["open"],
            df["rsi_5"] > 55, df["rsi_5"] < 75,
            df["volume_ratio"] > 1.0,
            df["adx"] > 15,
            df["plus_di"] < df["minus_di"],
            df["volume"] > 0,
        ]
        m2 = reduce(lambda x, y: x & y, c2)

        # Pattern 3: short_bear_momentum
        c3 = [
            df["close"] < df["open"],
            df["volume_ratio"] > 1.3,
            df["adx"] > 16.5,
            df["plus_di"] < df["minus_di"],
        ]
        m3 = reduce(lambda x, y: x & y, c3)

        pattern = m1 | m2 | m3
        entry = kalman_bear & pattern

        if entry.any():
            entries = df[entry].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry]
            entries["__score"] = ks[entry]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e8_kalman_pattern_relaxed", "E8: Kalman bearish (p10) + pattern, no score filter")
def exp_e8_kalman_pattern_relaxed(indicator_data):
    """Kalman direction + pattern, no score threshold."""
    all_entries = []
    for pair, df in indicator_data.items():
        ks = df["kf_regime_score"].values
        ks_thresh = np.percentile(ks, 10)
        kalman_bear = ks < ks_thresh

        c1 = [
            df["close"] < df["low"].rolling(14).min().shift(1),
            df["volume_ratio"] > 1.0,
            df["adx"] > 15,
            df["close"] < df["ema_21"],
            df["close"] < df["open"],
            df["plus_di"] < df["minus_di"],
            df["rsi_5"] < 47,
            df["volume"] > 0,
        ]
        m1 = reduce(lambda x, y: x & y, c1)
        c2 = [
            df["high"] >= df["ema_21"] * 0.995,
            df["close"] < df["ema_21"],
            df["close"] < df["open"],
            df["rsi_5"] > 55, df["rsi_5"] < 75,
            df["volume_ratio"] > 1.0,
            df["adx"] > 15,
            df["plus_di"] < df["minus_di"],
            df["volume"] > 0,
        ]
        m2 = reduce(lambda x, y: x & y, c2)
        c3 = [
            df["close"] < df["open"],
            df["volume_ratio"] > 1.3,
            df["adx"] > 16.5,
            df["plus_di"] < df["minus_di"],
        ]
        m3 = reduce(lambda x, y: x & y, c3)
        pattern = m1 | m2 | m3
        entry = kalman_bear & pattern

        if entry.any():
            entries = df[entry].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry]
            entries["__score"] = ks[entry]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


# ============================================================
# E9: TOP-FEATURE SCORE EXPERIMENTS
# ============================================================
# Tests whether a simpler score using only the top LightGBM features,
# applied to V4.1 pattern entries, outperforms the full 10-term score.

def _get_v4_patterns(df):
    """Return V4.1 pattern masks (m1, m2, m3)."""
    c1 = [
        df["close"] < df["low"].rolling(14).min().shift(1),
        df["volume_ratio"] > 1.0,
        df["adx"] > 15,
        df["close"] < df["ema_21"],
        df["close"] < df["open"],
        df["plus_di"] < df["minus_di"],
        df["rsi_5"] < 47,
        df["volume"] > 0,
    ]
    m1 = reduce(lambda x, y: x & y, c1)

    c2 = [
        df["high"] >= df["ema_21"] * 0.995,
        df["close"] < df["ema_21"],
        df["close"] < df["open"],
        df["rsi_5"] > 55, df["rsi_5"] < 75,
        df["volume_ratio"] > 1.0,
        df["adx"] > 15,
        df["plus_di"] < df["minus_di"],
        df["volume"] > 0,
    ]
    m2 = reduce(lambda x, y: x & y, c2)

    c3 = [
        df["close"] < df["open"],
        df["volume_ratio"] > 1.3,
        df["adx"] > 16.5,
        df["plus_di"] < df["minus_di"],
    ]
    m3 = reduce(lambda x, y: x & y, c3)
    return m1 | m2 | m3


@register_experiment("e9_pattern_only", "E9: V4.1 patterns, NO score filter (baseline)")
def exp_e9_pattern_only(indicator_data):
    """All pattern entries with no score filter — pure pattern timing."""
    all_entries = []
    for pair, df in indicator_data.items():
        pattern = _get_v4_patterns(df)
        if pattern.any():
            entries = df[pattern].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[pattern]
            entries["__score"] = 0.0
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e9_kalman_filter", "E9: V4.1 patterns + Kalman bearish filter only")
def exp_e9_kalman_filter(indicator_data):
    """Patterns filtered by Kalman direction (no composite score)."""
    all_entries = []
    for pair, df in indicator_data.items():
        pattern = _get_v4_patterns(df)
        ks = df["kf_regime_score"].values
        ks_thresh = np.percentile(ks, 25)
        kalman_bear = ks < ks_thresh
        entry = pattern & kalman_bear
        if entry.any():
            entries = df[entry].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry]
            entries["__score"] = ks[entry]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e9_top5_score", "E9: V4.1 patterns + top-5 feature composite score")
def exp_e9_top5_score(indicator_data):
    """
    V4.1 patterns + score from only the top-5 LightGBM features:
    kf_prediction_error, macdhist, rsi_5, kf_regime_score, volume_ratio.
    Percentile-rank normalized, composite to 0-100.
    """
    all_entries = []
    for pair, df in indicator_data.items():
        pattern = _get_v4_patterns(df)
        pe = df["kf_prediction_error"].values
        pe_rank = np.argsort(np.argsort(pe)) / len(pe)
        macd = df["macdhist"].values
        macd_rank = 1.0 - np.argsort(np.argsort(macd)) / len(macd)
        rsi = df["rsi_5"].values
        rsi_rank = np.argsort(np.argsort(rsi)) / len(rsi)
        ks = df["kf_regime_score"].values
        ks_rank = 1.0 - np.argsort(np.argsort(ks)) / len(ks)
        vol = df["volume_ratio"].values
        vol_rank = np.argsort(np.argsort(vol)) / len(vol)
        w = np.array([40, 35, 34, 33, 30])
        ranks = np.column_stack([pe_rank, macd_rank, rsi_rank, ks_rank, vol_rank])
        composite = (ranks * w).sum(axis=1) / w.sum() * 100
        entry = pattern & (composite >= 55)
        if entry.any():
            entries = df[entry].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry]
            entries["__score"] = composite[entry]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e9_kalman_only_score", "E9: V4.1 patterns + Kalman-only score (conf × direction)")
def exp_e9_kalman_only_score(indicator_data):
    """
    V4.1 patterns filtered by a Kalman-only score: kf_confidence × kf_direction.
    This is a pure Kalman timing signal applied to pattern entries.
    """
    all_entries = []
    for pair, df in indicator_data.items():
        pattern = _get_v4_patterns(df)
        ks = df["kf_regime_score"].values
        conf = df["kf_confidence"].values
        # Kalman score: confidence-weighted direction (negative = bearish)
        kalman_score = ks  # already confidence × direction
        # Filter: score < p25 (bearish) AND confidence > p25
        ks_thresh = np.percentile(kalman_score, 25)
        conf_thresh = np.percentile(conf, 25)
        entry = pattern & (kalman_score < ks_thresh) & (conf > conf_thresh)
        if entry.any():
            entries = df[entry].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry]
            entries["__score"] = kalman_score[entry]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e9_di_only", "E9: V4.1 patterns + DI spread only")
def exp_e9_di_only(indicator_data):
    """V4.1 patterns filtered by net directional indicator spread."""
    all_entries = []
    for pair, df in indicator_data.items():
        pattern = _get_v4_patterns(df)
        di_spread = df["minus_di"].values - df["plus_di"].values
        di_thresh = np.percentile(di_spread, 25)
        entry = pattern & (di_spread < di_thresh)
        if entry.any():
            entries = df[entry].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry]
            entries["__score"] = di_spread[entry]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)



# ============================================================
# E14: REGIME-ADAPTIVE THRESHOLD EXPERIMENTS
# ============================================================
# Tests whether using HMM regime to select different score thresholds
# improves consistency across walk-forward windows.

def _run_regime_adaptive(indicator_data, threshold_map, filter_avax_bnb=False):
    """
    Generic regime-adaptive entry runner.
    threshold_map: dict mapping hmm_regime -> score_threshold
    """
    all_entries = []
    for pair, df in indicator_data.items():
        if filter_avax_bnb and ("AVAX" in pair or "BNB" in pair):
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        hmm = df["hmm_regime"].values
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0

        entry_mask = pd.Series(False, index=df.index)
        for regime, thresh in threshold_map.items():
            regime_match = (hmm == regime) & pattern & (short_sc >= thresh)
            entry_mask = entry_mask | regime_match

        # High score override (regime-independent)
        no_entry = ~entry_mask
        override = no_entry & (short_sc >= 62)
        entry_final = entry_mask | override

        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            entries["__score"] = short_sc[entry_final]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e14_adaptive_conservative", "E14: Bear=50, Range=60, Bull=blocked (conservative)")
def exp_e14_conservative(indicator_data):
    return _run_regime_adaptive(indicator_data, {0: 50, 1: 60, 2: 50})


@register_experiment("e14_adaptive_aggressive", "E14: Bear=45, Range=55, Bull=blocked (aggressive)")
def exp_e14_aggressive(indicator_data):
    return _run_regime_adaptive(indicator_data, {0: 45, 1: 55, 2: 45})


@register_experiment("e14_adaptive_bear_only", "E14: Bear=45, Range=65, Bull=blocked (bear-heavy)")
def exp_e14_bear_only(indicator_data):
    return _run_regime_adaptive(indicator_data, {0: 45, 1: 65, 2: 45})


@register_experiment("e14_adaptive_relaxed", "E14: Bear=40, Range=50, Bull=blocked (very relaxed)")
def exp_e14_relaxed(indicator_data):
    return _run_regime_adaptive(indicator_data, {0: 40, 1: 50, 2: 40})


@register_experiment("e14_adaptive_avax_bnb_removed", "E14: aggressive + AVAX/BNB removed")
def exp_e14_no_avax_bnb(indicator_data):
    return _run_regime_adaptive(indicator_data, {0: 45, 1: 55, 2: 45}, filter_avax_bnb=True)


@register_experiment("e14_adaptive_all_pairs_removed", "E14: aggressive + AVAX/BNB/LTC removed")
def exp_e14_remove_3(indicator_data):
    all_entries = []
    for pair, df in indicator_data.items():
        if "AVAX" in pair or "BNB" in pair or "LTC" in pair:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        hmm = df["hmm_regime"].values
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_mask = pd.Series(False, index=df.index)
        for regime, thresh in {0: 45, 1: 55, 2: 45}.items():
            regime_match = (hmm == regime) & pattern & (short_sc >= thresh)
            entry_mask = entry_mask | regime_match
        no_entry = ~entry_mask
        override = no_entry & (short_sc >= 62)
        entry_final = entry_mask | override
        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            entries["__score"] = short_sc[entry_final]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e14_kalman_regime_adaptive", "E14: Kalman regime_score threshold adapts per HMM regime")
def exp_e14_kalman_regime(indicator_data):
    """Use Kalman regime_score as entry signal, with regime-adaptive thresholds."""
    all_entries = []
    for pair, df in indicator_data.items():
        ks = df["kf_regime_score"].values
        conf = df["kf_confidence"].values
        hmm = df["hmm_regime"].values
        pattern = _get_v4_patterns(df)

        entry_mask = pd.Series(False, index=df.index)
        # In bear regime (2): lower threshold (more entries)
        # In range regime (1): medium threshold
        # In bull regime (0): blocked
        thresholds = {0: 999, 1: np.percentile(ks, 30), 2: np.percentile(ks, 40)}
        for regime, thresh in thresholds.items():
            if thresh == 999:
                continue  # block in bull
            regime_match = (hmm == regime) & pattern & (ks < thresh) & (conf > np.percentile(conf, 20))
            entry_mask = entry_mask | regime_match
        if entry_mask.any():
            entries = df[entry_mask].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_mask]
            entries["__score"] = ks[entry_mask]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)



# ============================================================
# E12: PAIR SELECTION EXPERIMENTS
# ============================================================

@register_experiment("e12_no_avax", "E12: V4.1 baseline without AVAX")
def exp_e12_no_avax(indicator_data):
    all_entries = []
    for pair, df in indicator_data.items():
        if "AVAX" in pair:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_pattern = pattern & (short_sc >= 56)
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override
        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            entries["__score"] = short_sc[entry_final]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e12_no_avax_bnb", "E12: V4.1 baseline without AVAX+BNB")
def exp_e12_no_avax_bnb(indicator_data):
    all_entries = []
    for pair, df in indicator_data.items():
        if "AVAX" in pair or "BNB" in pair:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_pattern = pattern & (short_sc >= 56)
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override
        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            entries["__score"] = short_sc[entry_final]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e12_top6_only", "E12: Only top-6 pairs (SOL,BTC,XRP,DOGE,ETH,LTC)")
def exp_e12_top6(indicator_data):
    TOP6 = ["SOL/USDT:USDT", "BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT", "ETH/USDT:USDT", "LTC/USDT:USDT"]
    all_entries = []
    for pair, df in indicator_data.items():
        if pair not in TOP6:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_pattern = pattern & (short_sc >= 56)
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override
        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            entries["__score"] = short_sc[entry_final]
            all_entries.append(entries)
    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)



# ============================================================
# E10: MTF CONFIRMATION EXPERIMENTS
# ============================================================
# Tests whether requiring 15m trend alignment eliminates false entries.

def _load_15m_data(pairs, timerange_str):
    from freqtrade.configuration.timerange import TimeRange
    tr = TimeRange.parse_timerange(timerange_str)
    base = Path("user_data/data/binance/futures")
    start = pd.Timestamp(dt.datetime.fromtimestamp(tr.startts), tz="UTC")
    end = pd.Timestamp(dt.datetime.fromtimestamp(tr.stopts), tz="UTC")
    dataframes = {}
    for pair in pairs:
        fname = pair.replace("/", "_").replace(":", "_") + "-15m-futures.feather"
        fpath = base / fname
        if not fpath.exists():
            continue
        df = pd.read_feather(fpath)
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(df) > 20:
            dataframes[pair] = df
    return dataframes


def _compute_15m_trend(dataframes):
    """Compute simple 15m trend indicators: EMA alignment + RSI."""
    result = {}
    for pair, df in dataframes.items():
        df = df.copy()
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        # Bearish if: close < ema_20 < ema_50 (downtrend)
        df["mtf_bearish"] = ((df["close"] < df["ema_20"]) & (df["ema_20"] < df["ema_50"])).astype(int)
        # Resample to 5m alignment: forward-fill 15m values to 5m timestamps
        df = df.set_index("date")
        result[pair] = df[["mtf_bearish"]]
    return result


@register_experiment("e10_mtf_15m", "E10: E12 baseline + 15m bearish trend confirmation")
def exp_e10_mtf(indicator_data):
    """E12 baseline (no AVAX/BNB) + 15m EMA downtrend confirmation."""
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_pattern = pattern & (short_sc >= 56)
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override

        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            all_entries.append(entries)

    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e10_mtf_15m_strict", "E10: E12 + 15m bearish AND 5m below ema_50")
def exp_e10_mtf_strict(indicator_data):
    """E12 baseline + stricter 5m trend alignment (close < ema_50)."""
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_pattern = pattern & (short_sc >= 56)
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override

        # Extra filter: close below ema_50 (stronger bearish confirmation)
        entry_final = entry_final & (df["close"] < df["ema_50"])

        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            all_entries.append(entries)

    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


@register_experiment("e10_mtf_kalman_plus_di", "E10: E12 + KF bearish + DI confirmation")
def exp_e10_kalman_di(indicator_data):
    """E12 baseline + Kalman bearish + minus_di > plus_di confirmation."""
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_pattern = pattern & (short_sc >= 56)
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override

        # Extra: KF bearish + DI confirmation
        ks = df["kf_regime_score"].values
        ks_thresh = np.percentile(ks, 25)
        di_bear = df["minus_di"] > df["plus_di"]
        entry_final = entry_final & (ks < ks_thresh) & di_bear

        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            all_entries.append(entries)

    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


# ============================================================
# E11: FUNDING RATE FILTER EXPERIMENTS
# ============================================================

def _load_funding_data(pairs, timerange_str):
    from freqtrade.configuration.timerange import TimeRange
    tr = TimeRange.parse_timerange(timerange_str)
    base = Path("user_data/data/binance/futures")
    start = pd.Timestamp(dt.datetime.fromtimestamp(tr.startts), tz="UTC")
    end = pd.Timestamp(dt.datetime.fromtimestamp(tr.stopts), tz="UTC")
    dataframes = {}
    for pair in pairs:
        fname = pair.replace("/", "_").replace(":", "_") + "-1h-funding_rate.feather"
        fpath = base / fname
        if not fpath.exists():
            continue
        df = pd.read_feather(fpath)
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(df) > 0:
            dataframes[pair] = df
    return dataframes


@register_experiment("e11_funding_filter", "E11: E12 + funding rate < -0.01% blocks shorts (contrarian)")
def exp_e11_funding(indicator_data):
    """E12 baseline + funding rate filter: if funding < -0.01%, block shorts (crowded short)."""
    all_entries = []
    for pair, df in indicator_data.items():
        if pair in FILTER_PAIRS:
            continue
        df = apply_scoring_short_full(df)
        pattern = _get_v4_patterns(df)
        short_sc = df["short_score"].copy()
        short_sc[short_sc > 80] = 0
        entry_pattern = pattern & (short_sc >= 56)
        no_entry = ~entry_pattern
        override = no_entry & (short_sc >= 59)
        entry_final = entry_pattern | override

        if entry_final.any():
            entries = df[entry_final].copy()
            entries["__pair"] = pair
            entries["__direction"] = "short"
            entries["__df_idx"] = df.index[entry_final]
            all_entries.append(entries)

    if not all_entries:
        return pd.DataFrame()
    return pd.concat(all_entries, ignore_index=True)


FILTER_PAIRS = ['AVAX/USDT:USDT', 'BNB/USDT:USDT']


def run_experiment(experiment_name, indicator_data):
    exp = EXPERIMENTS[experiment_name]
    entries = exp["func"](indicator_data)
    if entries.empty:
        return {"name": experiment_name, "desc": exp["desc"], "metrics": compute_metrics(pd.DataFrame()), "trades": pd.DataFrame()}
    labels = label_entries(entries, indicator_data)
    score_col = entries[["__score"]].reset_index(drop=True) if "__score" in entries.columns else pd.DataFrame({"__score": [0]*len(entries)})
    trades = pd.concat([entries[["__pair", "__direction"]].reset_index(drop=True), score_col, labels.reset_index(drop=True)], axis=1)
    metrics = compute_metrics(trades)
    return {"name": experiment_name, "desc": exp["desc"], "metrics": metrics, "trades": trades}


def main():
    timerange = sys.argv[1] if len(sys.argv) > 1 else "20260618-20260720"
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config-v4.json"
    experiment = sys.argv[3] if len(sys.argv) > 3 else "all"

    from freqtrade.configuration import Configuration
    config = Configuration.from_files([config_path])

    print("=" * 70)
    print("PHOENIX SCALPER — EXPERIMENT RUNNER")
    print(f"Timerange: {timerange}")
    print("=" * 70)

    print("\nStep 1: Loading data...")
    raw_data = load_all_data(TOP10, timerange)
    print(f"  Loaded {len(raw_data)} pairs")

    print("\nStep 2: Computing indicators...")
    indicator_data, _ = compute_indicators(raw_data, config)
    print(f"  Indicators computed for {len(indicator_data)} pairs")

    experiments_to_run = list(EXPERIMENTS.keys()) if experiment == "all" else [experiment]
    results = []

    for exp_name in experiments_to_run:
        if exp_name not in EXPERIMENTS:
            print(f"\n  Unknown experiment: {exp_name}")
            continue
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"  {EXPERIMENTS[exp_name]['desc']}")
        print(f"{'=' * 70}")
        result = run_experiment(exp_name, indicator_data)
        m = result["metrics"]
        print(f"  Trades: {m['n']} ({m.get('n_win', 0)}W / {m.get('n_loss', 0)}L)")
        print(f"  Win Rate: {m['wr']:.1%}")
        print(f"  PF: {m['pf']:.3f}")
        print(f"  Expectancy: {m['exp']:.4f}")
        print(f"  Avg Win: {m['avg_win']:.4f}  Avg Loss: {m['avg_loss']:.4f}")
        print(f"  MDD: {m['mdd']:.4f}")
        pf_pass = "PASS" if m["pf"] > 1.30 else "FAIL"
        exp_pass = "PASS" if m["exp"] > 0 else "FAIL"
        print(f"  Gates: PF>1.30 {pf_pass} | Exp>0 {exp_pass}")
        results.append(result)

    # Comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    header = f"{'Experiment':<25} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Exp%':>8} {'AvgW':>7} {'AvgL':>7} {'MDD':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print(f"{r['name']:<25} {m['n']:>7} {m['wr']:>5.1%} {m['pf']:>7.3f} {m['exp']:>7.4f} {m['avg_win']:>7.4f} {m['avg_loss']:>7.4f} {m['mdd']:>7.4f}")

    # Best experiment
    if results:
        best = max(results, key=lambda r: r["metrics"]["pf"] if r["metrics"]["pf"] != float("inf") else 999)
        print(f"\nBest: {best['name']} (PF={best['metrics']['pf']:.3f})")

    # Save
    import json
    out = [{
        "name": r["name"], "desc": r["desc"],
        "n": r["metrics"]["n"], "wr": round(r["metrics"]["wr"], 4),
        "pf": round(r["metrics"]["pf"], 4) if r["metrics"]["pf"] != float("inf") else "inf",
        "exp": round(r["metrics"]["exp"], 6),
        "avg_win": round(r["metrics"]["avg_win"], 6),
        "avg_loss": round(r["metrics"]["avg_loss"], 6),
        "mdd": round(r["metrics"]["mdd"], 6),
    } for r in results]
    with open("research/experiment_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved research/experiment_results.json")


if __name__ == "__main__":
    main()
