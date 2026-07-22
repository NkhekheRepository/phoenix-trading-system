"""
E6 — LightGBM feature importance for PhoenixScalperV4_1.

Bypasses the freqtrade backtesting API entirely:
1. Loads raw 5m OHLCV data for top-10 pairs
2. Runs populate_indicators() to compute all features
3. Runs populate_entry_trend() to identify entry signals
4. Labels each entry as win/loss by forward 12-bar (1h) price action
5. Trains LightGBM classifier and reports feature importance by family

Read-only w.r.t. the strategy.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

TOP10 = [
    "SOL/USDT:USDT", "BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "BNB/USDT:USDT", "ETH/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT",
]

LEAK_COLS = {
    "date", "open", "high", "low", "close", "volume", "enter_long",
    "enter_short", "exit_long", "exit_short", "enter_tag", "exit_tag",
    "do_predict", "kf_S",
}

FAMILIES = {
    "hmm_bull": "HMM", "hmm_bear": "HMM", "hmm_p_bull": "HMM",
    "hmm_p_bear": "HMM", "hmm_p_range": "HMM", "hmm_regime_stability": "HMM",
    "hmm_vol_regime": "HMM", "hmm_trend_strength": "HMM", "hmm_regime": "HMM",
    "kf_confidence": "Kalman", "kf_trend": "Kalman",
    "kf_trend_acceleration": "Kalman", "kf_price_momentum": "Kalman",
    "kf_prediction": "Kalman", "kf_direction": "Kalman",
    "kf_innovation": "Kalman", "kf_vol_of_trend": "Kalman",
    "kf_atr_ratio": "Kalman", "kf_regime_score": "Kalman", "kf_price": "Kalman",
    "rsi": "RSI", "adx": "ADX", "plus_di": "DI", "minus_di": "DI",
    "macd": "MACD", "macdhist": "MACD", "macdsignal": "MACD",
    "bb": "BB", "atr": "ATR", "atr_pct": "ATR", "volume_ratio": "Volume",
    "obv": "OBV", "vwap": "VWAP",
}


def load_data(pairs, timerange_str):
    """Load 5m OHLCV futures data directly from feather files."""
    from freqtrade.configuration.timerange import TimeRange
    import datetime as dt

    tr = TimeRange.parse_timerange(timerange_str)
    base = Path("user_data/data/binance/futures")
    dataframes = {}
    for pair in pairs:
        # feather files use underscores: SOL_USDT_USDT-5m-futures.feather
        fname = pair.replace("/", "_").replace(":", "_") + "-5m-futures.feather"
        fpath = base / fname
        if not fpath.exists():
            print(f"  SKIP {pair}: file not found ({fname})")
            continue
        df = pd.read_feather(fpath)
        # Filter by timerange
        start = pd.Timestamp(dt.datetime.fromtimestamp(tr.startts)).tz_localize("UTC")
        end = pd.Timestamp(dt.datetime.fromtimestamp(tr.stopts)).tz_localize("UTC")
        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(df) > 60:
            dataframes[pair] = df
            print(f"  {pair}: {len(df)} candles")
        else:
            print(f"  SKIP {pair}: {len(df)} candles after filter")
    return dataframes


def main():
    timerange = sys.argv[1] if len(sys.argv) > 1 else "20260618-20260720"
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config-v4.json"

    from freqtrade.configuration import Configuration
    config = Configuration.from_files([config_path])
    config["strategy"] = "PhoenixScalperV4_1"
    config["strategy_path"] = "strategies"
    config["exchange"]["pair_whitelist"] = TOP10
    config["pairlists"] = [{"method": "StaticPairList"}]

    # Step 1: Load raw data
    print("Step 1: Loading 5m OHLCV futures data...")
    raw_data = load_data(TOP10, timerange)
    if not raw_data:
        print("No data loaded.")
        return

    # Step 2: Resolve strategy
    from freqtrade.resolvers import StrategyResolver
    strategy = StrategyResolver.load_strategy(config)

    # Step 3: Populate indicators + entries per pair (compute once, store)
    print("\nStep 2-3: Populating indicators and entry signals...")
    indicator_data = {}  # pair -> fully-indicator-enriched dataframe
    all_entries = []

    for pair in TOP10:
        if pair not in raw_data:
            continue
        df = raw_data[pair].copy()
        df = strategy.populate_indicators(df, {"pair": pair})
        df = strategy.populate_entry_trend(df, {"pair": pair})
        indicator_data[pair] = df

        long_mask = df.get("enter_long", pd.Series(dtype=int)) == 1 if "enter_long" in df.columns else pd.Series([False]*len(df))
        short_mask = df.get("enter_short", pd.Series(dtype=int)) == 1 if "enter_short" in df.columns else pd.Series([False]*len(df))
        n_long = long_mask.sum()
        n_short = short_mask.sum()

        # Extract entry rows
        for direction, mask in [("long", long_mask), ("short", short_mask)]:
            if mask.any():
                entries = df[mask].copy()
                entries["__pair"] = pair
                entries["__direction"] = direction
                # Store original dataframe index for forward-labeling
                entries["__df_idx"] = df.index[mask]
                all_entries.append(entries)

        print(f"  {pair}: {n_long} long + {n_short} short = {n_long + n_short} entries")

    if not all_entries:
        print("No entry signals found.")
        return

    all_entries = pd.concat(all_entries, ignore_index=True)
    print(f"\nTotal entries: {len(all_entries)}")

    # Step 4: Label entries by forward 12-bar price action (1h hold at 5m)
    print("\nStep 4: Labeling by forward price action (12 bars = 1h)...")
    HOLD = 12
    wins, losses = 0, 0

    labels = []
    for _, entry in all_entries.iterrows():
        pair = entry["__pair"]
        direction = entry["__direction"]
        df_idx = int(entry["__df_idx"])
        entry_price = entry["close"]

        df = indicator_data[pair]
        future_idx = df_idx + HOLD
        if future_idx >= len(df):
            labels.append({"__y_win": 0, "__y_profit": 0.0})
            continue

        exit_price = df.iloc[future_idx]["close"]
        if direction == "long":
            pct = (exit_price - entry_price) / entry_price
        else:
            pct = (entry_price - exit_price) / entry_price

        win = 1 if pct > 0 else 0
        wins += win
        losses += (1 - win)
        labels.append({"__y_win": win, "__y_profit": float(pct)})

    labels_df = pd.DataFrame(labels)
    all_entries = pd.concat([all_entries.reset_index(drop=True), labels_df], axis=1)
    print(f"  {wins} wins ({100*wins/(wins+losses):.1f}%), {losses} losses")

    # Step 5: Feature columns
    feat_cols = [c for c in all_entries.columns
                 if c not in LEAK_COLS and not c.startswith("__")
                 and pd.api.types.is_numeric_dtype(all_entries[c])]
    print(f"\nStep 5: {len(feat_cols)} numeric features")

    X = all_entries[feat_cols].fillna(0.0)
    y = all_entries["__y_win"].astype(int)

    # Step 6: Train LightGBM
    print("\nStep 6: Training LightGBM...")
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import roc_auc_score, classification_report

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"  Train: {len(Xtr)} ({ytr.sum()} wins), Test: {len(Xte)} ({yte.sum()} wins)")

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr)

    yte_proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, yte_proba) if len(set(yte)) > 1 else float("nan")
    print(f"  Test AUC: {auc:.3f}")

    if len(set(y)) > 1 and y.sum() >= 5:
        cv_scores = cross_val_score(model, X, y, cv=min(5, int(y.sum())), scoring="roc_auc")
        print(f"  CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    print("\n  Classification report:")
    print(classification_report(yte, model.predict(Xte), zero_division=0))

    # Step 7: Feature importance
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n=== TOP 25 FEATURES (split gain) ===")
    print(imp.head(25).to_string())

    # Step 8: Family importance
    fam_imp = {}
    for c, v in imp.items():
        fam = "OTHER"
        for prefix, fname in FAMILIES.items():
            if c.startswith(prefix):
                fam = fname
                break
        fam_imp[fam] = fam_imp.get(fam, 0.0) + float(v)
    fam_imp = dict(sorted(fam_imp.items(), key=lambda x: -x[1]))
    total = sum(fam_imp.values()) or 1.0

    print("\n=== FEATURE FAMILY IMPORTANCE (% of total) ===")
    for f, v in fam_imp.items():
        pct = 100 * v / total
        bar = "#" * int(pct / 2)
        print(f"  {f:10s}: {pct:5.1f}%  {bar}")

    hmm_pct = fam_imp.get("HMM", 0) / total * 100
    kalman_pct = fam_imp.get("Kalman", 0) / total * 100
    print(f"\n  HMM:    {hmm_pct:.1f}%")
    print(f"  Kalman: {kalman_pct:.1f}%")
    print(f"  ML signal total: {hmm_pct + kalman_pct:.1f}%")

    # Save
    import json
    out = {
        "n_entries": int(len(all_entries)),
        "n_wins": int(y.sum()),
        "auc": float(auc),
        "top_features": {k: int(v) for k, v in imp.head(25).items()},
        "family_importance": {k: round(v / total * 100, 1) for k, v in fam_imp.items()},
    }
    with open("research/feature_importance_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved research/feature_importance_result.json")


if __name__ == "__main__":
    main()
