import sys, os, numpy as np, pandas as pd, logging, joblib, time, gc, json as _json, zipfile
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import lightgbm as lgb

DATA_DIR = "/freqtrade/user_data/data"
BACKTEST_DIR = "/freqtrade/user_data/backtest_results"
MODELS_DIR = "/freqtrade/user_data/models"
OUTPUT_PATH = os.path.join(MODELS_DIR, "strategy_winrate_model.pkl")

PAIRS = [
    "AAVE/USDT:USDT", "ADA/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "ATOM/USDT:USDT", "AVAX/USDT:USDT", "BNB/USDT:USDT", "BTC/USDT:USDT",
    "CRV/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT", "ETC/USDT:USDT",
    "ETH/USDT:USDT", "FIL/USDT:USDT", "LINK/USDT:USDT", "NEAR/USDT:USDT",
    "OP/USDT:USDT", "SAND/USDT:USDT", "SOL/USDT:USDT", "TRX/USDT:USDT",
    "UNI/USDT:USDT", "XLM/USDT:USDT", "XRP/USDT:USDT",
]


def pair_to_dir(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def find_backtest_zip() -> str | None:
    if not os.path.isdir(BACKTEST_DIR):
        return None
    files = sorted(os.listdir(BACKTEST_DIR), reverse=True)
    for f in files:
        if f.endswith(".zip") and f.startswith("ml_gate"):
            return os.path.join(BACKTEST_DIR, f)
    for f in files:
        if f.endswith(".zip"):
            return os.path.join(BACKTEST_DIR, f)
    return None


def load_backtest_trades() -> list[dict]:
    zip_path = find_backtest_zip()
    if not zip_path:
        raise FileNotFoundError(f"No backtest zip in {BACKTEST_DIR}")
    logger.info(f"Loading trades from {zip_path}...")
    with zipfile.ZipFile(zip_path) as z:
        json_files = [n for n in z.namelist() if n.endswith(".json")]
        if not json_files:
            raise FileNotFoundError("No JSON in backtest zip")
        with z.open(json_files[0]) as f:
            data = _json.load(f)
    strategy_data = data.get("strategy", {})
    trades = []
    for strat_name, strat_info in strategy_data.items():
        trades = strat_info.get("trades", [])
        if trades:
            break
    logger.info(f"Loaded {len(trades)} trades")
    return trades


def load_pair_5m(pair: str) -> pd.DataFrame | None:
    pdir = pair_to_dir(pair)
    path = os.path.join(DATA_DIR, "binance", "futures", f"{pdir}-5m-futures.feather")
    if not os.path.exists(path):
        logger.warning(f"Missing 5m data: {path}")
        return None
    df = pd.read_feather(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def compute_indicator_row(row: pd.Series) -> dict:
    features = {
        "open": row.get("open", 0),
        "high": row.get("high", 0),
        "low": row.get("low", 0),
        "close": row.get("close", 0),
        "volume": row.get("volume", 0),
        "rsi_14": row.get("rsi_14", 50),
        "adx": row.get("adx", 20),
        "plus_di": row.get("plus_di", 20),
        "minus_di": row.get("minus_di", 20),
        "macd": row.get("macd", 0),
        "macdsignal": row.get("macdsignal", 0),
        "macdhist": row.get("macdhist", 0),
        "bb_upper": row.get("bb_upper", 0),
        "bb_middle": row.get("bb_middle", 0),
        "bb_lower": row.get("bb_lower", 0),
        "bb_width": row.get("bb_width", 0),
        "volume_ema": row.get("volume_ema", 0),
        "volume_ratio": row.get("volume_ratio", 1),
        "atr": row.get("atr", 0),
        "atr_pct": row.get("atr_pct", 0),
        "ema_50": row.get("ema_50", 0),
        "ema_200": row.get("ema_200", 0),
        "obv": row.get("obv", 0),
        "obv_ema": row.get("obv_ema", 0),
    }
    return features


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macdsignal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macdhist"] = df["macd"] - df["macdsignal"]

    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    df["volume_ema"] = df["volume"].ewm(span=10, adjust=False).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ema"] + 1e-10)

    df["bb_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - close.shift()).abs()
    tr3 = (df["low"] - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / (close + 1e-10)

    df["obv"] = (df["volume"] * ((close.diff() > 0).astype(int) * 2 - 1)).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=10, adjust=False).mean()

    df = df.bfill().ffill().fillna(0)
    return df


def extract_features_for_trade(trade: dict, pair_data: dict) -> dict | None:
    pair = trade["pair"]
    df = pair_data.get(pair)
    if df is None:
        return None
    open_date = pd.Timestamp(trade["open_date"])
    idx = (df["date"] - open_date).abs().idxmin()
    row = df.iloc[idx]
    return compute_indicator_row(row)


def main():
    t0 = time.time()
    logger.info("Loading backtest trades...")
    trades = load_backtest_trades()

    logger.info("Loading OHLCV data for all pairs...")
    pair_data = {}
    pairs_found = set()
    for t in trades:
        pairs_found.add(t["pair"])
    logger.info(f"Pairs in trades: {len(pairs_found)}")
    for pair in list(pairs_found):
        df = load_pair_5m(pair)
        if df is not None:
            df = compute_all_indicators(df)
            pair_data[pair] = df
            logger.info(f"  {pair}: {len(df)} candles")
    logger.info(f"Loaded data for {len(pair_data)}/{len(pairs_found)} pairs")

    samples = []
    for trade in trades:
        profit = trade.get("profit_ratio", 0)
        target = 1 if profit > 0 else 0
        feats = extract_features_for_trade(trade, pair_data)
        if feats is None:
            continue
        feats["target"] = target
        samples.append(feats)

    logger.info(f"Extracted {len(samples)} samples from {len(trades)} trades")
    df = pd.DataFrame(samples)
    feature_cols = [c for c in df.columns if c != "target"]
    logger.info(f"Features: {len(feature_cols)}, WR: {df['target'].mean():.3f}")

    X = df[feature_cols].values
    y = df["target"].values

    np.random.seed(42)
    perm = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Test WR: {y_test.mean():.3f}")

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "gbdt",
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "verbose": -1,
    }

    logger.info("Training LightGBM on strategy signals...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(-1)],
    )

    y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    acc = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    brier = np.mean((y_proba - y_test) ** 2)

    logger.info(f"Test: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} Brier={brier:.3f}")

    thresholds = np.linspace(0.3, 0.9, 61)
    results = []
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        tp_t = np.sum((y_test == 1) & (preds == 1))
        fp_t = np.sum((y_test == 0) & (preds == 1))
        fn_t = np.sum((y_test == 1) & (preds == 0))
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
        wr_t = y_test[preds == 1].mean() if preds.sum() > 0 else 0
        n_pass = int(preds.sum())
        results.append((thresh, f1_t, wr_t, n_pass))

    logger.info("Threshold scan (targeting 86% WR):")
    for thresh, f1_t, wr_t, n_pass in results:
        if wr_t >= 0.86:
            logger.info(f"  thr={thresh:.2f} F1={f1_t:.3f} WR={wr_t:.3f} n={n_pass}")
            break

    best_f1_idx = np.argmax([r[1] for r in results])
    best_thr, best_f1, best_wr, best_n = results[best_f1_idx]
    logger.info(f"Best F1: thr={best_thr:.2f} F1={best_f1:.3f} WR={best_wr:.3f} n={best_n}")

    target_thr = 0.5
    for thresh, _, wr_t, _ in results:
        if wr_t >= 0.86:
            target_thr = thresh
            break

    logger.info(f"Target threshold (for 86% WR): {target_thr:.2f}")

    importances = model.feature_importance()
    top_idx = np.argsort(importances)[::-1][:10]
    logger.info("Top 10 features:")
    for idx in top_idx:
        logger.info(f"  {feature_cols[idx]}: {importances[idx]}")

    metadata = {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": float(target_thr),
        "version": "2.0-strategy-signals",
        "training_date": pd.Timestamp.now().isoformat(),
        "feature_count": len(feature_cols),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_wr": float(y_test.mean()),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(metadata, OUTPUT_PATH)
    logger.info(f"Model saved to {OUTPUT_PATH}")
    logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
