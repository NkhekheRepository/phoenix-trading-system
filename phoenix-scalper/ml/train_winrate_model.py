import sys, os, numpy as np, pandas as pd, logging, joblib, time, gc, json
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import lightgbm as lgb
from feature_engine import FeatureEngine

DATA_DIR = "/freqtrade/user_data/data"
MODELS_DIR = "/freqtrade/user_data/models"
TRAINING_DATA_PATH = os.path.join(MODELS_DIR, "training_data.pkl")
OUTPUT_PATH = os.path.join(MODELS_DIR, "winrate_model.pkl")

PAIRS = [
    "1000PEPE/USDT:USDT", "1000SHIB/USDT:USDT", "AAVE/USDT:USDT", "ADA/USDT:USDT",
    "APE/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT", "ATOM/USDT:USDT",
    "AVAX/USDT:USDT", "BCH/USDT:USDT", "BNB/USDT:USDT", "BTC/USDT:USDT",
    "CHZ/USDT:USDT", "COMP/USDT:USDT", "CRV/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT", "EOS/USDT:USDT", "ETC/USDT:USDT", "ETH/USDT:USDT",
    "FIL/USDT:USDT", "LINK/USDT:USDT", "LTC/USDT:USDT", "MATIC/USDT:USDT",
    "NEAR/USDT:USDT", "OP/USDT:USDT", "QNT/USDT:USDT", "RUNE/USDT:USDT",
    "SAND/USDT:USDT", "SOL/USDT:USDT", "SUSHI/USDT:USDT", "THETA/USDT:USDT",
    "TRX/USDT:USDT", "UNI/USDT:USDT", "WAVES/USDT:USDT", "XLM/USDT:USDT",
    "XRP/USDT:USDT", "XTZ/USDT:USDT", "YFI/USDT:USDT", "ZEC/USDT:USDT",
]
TF_MAP = {"5m": "5m", "1h": "1h", "4h": "4h"}


def pair_to_file(pair: str, tf: str) -> str:
    p = pair.replace("/", "_").replace(":", "_")
    return os.path.join(DATA_DIR, "binance", "futures", f"{p}-{tf}-futures.feather")


def load_pair_data(pair: str) -> pd.DataFrame | None:
    futures_dir = os.path.join(DATA_DIR, "binance", "futures")
    if not os.path.isdir(futures_dir):
        logger.warning(f"Data directory not found: {futures_dir}")
        return None

    df_list = []
    for tf in ["5m", "1h", "4h"]:
        path = pair_to_file(pair, tf)
        if os.path.exists(path):
            df_tf = pd.read_feather(path)
            df_tf.columns = [c.lower() for c in df_tf.columns]
            if "date" not in df_tf.columns:
                continue
            df_tf["date"] = pd.to_datetime(df_tf["date"])
            suffix = "" if tf == "5m" else f"_{tf}"
            df_tf = df_tf.rename(columns={c: f"{c}{suffix}" for c in df_tf.columns if c not in ["date"]})
            df_list.append(df_tf)
        else:
            logger.warning(f"Missing data: {path}")

    if not df_list:
        return None

    df = df_list[0]
    for df_other in df_list[1:]:
        df = pd.merge_asof(df.sort_values("date"), df_other.sort_values("date"), on="date", direction="backward")

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df.get("close", df.get("close_5m"))
    high = df.get("high", df.get("high_5m"))
    low = df.get("low", df.get("low_5m"))
    volume = df.get("volume", df.get("volume_5m", pd.Series(0, index=df.index)))

    df["rsi_14"] = 50.0
    df["adx"] = 20.0
    df["plus_di"] = 20.0
    df["minus_di"] = 20.0
    df["macd"] = 0.0
    df["macdsignal"] = 0.0
    df["macdhist"] = 0.0

    if close is not None and len(close) > 20:
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

    df["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-10)
    df["volume_ema"] = volume.ewm(span=20, adjust=False).mean()

    df["bb_upper"] = close.rolling(20).mean() + 2 * close.rolling(20).std()
    df["bb_middle"] = close.rolling(20).mean()
    df["bb_lower"] = close.rolling(20).mean() - 2 * close.rolling(20).std()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    df["atr"] = 0.0
    if high is not None and low is not None and close is not None:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    df["ema_50"] = close.ewm(span=50, adjust=False).mean() if close is not None else 0
    df["ema_200"] = close.ewm(span=200, adjust=False).mean() if close is not None else 0
    df["is_bull"] = (df["ema_50"] > df["ema_200"]).astype(int)

    df["is_bull_1h"] = df.get("is_bull_1h", 1)
    df["rsi_14_1h"] = df.get("rsi_14_1h", 50)
    df["is_bull_4h"] = df.get("is_bull_4h", 1)
    df["adx_1h"] = df.get("adx_1h", 20)
    df["adx_4h"] = df.get("adx_4h", 20)
    df["rsi_14_4h"] = df.get("rsi_14_4h", 50)

    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday

    df["obv"] = 0
    df["obv_ema"] = 0

    df = df.bfill().ffill().fillna(0)

    kf_cols = [c for c in df.columns if c.startswith("kf_")]
    if not kf_cols:
        for col in ["kf_price", "kf_trend", "kf_prediction", "kf_confidence",
                     "kf_direction", "kf_innovation", "kf_S", "kf_price_momentum",
                     "kf_trend_acceleration", "kf_prediction_error", "kf_regime_score",
                     "kf_vol_of_trend", "kf_atr_ratio"]:
            df[col] = 0.0

    hmm_cols = [c for c in df.columns if c.startswith("hmm_")]
    if not hmm_cols:
        for col in ["hmm_regime", "hmm_p_bull", "hmm_p_range", "hmm_p_bear",
                     "hmm_regime_stability", "hmm_transition_risk", "hmm_vol_regime",
                     "hmm_trend_strength"]:
            df[col] = 0.5

    for col in ["di_spread", "macd_hist_sign", "volume_spike",
                 "ema_aligned", "bullish_candle", "close_gt_open_pct",
                 "candle_range", "candle_body", "close_open_ratio"]:
        if col not in df.columns:
            df[col] = 0.0

    if "atr_pct" not in df.columns:
        df["atr_pct"] = df["atr"] / (close + 1e-10)

    if "pair_enc" not in df.columns:
        df["pair_enc"] = 0
    if "tag_enc" not in df.columns:
        df["tag_enc"] = 0

    return df


def generate_training_data() -> pd.DataFrame:
    engine = FeatureEngine(forward_bars=24, min_samples=50000)
    all_samples = []
    pair_idx = 0

    for pair in PAIRS:
        t0 = time.time()
        logger.info(f"Loading {pair}...")
        df = load_pair_data(pair)
        if df is None or len(df) < 500:
            logger.warning(f"Skipping {pair}, insufficient data")
            continue
        logger.info(f"  {len(df)} candles, computing indicators...")
        df = compute_indicators(df)
        df["pair_enc"] = pair_idx
        pair_idx += 1
        samples = engine.generate_training_data(df)
        if len(samples) > 0:
            pos_rate = samples["target"].mean()
            logger.info(f"  {pair}: {len(samples)} samples, win rate {pos_rate:.3f}")
            all_samples.append(samples)
        else:
            logger.warning(f"  {pair}: 0 samples")
        logger.info(f"  done in {time.time()-t0:.1f}s")

    if not all_samples:
        raise RuntimeError("No training data generated!")

    result = pd.concat(all_samples, ignore_index=True)
    logger.info(f"Total: {len(result)} samples, win rate {result['target'].mean():.3f}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    result.to_pickle(TRAINING_DATA_PATH)
    logger.info(f"Saved to {TRAINING_DATA_PATH}")
    return result


def metrics(y_true, y_pred, y_proba=None):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    brier = np.mean((y_proba - y_true) ** 2) if y_proba is not None else 0
    return acc, prec, rec, f1, brier


def main():
    t0 = time.time()
    model_data_path = TRAINING_DATA_PATH

    if not os.path.exists(model_data_path):
        logger.info("No cached training data found, generating...")
        generate_training_data()

    logger.info("Loading training data...")
    df = pd.read_pickle(model_data_path)
    logger.info(f"Loaded {len(df)} samples, {len(df.columns)-1} features")

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    logger.info(f"Win rate: {y.mean():.3f}")
    logger.info(f"Features: {len(feature_cols)}")

    del df
    gc.collect()

    np.random.seed(42)
    perm = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "verbose": -1,
    }

    logger.info("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    acc, prec, rec, f1, brier_sc = metrics(y_test, y_pred, y_proba)
    logger.info(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    logger.info(f"Brier score: {brier_sc:.4f}")

    thresholds = np.linspace(0.3, 0.9, 61)
    best_f1 = 0
    best_threshold = 0.5
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        tp_t = np.sum((y_test == 1) & (preds == 1))
        fp_t = np.sum((y_test == 0) & (preds == 1))
        fn_t = np.sum((y_test == 1) & (preds == 0))
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = thresh

    mask = y_proba >= best_threshold
    wr_at_best = y_test[mask].mean() if np.sum(mask) > 0 else 0
    n_at_best = int(np.sum(mask))
    logger.info(f"Best threshold: {best_threshold:.2f} (F1={best_f1:.3f}, WR={wr_at_best:.3f}, n={n_at_best})")

    importances = model.feature_importance()
    top_idx = np.argsort(importances)[::-1][:15]
    logger.info("Top 15 features:")
    for idx in top_idx:
        logger.info(f"  {feature_cols[idx]}: {importances[idx]}")

    metadata = {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "version": "1.0-scalper",
        "training_date": pd.Timestamp.now().isoformat(),
        "feature_count": len(feature_cols),
        "training_samples": len(X_train),
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(metadata, OUTPUT_PATH)
    logger.info(f"Model saved to {OUTPUT_PATH}")
    logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
