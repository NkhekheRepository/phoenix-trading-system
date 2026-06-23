import sys, os, numpy as np, pandas as pd, logging, joblib, time
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

DATA_PATH = os.path.join(os.path.dirname(__file__), "models", "training_data.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "models", "winrate_model.pkl")


def main():
    t0 = time.time()
    logger.info("Loading training data...")
    df = pd.read_pickle(DATA_PATH)
    logger.info(f"Loaded {len(df)} samples, {len(df.columns)-1} features")

    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols].values
    y = df['target'].values

    logger.info(f"Win rate: {y.mean():.3f}")
    logger.info(f"Features: {len(feature_cols)}")

    del df
    import gc; gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    logger.info("Training LightGBM...")
    model = LGBMClassifier(
        n_estimators=500, max_depth=6, num_leaves=31, min_child_samples=100,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
        learning_rate=0.05, class_weight="balanced", random_state=42, verbosity=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(50), log_evaluation(-1)],
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_proba)

    logger.info(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    logger.info(f"Brier score: {brier:.4f}")

    thresholds = np.linspace(0.3, 0.9, 61)
    best_f1 = 0
    best_threshold = 0.5
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        f1_t = f1_score(y_test, preds, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = thresh

    wr_at_best = y_test[y_proba >= best_threshold].mean() if np.sum(y_proba >= best_threshold) > 0 else 0
    n_at_best = int(np.sum(y_proba >= best_threshold))
    logger.info(f"Best threshold: {best_threshold:.2f} (F1={best_f1:.3f}, WR={wr_at_best:.3f}, n={n_at_best})")

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    logger.info("Top 15 features:")
    for idx in top_idx:
        logger.info(f"  {feature_cols[idx]}: {importances[idx]}")

    metadata = {
        'model': model,
        'feature_cols': feature_cols,
        'threshold': best_threshold,
        'version': '3.0',
        'training_date': pd.Timestamp.now().isoformat(),
        'feature_count': len(feature_cols),
        'training_samples': len(X_train),
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
    }

    joblib.dump(metadata, OUTPUT_PATH)
    logger.info(f"Model saved to {OUTPUT_PATH}")
    logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
