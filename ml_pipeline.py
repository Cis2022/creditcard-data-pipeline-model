# ml_pipeline.py
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)

# Paths
ROOT_DIR = Path(".")
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
ML_DIR = ARTIFACTS_DIR / "ml"
LOGS_DIR = ROOT_DIR / "logs"
METRICS_DIR = ROOT_DIR / "metrics"

# Ensure directories
for p in [ARTIFACTS_DIR, PROCESSED_DIR, ML_DIR, LOGS_DIR, METRICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str, logfile: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def load_processed_or_raw() -> pd.DataFrame:
    processed = PROCESSED_DIR / "creditcard_processed.csv"
    if processed.exists():
        return pd.read_csv(processed)
    else:
        # Fallback if DataOps not run yet
        from data_pipeline import DATA_PATH
        return pd.read_csv(DATA_PATH)

def evaluate_model(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def plot_roc_curves(y_test, proba_dict: dict[str, np.ndarray]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, y_proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{model_name}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    out = ML_DIR / "roc_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_rf_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> Path:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
    ax.set_title("Random Forest Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    out = ML_DIR / "rf_feature_importance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

def run_mlops_pipeline() -> dict:
    logger = setup_logger("mlops", LOGS_DIR / "mlops.log")
    logger.info("=== ML pipeline started ===")

    df = load_processed_or_raw()
    # Basic split
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 1) Logistic Regression (with class_weight for imbalance)
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    metrics_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr)

    # 2) Random Forest (with class_weight for imbalance)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    metrics_rf = evaluate_model(y_test, y_pred_rf, y_proba_rf)

    logger.info(
        f"LR metrics: {metrics_lr} | RF metrics: {metrics_rf}"
    )

    # ROC Curves
    roc_path = plot_roc_curves(y_test, {"LogisticRegression": y_proba_lr, "RandomForest": y_proba_rf})
    # RF Feature Importance
    rf_imp_path = plot_rf_feature_importance(rf, X.columns.tolist())

    # Log metrics history (at least 4 metrics)
    ts = datetime.utcnow().isoformat()
    history_csv = METRICS_DIR / "ml_metrics_history.csv"
    row_lr = {
        "timestamp": ts,
        "model": "LogisticRegression",
        **metrics_lr
    }
    row_rf = {
        "timestamp": ts,
        "model": "RandomForest",
        **metrics_rf
    }
    hist_df = pd.DataFrame([row_lr, row_rf])
    if history_csv.exists():
        hist_df.to_csv(history_csv, mode="a", header=False, index=False)
    else:
        hist_df.to_csv(history_csv, index=False)

    # Latest snapshot JSON
    latest_json = METRICS_DIR / "ml_latest.json"
    with open(latest_json, "w") as f:
        json.dump({"timestamp": ts, "LogisticRegression": metrics_lr, "RandomForest": metrics_rf}, f, indent=2)

    logger.info("=== ML pipeline finished ===")

    return {
        "metrics": {
            "LogisticRegression": metrics_lr,
            "RandomForest": metrics_rf
        },
        "artifacts": {
            "roc_curve": roc_path,
            "rf_feature_importance": rf_imp_path,
            "metrics_history_csv": history_csv,
            "latest_json": latest_json
        }
    }

# Expose for app.py
__all__ = [
    "run_mlops_pipeline",
    "METRICS_DIR"
]