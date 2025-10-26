import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime

# Create a custom logger for ML pipeline
ml_logger = logging.getLogger("ml_pipeline")
ml_logger.setLevel(logging.INFO)

# Create file handler
ml_handler = logging.FileHandler("ml_pipeline.log")
ml_handler.setLevel(logging.INFO)

# Create formatter and add to handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ml_handler.setFormatter(formatter)

# Avoid duplicate logs
if not ml_logger.handlers:
    ml_logger.addHandler(ml_handler)

def run_ml_pipeline(df: pd.DataFrame):
    result = {}
    try:
        df = df.copy()
        ml_logger.info("=== ML Pipeline started ===")

        # Optional: Sample smaller data for memory efficiency
        if len(df) > 20000:
            df = df.sample(n=20000, random_state=42)
            ml_logger.info("Sampled 20,000 rows from dataset for memory efficiency.")

        if "Class" not in df.columns:
            raise ValueError("Target column 'Class' not found in dataset.")
        ml_logger.info("Target column 'Class' found.")

        X = df.drop("Class", axis=1).select_dtypes(include=[np.number]).fillna(0)
        y = df["Class"].fillna(0)
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        y = y.astype(int)
        ml_logger.info(f"Target y type: {type(y)}, shape: {y.shape}, unique values: {y.unique().tolist()}")

        if len(y.unique()) < 2:
            raise ValueError("Target column must have at least two classes.")
        ml_logger.info("Target column has sufficient class diversity.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        ml_logger.info(f"Data split into train and test sets: Train={len(X_train)}, Test={len(X_test)}")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga', n_jobs=1),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics[name] = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred), 4)
            }
            ml_logger.info(f"{name} trained and evaluated successfully.")
            ml_logger.info(f"{name} Metrics: {metrics[name]}")

        result["metrics"] = metrics

        try:
            with open("ml_pipeline.log", "r") as f:
                result["logs"] = f.read()[-5000:]
        except:
            result["logs"] = ""

        result["run_ts"] = datetime.now().isoformat()
        ml_logger.info("=== ML Pipeline completed ===")
        return result

    except Exception as e:
        ml_logger.exception("ML pipeline failed.")
        result["error"] = str(e)
        try:
            with open("ml_pipeline.log", "r") as f:
                result["logs"] = f.read()[-5000:]
        except:
            result["logs"] = ""
        return result