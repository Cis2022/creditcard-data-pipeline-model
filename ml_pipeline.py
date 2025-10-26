# ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(filename="ml_pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def run_ml_pipeline(df: pd.DataFrame):
    result = {}
    try:
        df = df.copy()
        logging.info("=== ML Pipeline started ===")

        # Target check
        if "Class" not in df.columns:
            raise ValueError("Target column 'Class' not found.")

        X = df.drop("Class", axis=1).select_dtypes(include=[np.number]).fillna(0)
        y = df["Class"].fillna(0)

        if len(y.unique()) < 2:
            raise ValueError("Target must have at least two classes.")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
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
            logging.info(f"{name} trained successfully. Metrics: {metrics[name]}")

        result["metrics"] = metrics

        # Logs
        try:
            with open("ml_pipeline.log", "r") as f:
                result["logs"] = f.read()[-5000:]
        except:
            result["logs"] = ""

        result["run_ts"] = datetime.now().isoformat()
        logging.info("=== ML Pipeline completed ===")
        return result

    except Exception as e:
        logging.exception("ML pipeline failed.")
        result["error"] = str(e)
        try:
            with open("ml_pipeline.log", "r") as f:
                result["logs"] = f.read()[-5000:]
        except:
            result["logs"] = ""
        return result
