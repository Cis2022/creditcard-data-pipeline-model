import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(filename="ml_pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def run_ml_pipeline(df: pd.DataFrame):
    result = {}
    try:
        df = df.copy()
        logging.info("=== ML Pipeline started ===")

        # Check if target column exists
        if "Class" not in df.columns:
            raise ValueError("Target column 'Class' not found in dataset.")
        logging.info("Target column 'Class' found.")

        # Prepare features and target
        X = df.drop("Class", axis=1).select_dtypes(include=[np.number]).fillna(0)
        y = df["Class"].fillna(0)

        # Check if target has at least two classes
        if len(y.unique()) < 2:
            raise ValueError("Target column must have at least two classes.")
        logging.info("Target column has sufficient class diversity.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logging.info(f"Data split into train and test sets: Train={len(X_train)}, Test={len(X_test)}")

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        # Train and evaluate
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
            logging.info(f"{name} trained and evaluated successfully.")
            logging.info(f"{name} Metrics: {metrics[name]}")

        result["metrics"] = metrics

        # Read logs
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