# data_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging, time

logging.basicConfig(filename="data_pipeline.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline(df):
    start = time.time()
    logging.info("=== Pipeline started ===")

    result = {}

    # Summary
    result["summary"] = df.describe(include="all")

    # Data types
    result["dtypes"] = df.dtypes.to_dict()

    # Missing values
    missing = df.isnull().sum()
    result["missing"] = missing.to_dict()
    logging.info(f"Missing values:\n{missing.to_string()}")

    # Impute numeric missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        logging.info("Numeric missing values imputed with column means")
    else:
        logging.info("No numeric columns to impute")

    # Normalize 'Amount' if present
    if "Amount" in df.columns:
        df["normalized_amount"] = StandardScaler().fit_transform(df[["Amount"]])
        logging.info("'Amount' normalized to 'normalized_amount'")
    else:
        logging.info("'Amount' column not present")

    # Correlation heatmap
    try:
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(8,6))
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info("Saved correlation_heatmap.png")
    except Exception as e:
        logging.error(f"Failed to create correlation heatmap: {e}")

    # Class distribution plot if Class column exists
    if "Class" in df.columns:
        try:
            plt.figure(figsize=(5,4))
            df["Class"].value_counts().plot(kind="bar")
            plt.title("Class Distribution (0=Normal,1=Fraud)")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
            plt.close()
            logging.info("Saved class_distribution.png")
            result["class_counts"] = df["Class"].value_counts().to_dict()
        except Exception as e:
            logging.error(f"Failed to create class distribution: {e}")
    else:
        logging.info("No 'Class' column; skipping class distribution")

    # Feature importance (RandomForest) if Class present and numeric features exist
    if "Class" in df.columns:
        X = df.drop(columns=["Class"], errors="ignore").select_dtypes(include=[np.number])
        y = df["Class"]
        if X.shape[1] > 0 and len(y.unique()) > 1:
            try:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X.fillna(0), y)
                imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                plt.figure(figsize=(8,5))
                imp.head(10).plot(kind="barh")
                plt.gca().invert_yaxis()
                plt.title("Top 10 Feature Importances")
                plt.tight_layout()
                plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
                plt.close()
                logging.info("Saved feature_importance.png")
                result["feature_importance"] = imp.head(10).to_dict()
            except Exception as e:
                logging.error(f"Failed to compute feature importance: {e}")
        else:
            logging.info("Not enough features or single-class target for feature importance")
    else:
        logging.info("No 'Class' column; skipping feature importance")

    end = time.time()
    logging.info(f"Pipeline completed in {end - start:.2f} seconds")
    # include last part of logs in return
    try:
        with open("data_pipeline.log", "r") as f:
            logs = f.read()[-4000:]
    except FileNotFoundError:
        logs = ""
    result["logs"] = logs
    result["status"] = "success"
    return result
