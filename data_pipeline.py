"""
data_pipeline.py
Implements run_pipeline(df) that:
- computes summary stats
- checks & imputes missing numeric values
- displays data types
- normalizes numeric columns (creates normalized columns)
- computes correlation heatmap (saves image)
- plots class distribution (saves image)
- generates univariate & bivariate plots (saves images)
- computes feature importance using RandomForest (saves image)
- logs all actions to data_pipeline.log
Returns a dict with summary, missing, dtypes, feature_importance, logs, run_ts, elapsed_time
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging, time, os
from datetime import datetime

# Create a custom logger for Data Pipeline
data_logger = logging.getLogger("data_pipeline")
data_logger.setLevel(logging.INFO)

# Create file handler
handler = logging.FileHandler("data_pipeline.log")
handler.setLevel(logging.INFO)

# Create formatter and add to handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Avoid duplicate handlers
if not data_logger.handlers:
    data_logger.addHandler(handler)

def run_pipeline(df: pd.DataFrame):
    start = time.time()
    data_logger.info("=== Pipeline started ===")
    result = {}
    try:
        df = df.copy()

        summary = df.describe(include="all")
        result["summary"] = summary.fillna("").to_dict()
        result["dtypes"] = df.dtypes.astype(str).to_dict()
        data_logger.info("Computed summary and dtypes.")

        missing = df.isnull().sum()
        result["missing"] = missing.to_dict()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            data_logger.info("Imputed numeric missing values with column means.")

        if "Amount" in df.columns:
            try:
                scaler = StandardScaler()
                df["normalized_amount"] = scaler.fit_transform(df[["Amount"]])
                data_logger.info("Normalized 'Amount' and stored in 'normalized_amount'.")
            except Exception as e:
                data_logger.exception("Normalization failed for 'Amount'.")
        else:
            if len(num_cols) > 0:
                scaler = StandardScaler()
                normalized = scaler.fit_transform(df[num_cols])
                for i, col in enumerate(num_cols):
                    new_col = f"norm_{col}"
                    df[new_col] = normalized[:, i]
                data_logger.info("Normalized numeric columns to 'norm_' prefix.")

        try:
            corr = df.select_dtypes(include=[np.number]).corr()
            if corr.size > 0:
                plt.figure(figsize=(10, 7))
                plt.title("Correlation Heatmap")
                im = plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
                plt.yticks(ticks=np.arange(len(corr.columns)), labels=corr.columns)
                plt.tight_layout()
                plt.savefig("correlation_heatmap.png", bbox_inches="tight")
                plt.close()
                data_logger.info("Saved correlation_heatmap.png")
        except Exception:
            data_logger.exception("Failed to create correlation heatmap.")

        if "Class" in df.columns:
            try:
                counts = df["Class"].value_counts()
                plt.figure(figsize=(6,4))
                counts.plot(kind="bar")
                plt.title("Class Distribution (0=Normal,1=Fraud)")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig("class_distribution.png", bbox_inches="tight")
                plt.close()
                data_logger.info("Saved class_distribution.png")
            except Exception:
                data_logger.exception("Failed to save class distribution.")

        if len(num_cols) >= 1:
            try:
                plt.figure(figsize=(6,4))
                df[num_cols[0]].hist(bins=30)
                plt.title(f"Univariate: {num_cols[0]}")
                plt.tight_layout()
                plt.savefig("univariate_plot.png", bbox_inches="tight")
                plt.close()
                data_logger.info("Saved univariate_plot.png")
            except Exception:
                data_logger.exception("Failed univariate plot.")

        if len(num_cols) >= 2:
            try:
                plt.figure(figsize=(6,4))
                plt.scatter(df[num_cols[0]], df[num_cols[1]], s=6)
                plt.xlabel(num_cols[0]); plt.ylabel(num_cols[1])
                plt.title(f"Bivariate: {num_cols[0]} vs {num_cols[1]}")
                plt.tight_layout()
                plt.savefig("bivariate_plot.png", bbox_inches="tight")
                plt.close()
                data_logger.info("Saved bivariate_plot.png")
            except Exception:
                data_logger.exception("Failed bivariate plot.")

        if "Class" in df.columns:
            try:
                X = df.drop("Class", axis=1).select_dtypes(include=[np.number]).fillna(0)
                y = df["Class"].fillna(0)
                if X.shape[1] > 0 and len(y.unique()) > 1:
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    model.fit(X, y)
                    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    topk = importances.head(10)
                    plt.figure(figsize=(8,6))
                    topk.plot(kind="barh")
                    plt.gca().invert_yaxis()
                    plt.title("Top 10 Feature Importances")
                    plt.tight_layout()
                    plt.savefig("feature_importance.png", bbox_inches="tight")
                    plt.close()
                    result["feature_importance"] = topk.to_dict()
                    data_logger.info("Saved feature_importance.png")
                else:
                    data_logger.info("Feature importance skipped (no numeric features or only one class).")
            except Exception:
                data_logger.exception("Feature importance generation failed.")

        end = time.time()
        elapsed = round(end - start, 2)
        data_logger.info(f"Pipeline completed in {elapsed} seconds")

        try:
            with open("data_pipeline.log", "r") as f:
                logs = f.read()
        except FileNotFoundError:
            logs = ""

        result["logs"] = logs[-5000:]
        result["run_ts"] = datetime.now().isoformat()
        result["elapsed_time"] = elapsed
        return result

    except Exception as e:
        data_logger.exception("Pipeline failed")
        with open("data_pipeline.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} - ERROR - {str(e)}\n")
        return {"error": str(e), "logs": open("data_pipeline.log").read(), "run_ts": datetime.now().isoformat()}