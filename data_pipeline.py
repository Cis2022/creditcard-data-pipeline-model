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

# Setup logging
logging.basicConfig(filename="data_pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline(df: pd.DataFrame):
    start = time.time()
    logging.info("=== Pipeline started ===")
    result = {}
    try:
        # Make a copy to avoid side-effects
        df = df.copy()

        # 1) Summary & dtypes
        summary = df.describe(include="all")
        result["summary"] = summary.fillna("").to_dict()
        result["dtypes"] = df.dtypes.astype(str).to_dict()
        logging.info("Computed summary and dtypes.")

        # 2) Missing values and impute numeric
        missing = df.isnull().sum()
        result["missing"] = missing.to_dict()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            logging.info("Imputed numeric missing values with column means.")

        # 3) Normalization
        if "Amount" in df.columns:
            try:
                scaler = StandardScaler()
                df["normalized_amount"] = scaler.fit_transform(df[["Amount"]])
                logging.info("Normalized 'Amount' and stored in 'normalized_amount'.")
            except Exception as e:
                logging.exception("Normalization failed for 'Amount'.")
        else:
            # create normalized versions for numeric columns with prefix norm_
            if len(num_cols) > 0:
                scaler = StandardScaler()
                normalized = scaler.fit_transform(df[num_cols])
                for i, col in enumerate(num_cols):
                    new_col = f"norm_{col}"
                    df[new_col] = normalized[:, i]
                logging.info("Normalized numeric columns to 'norm_' prefix.")

        # 4) Correlation heatmap (numeric only)
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
                logging.info("Saved correlation_heatmap.png")
        except Exception:
            logging.exception("Failed to create correlation heatmap.")

        # 5) Class distribution
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
                logging.info("Saved class_distribution.png")
            except Exception:
                logging.exception("Failed to save class distribution.")

        # 6) Univariate & Bivariate plots (pick first numeric cols)
        if len(num_cols) >= 1:
            try:
                plt.figure(figsize=(6,4))
                df[num_cols[0]].hist(bins=30)
                plt.title(f"Univariate: {num_cols[0]}")
                plt.tight_layout()
                plt.savefig("univariate_plot.png", bbox_inches="tight")
                plt.close()
                logging.info("Saved univariate_plot.png")
            except Exception:
                logging.exception("Failed univariate plot.")

        if len(num_cols) >= 2:
            try:
                plt.figure(figsize=(6,4))
                plt.scatter(df[num_cols[0]], df[num_cols[1]], s=6)
                plt.xlabel(num_cols[0]); plt.ylabel(num_cols[1])
                plt.title(f"Bivariate: {num_cols[0]} vs {num_cols[1]}")
                plt.tight_layout()
                plt.savefig("bivariate_plot.png", bbox_inches="tight")
                plt.close()
                logging.info("Saved bivariate_plot.png")
            except Exception:
                logging.exception("Failed bivariate plot.")

        # 7) Feature importance (if Class exists and data okay)
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
                    logging.info("Saved feature_importance.png")
                else:
                    logging.info("Feature importance skipped (no numeric features or only one class).")
            except Exception:
                logging.exception("Feature importance generation failed.")

        # 8) Finalize and return results
        end = time.time()
        elapsed = round(end - start, 2)
        logging.info(f"Pipeline completed in {elapsed} seconds")

        # read last logs
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
        logging.exception("Pipeline failed")
        # ensure logs available
        with open("data_pipeline.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} - ERROR - {str(e)}\n")
        return {"error": str(e), "logs": open("data_pipeline.log").read(), "run_ts": datetime.now().isoformat()}


