import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging
import time
from pathlib import Path

# --- Setup folders safely ---
Path("plots/univariate").mkdir(parents=True, exist_ok=True)
Path("plots/bivariate").mkdir(parents=True, exist_ok=True)

# --- Setup logging ---
logging.basicConfig(
    filename="data_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_pipeline(df: pd.DataFrame) -> dict:
    start = time.time()
    logging.info("=== Pipeline started ===")

    result = {}
    result["summary"] = df.describe(include="all")
    result["dtypes"] = df.dtypes.to_dict()
    result["missing"] = df.isnull().sum().to_dict()

    # --- Handle missing numeric values ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    logging.info("Missing numeric values imputed")

    # --- Normalize numeric columns ---
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logging.info("Numeric columns normalized")

    # --- Univariate Analysis ---
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"plots/univariate/{col}_hist.png")
        plt.close()

        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"plots/univariate/{col}_box.png")
        plt.close()

    # --- Bivariate Analysis ---
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/bivariate/correlation_heatmap.png")
    plt.close()

    # Fraud vs Non-Fraud Analysis
    if "Class" in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x="Class", data=df, palette="Set2")
        plt.title("Fraud (1) vs Non-Fraud (0)")
        plt.tight_layout()
        plt.savefig("plots/bivariate/class_distribution.png")
        plt.close()

        for col in num_cols[:5]:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df["Class"], y=df[col])
            plt.title(f"{col} vs Class")
            plt.tight_layout()
            plt.savefig(f"plots/bivariate/{col}_vs_Class.png")
            plt.close()

        # --- Feature importance (only if 2+ classes exist) ---
        X = df.drop("Class", axis=1).select_dtypes(include=[np.number])
        y = df["Class"]
        if len(y.unique()) > 1:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            plt.figure(figsize=(6,4))
            imp.head(10).plot(kind="barh")
            plt.gca().invert_yaxis()
            plt.title("Top 10 Feature Importances")
            plt.tight_layout()
            plt.savefig("plots/bivariate/feature_importance.png")
            plt.close()
            result["feature_importance"] = imp.head(10).to_dict()
        else:
            logging.warning("Not enough classes to train RandomForest.")
            result["feature_importance"] = {}

    end = time.time()
    logging.info(f"Pipeline completed in {end - start:.2f}s")

    # --- Safe log reading ---
    log_file = Path("data_pipeline.log")
    if log_file.exists():
        with open(log_file) as f:
            result["logs"] = f.read()[-3000:]
    else:
        result["logs"] = "No logs yet."

    return result
