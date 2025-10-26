import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging
import time
from pathlib import Path

# --- Setup folders ---
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
        # Histogram
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        p
