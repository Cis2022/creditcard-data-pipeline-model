# data_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(filename="data_pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline(df: pd.DataFrame):
    result = {}
    try:
        df = df.copy()
        logging.info("=== DataOps Pipeline started ===")

        # 1️⃣ Summary statistics
        summary = df.describe().transpose()
        result["summary"] = summary.to_dict()
        logging.info("Summary statistics computed.")

        # 2️⃣ Missing values
        missing = df.isnull().sum().to_dict()
        result["missing"] = missing
        logging.info("Missing values checked.")

        # 3️⃣ Data types
        result["dtypes"] = df.dtypes.apply(str).to_dict()

        # 4️⃣ Normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logging.info("Data normalized.")

        # 5️⃣ Univariate plots
        plt.figure(figsize=(10,6))
        df[numeric_cols].hist(bins=30, figsize=(12,8))
        plt.tight_layout()
        plt.savefig("univariate_plot.png")
        plt.close()

        # 6️⃣ Bivariate plot (correlation heatmap)
        plt.figure(figsize=(10,8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.tight_layout()
        plt.savefig("bivariate_plot.png")
        plt.close()

        # 7️⃣ Class distribution (Fraud vs Non-Fraud)
        if "Class" in df.columns:
            plt.figure(figsize=(6,4))
            sns.countplot(x="Class", data=df)
            plt.title("Fraud vs Non-Fraud")
            plt.savefig("class_distribution.png")
            plt.close()

        # 8️⃣ Feature importance (simple correlation with target)
        if "Class" in df.columns:
            corr_with_target = df.corr()["Class"].drop("Class").sort_values(ascending=False)
            plt.figure(figsize=(10,4))
            sns.barplot(x=corr_with_target.index, y=corr_with_target.values)
            plt.title("Feature Importance (Correlation with Class)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()

        # 9️⃣ Correlation heatmap saved separately
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

        # 10️⃣ Logging
        try:
            with open("data_pipeline.log", "r") as f:
                result["logs"] = f.read()[-5000:]
        except:
            result["logs"] = ""

        result["run_ts"] = datetime.now().isoformat()
        logging.info("=== DataOps Pipeline completed ===")
        return result

    except Exception as e:
        logging.exception("DataOps pipeline failed.")
        result["error"] = str(e)
        try:
            with open("data_pipeline.log", "r") as f:
                result["logs"] = f.read()[-5000:]
        except:
            result["logs"] = ""
        return result
