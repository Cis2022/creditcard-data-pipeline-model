import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(filename="data_pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline(df: pd.DataFrame):
    result = {}
    try:
        df = df.copy()
        logging.info("=== DataOps Pipeline started ===")
        result["run_ts"] = datetime.now().isoformat()

        # Summary
        result["summary"] = df.describe(include="all").T

        # Data types
        result["dtypes"] = df.dtypes.apply(str).to_dict()

        # Missing values
        result["missing"] = df.isnull().sum().to_dict()

        # Impute numeric missing values
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        # Normalize Amount column
        if "Amount" in df.columns:
            df["normalized_amount"] = StandardScaler().fit_transform(df[["Amount"]])
            logging.info("'Amount' normalized")

        # Correlation heatmap
        plt.figure(figsize=(8,6))
        corr = df.corr(numeric_only=True)
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png", bbox_inches="tight")
        plt.close()

        # Class distribution
        if "Class" in df.columns:
            plt.figure()
            df["Class"].value_counts().plot(kind="bar")
            plt.title("Class Distribution (0=Normal,1=Fraud)")
            plt.savefig("fraud_vs_nonfraud.png", bbox_inches="tight")
            plt.close()

        # Feature importance
        if "Class" in df.columns:
            X = df.drop("Class", axis=1).select_dtypes(include=[np.number])
            y = df["Class"]
            if len(y.unique()) > 1:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                imp.head(10).plot(kind="barh")
                plt.title("Top 10 Feature Importances")
                plt.gca().invert_yaxis()
                plt.savefig("feature_importance.png", bbox_inches="tight")
                plt.close()
                result["feature_importance"] = imp.head(10).to_dict()

        # Univariate plots (numeric columns)
        for col in num_cols[:5]:  # limit for demo
            plt.figure()
            df[col].hist(bins=30)
            plt.title(f"Histogram of {col}")
            plt.savefig(f"univariate_{col}.png", bbox_inches="tight")
            plt.close()

        # Bivariate plots (numeric vs numeric)
        if len(num_cols) >= 2:
            plt.figure()
            plt.scatter(df[num_cols[0]], df[num_cols[1]], alpha=0.5)
            plt.xlabel(num_cols[0])
            plt.ylabel(num_cols[1])
            plt.title(f"Bivariate Plot: {num_cols[0]} vs {num_cols[1]}")
            plt.savefig("bivariate_plot.png", bbox_inches="tight")
            plt.close()

        # Read logs
        try:
            with open("data_pipeline.log", "r") as f:
                result["logs"] = f.read()[-5000:]
        except:
            result["logs"] = ""

        logging.info("=== DataOps Pipeline completed ===")
        return result

    except Exception as e:
        logging.exception("DataOps pipeline failed.")
        result["error"] = str(e)
        return result
