# data_pipeline.py
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Paths
ROOT_DIR = Path(".")
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
EDA_DIR = ARTIFACTS_DIR / "eda"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories
for p in [ARTIFACTS_DIR, PROCESSED_DIR, EDA_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str, logfile: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers when re-run
    if not logger.handlers:
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

@dataclass
class DataOpsResult:
    summary: dict
    dtypes: dict
    normalized: dict
    artifacts: dict

def load_data(logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded data: shape={df.shape}")
    return df

def preprocess_data(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    # Summary stats (not printed here; collected for app)
    missing_counts = df.isna().sum()
    missing_total = int(missing_counts.sum())

    # Impute numeric columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputed_cols = []
    for col in num_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            imputed_cols.append(col)

    # Normalize numeric features (exclude target 'Class')
    feature_cols = [c for c in num_cols if c != "Class"]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaled_values

    logger.info(
        f"Preprocessing complete: missing_total={missing_total}, "
        f"imputed_cols={imputed_cols}, scaled_cols={len(feature_cols)}"
    )

    normalized_info = {
        "scaled_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return df_scaled, {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_total": missing_total,
        "imputed_cols": imputed_cols
    }, normalized_info

def eda_analysis(df: pd.DataFrame, logger: logging.Logger) -> dict:
    # Compute correlations (Pearson) among numeric features
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    corr = sample_df.drop(columns=["Class"]).corr()

    # Top features correlated with target (point-biserial via Pearson against 0/1)
    target_corr = {
        col: float(np.corrcoef(sample_df[col], sample_df["Class"])[0, 1])
        for col in sample_df.columns
        if col != "Class" and pd.api.types.is_numeric_dtype(sample_df[col])
    }
    top_target_corr = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    # Binning Amount into quantiles
    df["Amount_bin"] = pd.qcut(df["Amount"], q=10, duplicates="drop")
    amount_bin_counts = df["Amount_bin"].value_counts().to_dict()

    # One-hot encode the bin for demonstration
    amount_bin_ohe = pd.get_dummies(df["Amount_bin"], prefix="AmountBin")
    # Simple feature importance (RandomForest)
    feature_cols = [c for c in df.columns if c not in ["Class", "Amount_bin"]]
    X = df[feature_cols]
    y = df["Class"]
    rf = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    rf.fit(X, y)
    importances = dict(zip(feature_cols, rf.feature_importances_))
    top_feat_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]

    # Visualizations
    # 1) Class distribution
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Class", data=df, ax=ax1)
    ax1.set_title("Class Distribution (Imbalanced)")
    class_dist_path = EDA_DIR / "class_distribution.png"
    fig1.savefig(class_dist_path, bbox_inches="tight")
    plt.close(fig1)

    # 2) Amount histogram
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Amount"], bins=50, ax=ax2, kde=False)
    ax2.set_title("Amount Distribution")
    amount_hist_path = EDA_DIR / "amount_hist.png"
    fig2.savefig(amount_hist_path, bbox_inches="tight")
    plt.close(fig2)

    # 3) Correlation heatmap (sampled)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    ax3.set_title("Correlation Heatmap (Sampled)")
    corr_heatmap_path = EDA_DIR / "corr_heatmap.png"
    fig3.savefig(corr_heatmap_path, bbox_inches="tight")
    plt.close(fig3)

    # 4) Scatter Time vs Amount colored by Class
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x=sample_df["Time"], y=sample_df["Amount"], hue=sample_df["Class"].astype(str),
        alpha=0.6, ax=ax4
    )
    ax4.set_title("Time vs Amount (Class-colored)")
    time_amount_scatter_path = EDA_DIR / "time_amount_scatter.png"
    fig4.savefig(time_amount_scatter_path, bbox_inches="tight")
    plt.close(fig4)

    # Compose EDA report
    eda_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "top_target_correlations": top_target_corr,
        "amount_bin_counts": amount_bin_counts,
        "feature_importance_top15": top_feat_imp,
        "correlation_matrix_shape": corr.shape,
    }
    eda_report_path = EDA_DIR / "eda_report.json"
    with open(eda_report_path, "w") as f:
        json.dump(eda_report, f, indent=2)

    logger.info("EDA complete: visuals and JSON report saved.")

    return {
        "class_distribution": class_dist_path,
        "amount_hist": amount_hist_path,
        "corr_heatmap": corr_heatmap_path,
        "time_amount_scatter": time_amount_scatter_path,
        "eda_report_path": eda_report_path,
    }

def run_dataops_pipeline() -> dict:
    logger = setup_logger("dataops", LOGS_DIR / "dataops.log")
    logger.info("=== DataOps pipeline started ===")

    df = load_data(logger)
    df_processed, summary, normalized_info = preprocess_data(df, logger)

    # Save processed CSV
    processed_csv = PROCESSED_DIR / "creditcard_processed.csv"
    df_processed.to_csv(processed_csv, index=False)
    logger.info(f"Processed data saved at {processed_csv}")

    eda_artifacts = eda_analysis(df_processed, logger)

    result = DataOpsResult(
        summary=summary,
        dtypes=df.dtypes.astype(str).to_dict(),
        normalized=normalized_info,
        artifacts={
            "processed_csv": processed_csv,
            "eda": {
                "class_distribution": eda_artifacts["class_distribution"],
                "amount_hist": eda_artifacts["amount_hist"],
                "corr_heatmap": eda_artifacts["corr_heatmap"],
                "time_amount_scatter": eda_artifacts["time_amount_scatter"],
            },
            "eda_report_path": eda_artifacts["eda_report_path"],
        }
    )
    logger.info("=== DataOps pipeline finished ===")

    # Return plain dict for Streamlit
    return {
        "summary": result.summary,
        "dtypes": result.dtypes,
        "normalized": result.normalized,
        "artifacts": result.artifacts,
    }