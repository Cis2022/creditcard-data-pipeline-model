"""
app.py
Combined DataOps + ML Pipeline Dashboard for Credit Card Fraud
- Auto-refresh every 2 minutes
- Upload CSV or use sample dataset
- Runs data_pipeline.run_pipeline and ml_pipeline.run_ml_pipeline
- Displays charts, logs, metrics, insights
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline
from streamlit_autorefresh import st_autorefresh
import os, platform, sys
from datetime import datetime

# ---------- Auto-refresh configuration ----------
st_autorefresh(interval=120000, limit=None, key="refresh")

# ---------- Page config ----------
st.set_page_config(page_title="Credit Card Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Dashboard")
st.markdown(
    "Automated DataOps + ML Pipeline with charts, metrics, logs and monitoring (auto-refresh every 2 minutes)"
)

# ---------- Session State ----------
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

if st.session_state["last_run"]:
    st.info(f"🔁 Last pipeline run: {st.session_state['last_run']}")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📂 Choose CSV file (creditcard.csv) to run pipelines", type=["csv"])
use_sample = st.checkbox("Use packaged sample dataset (for demo)", value=False)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded uploaded file: {getattr(uploaded_file, 'name', 'uploaded')}; shape: {df.shape}")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_sample:
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.success(f"Loaded packaged sample_data.csv; shape: {df.shape}")
    else:
        st.error("Sample data missing from package. Please put sample_data.csv in the app folder.")

# ---------- Unified run function ----------
def run_both_pipelines(df):
    """
    Run DataOps pipeline and ML pipeline sequentially
    """
    dataops_results = run_pipeline(df)
    ml_results = run_ml_pipeline(df)
    return dataops_results, ml_results

# ---------- Run pipeline safely ----------
def safe_run(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait for it to finish.")
        return None, None
    try:
        st.session_state["is_processing"] = True
        results, ml_results = run_both_pipelines(df)
        # update last run timestamp
        run_ts = results.get("run_ts") if isinstance(results, dict) else None
        st.session_state["last_run"] = run_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return results, ml_results
    finally:
        st.session_state["is_processing"] = False

# Manual run button
run_now = st.button("▶️ Run pipelines now (manual)")

# ---------- Main UI ----------
if df is not None:
    st.header("1️⃣ Data Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

    should_run = True  # Always run pipelines when CSV uploaded

    if should_run:
        with st.spinner("Processing both pipelines..."):
            results, ml_results = safe_run(df)
    else:
        results, ml_results = None, None
        st.info("Waiting for scheduled run (or click 'Run pipelines now').")

    # ---------- Display DataOps Results ----------
    if results is None:
        st.warning("No DataOps results to show right now.")
    else:
        if "error" in results:
            st.error(f"DataOps Pipeline error: {results['error']}")
        else:
            st.header("2️⃣ DataOps: Summary Statistics")
            try:
                summary_df = pd.DataFrame(results["summary"])
                st.dataframe(summary_df)
            except Exception:
                st.write(results.get("summary"))

            st.header("3️⃣ DataOps: Missing Values")
            st.json(results.get("missing", {}))

            st.header("4️⃣ DataOps: Data Types")
            st.json(results.get("dtypes", {}))

            st.header("5️⃣ DataOps: Generated Visualizations")
            images = [
                "correlation_heatmap.png",
                "class_distribution.png",
                "feature_importance.png",
                "univariate_plot.png",
                "bivariate_plot.png",
                "normalized_amount.png",
                "fraud_vs_nonfraud.png"
            ]
            cols = st.columns(2)
            for i, img in enumerate(images):
                p = Path(img)
                if p.exists():
                    with cols[i % 2]:
                        st.image(str(p), use_column_width=True, caption=img)
                else:
                    with cols[i % 2]:
                        st.write(f"ℹ️ {img} not available for this run.")

            st.header("6️⃣ DataOps Pipeline Logs")
            st.text_area("📜 data_pipeline.log", value=results.get("logs", ""), height=240)

    # ---------- Display ML Results ----------
    if ml_results:
        if "error" in ml_results:
            st.error(f"ML Pipeline error: {ml_results['error']}")
        else:
            st.header("7️⃣ ML Pipeline: Model Evaluation")
            st.subheader("Selected Models: Logistic Regression & Random Forest")
            st.write("📊 Evaluation Metrics:")
            st.json(ml_results.get("metrics", {}))

            st.subheader("📁 ML Pipeline Logs")
            st.text_area("ML Pipeline Logs", value=ml_results.get("logs", ""), height=240)

    # ---------- Application Insights ----------
    st.markdown("---")
    st.header("🧭 Application Insights")
    app_details = {
        "Application Name": "Credit Card Data + ML Dashboard",
        "Deployed Environment": os.getenv("STREAMLIT_SERVER_ADDRESS", "Local/Streamlit Cloud"),
        "Python Version": sys.version.split()[0],
        "Platform": platform.system() + " " + platform.release(),
        "Streamlit Version": st.__version__,
        "Current Working Directory": os.getcwd(),
        "Last Pipeline Run": st.session_state["last_run"]
    }
    for k, v in app_details.items():
        st.write(f"**{k}:** {v}")

else:
    st.info("Please upload a CSV file or enable 'Use packaged sample dataset' to run the pipelines.")

# ---------- Footer ----------
st.markdown("---")
st.caption("DataOps + ML Pipelines: Auto-refresh every 2 minutes (streamlit_autorefresh)")
