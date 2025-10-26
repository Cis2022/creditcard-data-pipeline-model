"""
app.py
Streamlit front-end for Credit Card Fraud DataOps + ML Dashboard
Features:
- Auto-refresh every 2 minutes
- Upload CSV or use sample_data.csv
- DataOps: summary, missing values, dtypes, normalization, univariate/bivariate, feature importance, fraud vs non-fraud, correlation heatmap
- ML: Logistic Regression & Random Forest evaluation metrics
- Logs & Application Insights
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime
import platform, sys

# ---------- Auto-refresh every 2 minutes ----------
st_autorefresh(interval=120000, limit=None, key="refresh")

# ---------- Page configuration ----------
st.set_page_config(page_title="Credit Card DataOps + ML Dashboard", layout="wide")
st.title("💳 Credit Card Fraud DataOps + ML Dashboard")
st.markdown("Automated pipelines with logs, charts, ML metrics, and monitoring (auto-refresh every 2 minutes)")

# ---------- Session state ----------
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# ---------- File upload ----------
uploaded_file = st.file_uploader("📂 Choose CSV file (creditcard.csv)", type=["csv"])
use_sample = st.checkbox("Use packaged sample dataset (for demo)", value=False)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded uploaded file: {getattr(uploaded_file,'name','uploaded')}; shape: {df.shape}")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_sample:
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.success(f"Loaded packaged sample_data.csv; shape: {df.shape}")
    else:
        st.error("Sample data missing. Please put sample_data.csv in the app folder.")

# ---------- Safe run function ----------
def safe_run(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait for it to finish.")
        return None, None
    st.session_state["is_processing"] = True
    try:
        dataops_results = run_pipeline(df)
        ml_results = run_ml_pipeline(df)
        # update last run timestamp
        run_ts = dataops_results.get("run_ts") if isinstance(dataops_results, dict) else None
        st.session_state["last_run"] = run_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return dataops_results, ml_results
    finally:
        st.session_state["is_processing"] = False

# ---------- Manual run button ----------
run_now = st.button("▶️ Run pipelines now (manual)")

# ---------- Main UI ----------
if df is not None:
    st.header("1️⃣ Data Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

    # Decide whether to run pipelines
    should_run = run_now or st.session_state["last_run"] is None
    if st.session_state["last_run"]:
        try:
            last = datetime.fromisoformat(st.session_state["last_run"]) if "T" in st.session_state["last_run"] else datetime.strptime(st.session_state["last_run"], "%Y-%m-%d %H:%M:%S")
            delta = (datetime.now() - last).total_seconds()
            if delta >= 110:
                should_run = True
        except:
            should_run = True

    if should_run:
        with st.spinner("Processing DataOps + ML pipelines..."):
            results, ml_results = safe_run(df)
    else:
        results, ml_results = None, None
        st.info("Waiting for scheduled run (auto-refresh) or click 'Run pipelines now'.")

    # ---------- Display DataOps results ----------
    if results:
        if "error" in results:
            st.error(f"DataOps Pipeline error: {results['error']}")
        else:
            st.header("2️⃣ Summary Statistics")
            try:
                summary_df = pd.DataFrame(results["summary"])
                st.dataframe(summary_df)
            except:
                st.write(results.get("summary"))

            st.header("3️⃣ Missing Values")
            st.json(results.get("missing", {}))

            st.header("4️⃣ Data Types")
            st.json(results.get("dtypes", {}))

            st.header("5️⃣ Generated Visualizations")
            images = ["correlation_heatmap.png", "class_distribution.png", "feature_importance.png",
                      "univariate_plot.png", "bivariate_plot.p
