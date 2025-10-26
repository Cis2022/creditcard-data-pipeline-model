"""
Integrated Credit Card Fraud DataOps + ML Pipeline Dashboard
Features:
- Auto-refresh every 2 minutes
- Persist uploaded CSV across refresh
- DataOps: EDA, normalization, univariate/bivariate, feature importance, correlation heatmap, logs
- ML Pipeline: Logistic Regression & Random Forest, metrics, logs
- Two buttons to run pipelines manually
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import sys
import platform

from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline
from streamlit_autorefresh import st_autorefresh

# ----------------- Auto-refresh configuration -----------------
st_autorefresh(interval=120000, limit=None, key="refresh")

# ----------------- Page config -----------------
st.set_page_config(page_title="Credit Card Pipeline Dashboard", layout="wide")
st.title("💳 Credit Card Fraud DataOps + ML Dashboard")
st.markdown("Automated DataOps and ML pipelines with auto-refresh every 2 minutes")

# ----------------- Session state initialization -----------------
if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
if "last_run_dataops" not in st.session_state:
    st.session_state["last_run_dataops"] = None
if "last_run_ml" not in st.session_state:
    st.session_state["last_run_ml"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# ----------------- File upload -----------------
uploaded_file = st.file_uploader("📂 Choose CSV file (creditcard.csv)", type=["csv"])
use_sample = st.checkbox("Use packaged sample dataset (for demo)", value=False)

if uploaded_file is not None:
    try:
        st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)
        st.success(f"Loaded uploaded file: {getattr(uploaded_file, 'name', 'uploaded')}; shape: {st.session_state['uploaded_df'].shape}")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
elif use_sample:
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        st.session_state["uploaded_df"] = pd.read_csv(sample_path)
        st.success(f"Loaded packaged sample_data.csv; shape: {st.session_state['uploaded_df'].shape}")
    else:
        st.error("Sample data missing. Place sample_data.csv in app folder.")

df = st.session_state["uploaded_df"]

# ----------------- Pipeline runner -----------------
def run_all_pipelines(df):
    st.session_state["is_processing"] = True
    try:
        dataops_results = run_pipeline(df)
        ml_results = run_ml_pipeline(df)
        st.session_state["last_run_dataops"] = dataops_results.get("run_ts") or datetime.now().isoformat()
        st.session_state["last_run_ml"] = ml_results.get("run_ts") or datetime.now().isoformat()
        return dataops_results, ml_results
    finally:
        st.session_state["is_processing"] = False

# ----------------- Run buttons -----------------
col1, col2 = st.columns(2)
run_dataops = col1.button("▶️ Run DataOps Pipeline")
run_ml = col2.button("▶️ Run ML Pipeline")

should_run_dataops = run_dataops or (st.session_state["last_run_dataops"] is None)
should_run_ml = run_ml or (st.session_state["last_run_ml"] is None)

if df is not None:
    st.header("1️⃣ Data Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

    dataops_results, ml_results = None, None

    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait for it to finish.")
    else:
        if should_run_dataops or should_run_ml:
            with st.spinner("Running selected pipelines..."):
                dataops_results, ml_results = run_all_pipelines(df)

    # ----------------- DataOps Results -----------------
    if dataops_results:
        if "error" in dataops_results:
            st.error(f"DataOps Pipeline Error: {dataops_results['error']}")
        else:
            st.header("2️⃣ DataOps Summary Statistics")
            try:
                st.dataframe(pd.DataFrame(dataops_results["summary"]))
            except:
                st.write(dataops_results.get("summary"))

            st.header("3️⃣ Missing Values")
            st.json(dataops_results.get("missing", {}))

            st.header("4️⃣ Data Types")
            st.json(dataops_results.get("dtypes", {}))

            st.header("5️⃣ Generated Visualizations")
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
            st.text_area("📜 data_pipeline.log", value=dataops_results.get("logs", ""), height=240)

    # ----------------- ML Pipeline Results -----------------
    if ml_results:
        if "error" in ml_results:
            st.error(f"ML Pipeline Error: {ml_results['error']}")
        else:
            st.header("7️⃣ ML Model Evaluation")
            st.subheader("Selected Models: Logistic Regression & Random Forest")
            st.write("📊 Evaluation Metrics:")
            st.json(ml_results.get("metrics", {}))

            st.subheader("📁 ML Pipeline Logs")
            st.text_area("ML Pipeline Logs", value=ml_results.get("logs", ""), height=240)

    st.markdown("---")
    st.header("🧭 Application Insights")
    app_details = {
        "Application Name": "Credit Card Pipeline Dashboard",
        "Deployed Environment": os.getenv("STREAMLIT_SERVER_ADDRESS", "Local/Streamlit Cloud"),
        "Python Version": sys.version.split()[0],
        "Platform": platform.system() + " " + platform.release(),
        "Streamlit Version": st.__version__,
        "Current Working Directory": os.getcwd(),
        "Last DataOps Run": st.session_state.get("last_run_dataops"),
        "Last ML Pipeline Run": st.session_state.get("last_run_ml")
    }
    for k, v in app_details.items():
        st.write(f"**{k}:** {v}")
else:
    st.info("Please upload a CSV file or enable 'Use packaged sample dataset' to run the demo.")
