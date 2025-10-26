"""
app.py
Streamlit front-end for Credit Card Fraud DataOps + ML Pipeline Dashboard
Features:
- Two independent pipelines: DataOps & ML
- Auto-refresh every 2 minutes
- Upload CSV or use sample_data.csv
- Displays summary, charts, logs, evaluation metrics, and insights
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import os
import platform
import sys

# ---------- Auto-refresh configuration ----------
st_autorefresh(interval=120000, limit=None, key="refresh")  # 2 minutes

# ---------- Page config ----------
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Dashboard")
st.markdown("Automated DataOps & ML Pipeline with auto-refresh every 2 minutes")

# ---------- Initialize session state ----------
for key in ["is_processing_dataops", "is_processing_ml", "results_dataops", "results_ml",
            "last_run_dataops", "last_run_ml"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- File upload ----------
uploaded_file = st.file_uploader("📂 Choose CSV file (creditcard.csv)", type=["csv"])
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
        st.error("Sample data missing from package. Please add sample_data.csv.")

# ---------- Safe run functions ----------
def safe_run_dataops(df):
    if st.session_state.get("is_processing_dataops", False):
        st.warning("DataOps pipeline already running — please wait.")
        return st.session_state.get("results_dataops")
    try:
        st.session_state["is_processing_dataops"] = True
        results = run_pipeline(df)
        st.session_state["results_dataops"] = results
        st.session_state["last_run_dataops"] = results.get("run_ts") or datetime.now().isoformat()
        return results
    finally:
        st.session_state["is_processing_dataops"] = False

def safe_run_ml(df):
    if st.session_state.get("is_processing_ml", False):
        st.warning("ML pipeline already running — please wait.")
        return st.session_state.get("results_ml")
    try:
        st.session_state["is_processing_ml"] = True
        results = run_ml_pipeline(df)
        st.session_state["results_ml"] = results
        st.session_state["last_run_ml"] = results.get("run_ts") or datetime.now().isoformat()
        return results
    finally:
        st.session_state["is_processing_ml"] = False

# ---------- Manual run buttons ----------
col1, col2 = st.columns(2)
dataops_results = ml_results = None
with col1:
    if st.button("▶️ Run DataOps Pipeline"):
        if df is not None:
            dataops_results = safe_run_dataops(df)
        else:
            st.warning("Upload a CSV first.")
with col2:
    if st.button("▶️ Run ML Pipeline"):
        if df is not None:
            ml_results = safe_run_ml(df)
        else:
            st.warning("Upload a CSV first.")

# ---------- Auto-run if never run ----------
if df is not None:
    if st.session_state["results_dataops"] is None:
        dataops_results = safe_run_dataops(df)
    else:
        dataops_results = st.session_state["results_dataops"]

    if st.session_state["results_ml"] is None:
        ml_results = safe_run_ml(df)
    else:
        ml_results = st.session_state["results_ml"]

# ---------- Display DataOps Results ----------
if dataops_results:
    st.markdown("---")
    st.header("💠 DataOps Pipeline Results")

    if "error" in dataops_results:
        st.error(f"DataOps pipeline error: {dataops_results['error']}")
    else:
        st.subheader("1️⃣ Summary Statistics")
        try:
            st.dataframe(pd.DataFrame(dataops_results["summary"]))
        except Exception:
            st.write(dataops_results.get("summary"))

        st.subheader("2️⃣ Missing Values")
        st.json(dataops_results.get("missing", {}))

        st.subheader("3️⃣ Data Types")
        st.json(dataops_results.get("dtypes", {}))

        st.subheader("4️⃣ Charts / Visualizations")
        images = [
            "correlation_heatmap.png",
            "class_distribution.png",
            "feature_importance.png",
            "univariate_plot.png",
            "bivariate_plot.png"
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

        st.subheader("5️⃣ DataOps Logs")
        st.text_area("📜 data_pipeline.log", value=dataops_results.get("logs", ""), height=240)

# ---------- Display ML Pipeline Results ----------
if ml_results:
    st.markdown("---")
    st.header("💠 ML Pipeline Results")
    if "error" in ml_results:
        st.error(f"ML pipeline error: {ml_results['error']}")
    else:
        st.subheader("1️⃣ Selected Models: Logistic Regression & Random Forest")
        st.write("📊 Evaluation Metrics:")
        st.json(ml_results.get("metrics", {}))

        st.subheader("2️⃣ ML Pipeline Logs")
        st.text_area("📜 ml_pipeline.log", value=ml_results.get("logs", ""), height=240)

# ---------- Application Insights ----------
st.markdown("---")
st.header("🧭 Application Insights")
app_details = {
    "Application Name": "Credit Card Fraud Dashboard",
    "Deployed Environment": os.getenv("STREAMLIT_SERVER_ADDRESS", "Local/Streamlit Cloud"),
    "Python Version": sys.version.split()[0],
    "Platform": platform.system() + " " + platform.release(),
    "Streamlit Version": st.__version__,
    "Current Working Directory": os.getcwd(),
    "Last DataOps Run": st.session_state.get("last_run_dataops"),
    "Last ML Run": st.session_state.get("last_run_ml")
}
for k, v in app_details.items():
    st.write(f"**{k}:** {v}")
