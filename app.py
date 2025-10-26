"""
app.py
Unified Streamlit app for:
1️⃣ DataOps Pipeline (EDA, normalization, visualization)
2️⃣ ML Pipeline (Logistic Regression, Random Forest)
Auto-refreshes every 2 minutes.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import sys
from streamlit_autorefresh import st_autorefresh
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline

# ----------------------------------------------------------
# Auto-refresh every 2 minutes (120000 ms)
# ----------------------------------------------------------
st_autorefresh(interval=120000, limit=None, key="refresh")

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud DataOps + ML Dashboard", layout="wide")
st.title("💳 Credit Card Fraud DataOps + ML Dashboard")
st.markdown("""
This unified dashboard allows you to:
- 📊 Run **DataOps pipeline** for EDA, normalization, and visualizations
- 🤖 Run **ML pipeline** for model training and evaluation
- 🔁 Auto-refresh every 2 minutes
""")

# ----------------------------------------------------------
# State initialization
# ----------------------------------------------------------
if "last_run_dataops" not in st.session_state:
    st.session_state["last_run_dataops"] = None
if "last_run_ml" not in st.session_state:
    st.session_state["last_run_ml"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# ----------------------------------------------------------
# File upload or sample data
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📂 Choose CSV file (creditcard.csv)", type=["csv"])
use_sample = st.checkbox("Use sample_data.csv (for demo)", value=False)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded uploaded file: {uploaded_file.name} | Shape: {df.shape}")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_sample:
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.success(f"✅ Loaded sample_data.csv | Shape: {df.shape}")
    else:
        st.error("Sample data missing — please add sample_data.csv to the app folder.")

# ----------------------------------------------------------
# Helper: safe_run wrapper
# ----------------------------------------------------------
def safe_run_pipeline(pipeline_func, label, state_key):
    """Safely run pipeline to avoid overlap"""
    if st.session_state["is_processing"]:
        st.warning("⚙️ Pipeline already running — please wait...")
        return None
    try:
        st.session_state["is_processing"] = True
        result = pipeline_func(df)
        st.session_state[state_key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result
    finally:
        st.session_state["is_processing"] = False

# ----------------------------------------------------------
# Buttons for both pipelines
# ----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    run_dataops = st.button("🚀 Run DataOps Pipeline Now")
    if st.session_state["last_run_dataops"]:
        st.caption(f"Last DataOps Run: {st.session_state['last_run_dataops']}")
with col2:
    run_ml = st.button("🤖 Run ML Pipeline Now")
    if st.session_state["last_run_ml"]:
        st.caption(f"Last ML Run: {st.session_state['last_run_ml']}")

# ----------------------------------------------------------
# Display data preview
# ----------------------------------------------------------
if df is not None:
    st.header("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"Shape: {df.shape}")

# ----------------------------------------------------------
# Run and display DataOps Pipeline
# ----------------------------------------------------------
if df is not None and run_dataops:
    with st.spinner("🧩 Running DataOps Pipeline..."):
        dataops_results = safe_run_pipeline(run_pipeline, "DataOps", "last_run_dataops")

    if dataops_results:
        if "error" in dataops_results:
            st.error(f"DataOps Pipeline Error: {dataops_results['error']}")
        else:
            st.header("📊 DataOps Pipeline Results")
            try:
                summary_df = pd.DataFrame(dataops_results["summary"])
                st.dataframe(summary_df)
            except Exception:
                st.json(dataops_results.get("summary", {}))

            st.subheader("Missing Values")
            st.json(dataops_results.get("missing", {}))

            st.subheader("Data Types")
            st.json(dataops_results.get("dtypes", {}))

            st.subheader("Generated Visualizations")
            images = [
                "correlation_heatmap.png",
                "class_distribution.png",
                "feature_importance.png",
                "univariate_plot.png",
                "bivariate_plot.png",
            ]
            cols = st.columns(2)
            for i, img in enumerate(images):
                if Path(img).exists():
                    with cols[i % 2]:
                        st.image(img, use_column_width=True, caption=img)
                else:
                    with cols[i % 2]:
                        st.info(f"{img} not available for this run.")

            st.subheader("📜 DataOps Logs")
            st.text_area("data_pipeline.log", value=dataops_results.get("logs", ""), height=240)

# ----------------------------------------------------------
# Run and display ML Pipeline
# ----------------------------------------------------------
if df is not None and run_ml:
    with st.spinner("🤖 Running ML Pipeline..."):
        ml_results = safe_run_pipeline(run_ml_pipeline, "ML", "last_run_ml")

    if ml_results:
        if "error" in ml_results:
            st.error(f"ML Pipeline Error: {ml_results['error']}")
        else:
            st.header("🧠 ML Model Evaluation")
            st.subheader("Selected Models: Logistic Regression & Random Forest")
            st.write("📊 Evaluation Metrics:")
            st.json(ml_results.get("metrics", {}))

            st.subheader("📁 ML Pipeline Logs")
            st.text_area("ml_pipeline.log", value=ml_results.get("logs", ""), height=240)

# ----------------------------------------------------------
# Application Insights
# ----------------------------------------------------------
st.markdown("---")
st.header("🧭 Application Insights")
app_details = {
    "Application Name": "Credit Card Fraud DataOps + ML Dashboard",
    "Deployed Environment": os.getenv("STREAMLIT_SERVER_ADDRESS", "Local/Streamlit Cloud"),
    "Python Version": sys.version.split()[0],
    "Streamlit Version": st.__version__,
    "Platform": os.name,
    "Working Directory": os.getcwd(),
    "Last DataOps Run": st.session_state.get("last_run_dataops"),
    "Last ML Run": st.session_state.get("last_run_ml"),
}
for k, v in app_details.items():
    st.write(f"**{k}:** {v}")

st.markdown("---")
st.caption("🔁 Auto-refresh active (every 2 minutes). | Built by Prem Charan 🚀")
