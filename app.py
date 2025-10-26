import streamlit as st
import pandas as pd
from pathlib import Path
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import os
import platform

# Auto-refresh every 2 minutes
st_autorefresh(interval=120000, limit=None, key="refresh")

# Page config
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Dashboard (DataOps + ML)")
st.markdown("Automated EDA, ML Evaluation & Monitoring (Auto-refresh every 2 mins)")

# Upload CSV
uploaded_file = st.file_uploader("📂 Choose CSV file", type=["csv"])
use_sample = st.checkbox("Use packaged sample dataset", value=False)

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)

# Buttons for pipelines
run_dataops = st.button("▶️ Run DataOps Pipeline")
run_ml = st.button("▶️ Run ML Pipeline")

# Session state
if "dataops_results" not in st.session_state:
    st.session_state["dataops_results"] = None
if "ml_results" not in st.session_state:
    st.session_state["ml_results"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

def run_dataops_safe(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait...")
        return None
    st.session_state["is_processing"] = True
    try:
        res = run_pipeline(df)
        st.session_state["dataops_results"] = res
        return res
    finally:
        st.session_state["is_processing"] = False

def run_ml_safe(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait...")
        return None
    st.session_state["is_processing"] = True
    try:
        res = run_ml_pipeline(df)
        st.session_state["ml_results"] = res
        return res
    finally:
        st.session_state["is_processing"] = False

# Run pipelines if button clicked
if df is not None:
    if run_dataops:
        run_dataops_safe(df)
    if run_ml:
        run_ml_safe(df)

    # Display DataOps results
    if st.session_state["dataops_results"]:
        dataops_results = st.session_state["dataops_results"]
        st.header("📊 DataOps Pipeline Results")
        st.subheader("Summary Statistics")
        st.dataframe(dataops_results.get("summary", {}))
        st.subheader("Missing Values")
        st.json(dataops_results.get("missing", {}))
        st.subheader("Data Types")
        st.json(dataops_results.get("dtypes", {}))
        st.subheader("Visualizations")
        images = ["correlation_heatmap.png","fraud_vs_nonfraud.png","feature_importance.png"]
        cols = st.columns(2)
        for i,img in enumerate(images):
            p = Path(img)
            if p.exists():
                with cols[i % 2]:
                    st.image(str(p), use_column_width=True)
        st.subheader("Pipeline Logs")
        st.text_area("DataOps Logs", value=dataops_results.get("logs",""), height=200)

    # Display ML results
    if st.session_state["ml_results"]:
        ml_results = st.session_state["ml_results"]
        st.header("🤖 ML Pipeline Results")
        st.subheader("Evaluation Metrics")
        st.json(ml_results.get("metrics", {}))
        st.subheader("ML Pipeline Logs")
        st.text_area("ML Logs", value=ml_results.get("logs",""), height=200)

# Application insights
st.markdown("---")
st.header("🧭 Application Insights")
app_details = {
    "App Name": "Credit Card Fraud Dashboard",
    "Python Version": os.sys.version.split()[0],
    "Platform": platform.system() + " " + platform.release(),
    "Streamlit Version": st.__version__,
    "Current Directory": os.getcwd()
}
for k,v in app_details.items():
    st.write(f"**{k}:** {v}")
