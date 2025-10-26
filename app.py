"""
Integrated DataOps + ML Pipeline Streamlit App
Author: Prem Charan
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_autorefresh import st_autorefresh

# Import pipelines
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline

# ---------- Auto-refresh configuration ----------
st_autorefresh(interval=120000, limit=None, key="refresh")  # every 2 minutes

# ---------- Page config ----------
st.set_page_config(page_title="Credit Card Pipelines", layout="wide")
st.title("💳 Credit Card Fraud Dashboard (DataOps + ML)")

# ---------- Session state ----------
if "last_run_dataops" not in st.session_state:
    st.session_state["last_run_dataops"] = None
if "last_run_ml" not in st.session_state:
    st.session_state["last_run_ml"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# ---------- File upload ----------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
use_sample = st.checkbox("Use packaged sample dataset", value=False)

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
        st.error("Sample data missing. Please place sample_data.csv in app folder.")

# ---------- Safe run functions ----------
def safe_run_dataops(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait.")
        return None
    try:
        st.session_state["is_processing"] = True
        results = run_pipeline(df)
        st.session_state["last_run_dataops"] = results.get("run_ts", datetime.now().isoformat())
        return results
    finally:
        st.session_state["is_processing"] = False

def safe_run_ml(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait.")
        return None
    try:
        st.session_state["is_processing"] = True
        results = run_ml_pipeline(df)
        st.session_state["last_run_ml"] = results.get("run_ts", datetime.now().isoformat())
        return results
    finally:
        st.session_state["is_processing"] = False

# ---------- Buttons ----------
col1, col2 = st.columns(2)
run_dataops = col1.button("▶️ Run DataOps Pipeline")
run_ml = col2.button("▶️ Run ML Pipeline")

# ---------- Run pipelines ----------
dataops_results = None
ml_results = None

if df is not None:
    if run_dataops:
        with st.spinner("Running DataOps pipeline..."):
            dataops_results = safe_run_dataops(df)
    if run_ml:
        with st.spinner("Running ML pipeline..."):
            ml_results = safe_run_ml(df)
else:
    st.info("Please upload a CSV file or enable 'Use packaged sample dataset'.")

# ---------- Display DataOps results ----------
if dataops_results:
    if "error" in dataops_results:
        st.error(f"DataOps pipeline error: {dataops_results['error']}")
    else:
        st.header("📊 DataOps Pipeline Results")
        # 1️⃣ Summary
        st.subheader("1️⃣ Summary Statistics")
        try:
            st.dataframe(pd.DataFrame(dataops_results["summary"]))
        except:
            st.write(dataops_results.get("summary"))

        # 2️⃣ Missing
        st.subheader("2️⃣ Missing Values")
        st.json(dataops_results.get("missing", {}))

        # 3️⃣ Data types
        st.subheader("3️⃣ Data Types")
        st.json(dataops_results.get("dtypes", {}))

        # 4️⃣ Charts
        st.subheader("4️⃣ Charts & Plots")
        images = ["correlation_heatmap.png", "class_distribution.png",
                  "feature_importance.png", "univariate_plot.png", "bivariate_plot.png"]
        cols = st.columns(2)
        for i, img in enumerate(images):
            p = Path(img)
            if p.exists():
                with cols[i % 2]:
                    st.image(str(p), use_column_width=True, caption=img)
            else:
                with cols[i % 2]:
                    st.write(f"ℹ️ {img} not available for this run.")

        # 5️⃣ Logs
        st.subheader("5️⃣ DataOps Logs")
        st.text_area("📜 data_pipeline.log", value=dataops_results.get("logs", ""), height=240)

# ---------- Display ML results ----------
if ml_results:
    if "error" in ml_results:
        st.error(f"ML pipeline error: {ml_results['error']}")
    else:
        st.header("🤖 ML Pipeline Results")
        st.subheader("Selected Models: Logistic Regression & Random Forest")
        st.write("📊 Evaluation Metrics:")
        st.json(ml_results.get("metrics", {}))
        st.subheader("ML Logs")
        st.text_area("📜 ML Pipeline Logs", value=ml_results.get("logs", ""), height=240)

# ---------- Application Insights ----------
st.markdown("---")
st.header("🧭 Application Insights")
import platform, sys
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
