"""
app.py
Streamlit Dashboard for Credit Card Fraud
Features:
- DataOps Pipeline: EDA, Normalization, Feature Importance, Charts
- ML Pipeline: Logistic Regression & Random Forest, Metrics & Logs
- Auto-refresh every 2 minutes
- Two buttons for manual execution
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime
import platform
import sys

# ---------- Auto-refresh every 2 minutes ----------
st_autorefresh(interval=120000, limit=None, key="refresh")

# ---------- Page config ----------
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Dashboard")
st.markdown("DataOps + ML Pipeline | Auto-refresh every 2 minutes")

# ---------- Session state ----------
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# ---------- File upload ----------
uploaded_file = st.file_uploader("📂 Choose CSV file (creditcard.csv)", type=["csv"])
use_sample = st.checkbox("Use sample dataset (for demo)", value=False)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded uploaded file: {getattr(uploaded_file, 'name', 'uploaded')} | Shape: {df.shape}")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_sample:
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.success(f"Loaded sample_data.csv | Shape: {df.shape}")
    else:
        st.error("Sample data missing. Add sample_data.csv in app folder.")

# ---------- Run pipelines safely ----------
def safe_run_dataops(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait.")
        return None
    try:
        st.session_state["is_processing"] = True
        results = run_pipeline(df)
        run_ts = results.get("run_ts") if isinstance(results, dict) else None
        st.session_state["last_run"] = run_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return results
    finally:
        st.session_state["is_processing"] = False

# ---------- Buttons ----------
col1, col2 = st.columns(2)
run_dataops = col1.button("▶️ Run DataOps Pipeline")
run_ml = col2.button("▶️ Run ML Pipeline")

# ---------- Main ----------
results = None
ml_results = None

if df is not None:
    st.header("1️⃣ Data Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

    should_run_dataops = run_dataops or (st.session_state["last_run"] is None)
    if should_run_dataops:
        with st.spinner("Running DataOps pipeline..."):
            results = safe_run_dataops(df)

    # Run ML pipeline if button clicked or after DataOps run
    if run_ml or (results is not None and "error" not in results):
        if "Class" not in df.columns:
            st.warning("ML pipeline cannot run: 'Class' column missing in dataset.")
        else:
            with st.spinner("Running ML pipeline..."):
                ml_results = run_ml_pipeline(df)

    # ---------- Display DataOps ----------
    if results:
        if "error" in results:
            st.error(f"DataOps pipeline error: {results['error']}")
        else:
            st.header("2️⃣ Summary Statistics")
            try:
                st.dataframe(pd.DataFrame(results["summary"]))
            except Exception:
                st.write(results.get("summary"))

            st.header("3️⃣ Missing Values")
            st.json(results.get("missing", {}))

            st.header("4️⃣ Data Types")
            st.json(results.get("dtypes", {}))

            st.header("5️⃣ Charts / Visualizations")
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

            st.header("6️⃣ DataOps Logs")
            st.text_area("📜 data_pipeline.log", value=results.get("logs",""), height=240)

    # ---------- Display ML ----------
    if ml_results:
        if "error" in ml_results:
            st.error(f"ML pipeline error: {ml_results['error']}")
        else:
            st.header("7️⃣ ML Pipeline Evaluation")
            st.subheader("Selected Models: Logistic Regression & Random Forest")
            st.write("📊 Evaluation Metrics:")
            st.json(ml_results.get("metrics", {}))

            st.subheader("📁 ML Pipeline Logs")
            st.text_area("ML Pipeline Logs", value=ml_results.get("logs",""), height=240)

    # ---------- Application Insights ----------
    st.markdown("---")
    st.header("🧭 Application Insights")
    app_details = {
        "Application Name": "Credit Card Fraud Dashboard",
        "Deployed Environment": os.getenv("STREAMLIT_SERVER_ADDRESS","Local/Streamlit Cloud"),
        "Python Version": sys.version.split()[0],
        "Platform": platform.system() + " " + platform.release(),
        "Streamlit Version": st.__version__,
        "Current Working Directory": os.getcwd(),
        "Last DataOps Run": results.get("run_ts") if results else None,
        "Last ML Run": ml_results.get("run_ts") if ml_results else None
    }
    for k,v in app_details.items():
        st.write(f"**{k}:** {v}")

else:
    st.info("Please upload a CSV file or enable 'Use sample dataset' to run the dashboard.")
