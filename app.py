"""
app.py
Streamlit front-end for Credit Card Fraud DataOps Dashboard
Features:
- Auto-refresh every 2 minutes
- Upload CSV or use sample_data.csv
- Runs data_pipeline.run_pipeline and displays results, charts, logs, insights
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from data_pipeline import run_pipeline
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime

# ---------- Auto-refresh configuration ----------
# Re-run script every 120000 ms (2 minutes)
st_autorefresh(interval=120000, limit=None, key="refresh")

# ---------- Page config ----------
st.set_page_config(page_title="Credit Card Data Pipeline", layout="wide")
st.title("💳 Credit Card Fraud DataOps Dashboard")
st.markdown("Automated EDA, Normalization, Feature Importance, and Monitoring (auto-refresh every 2 minutes)")

# ---------- last run tracker ----------
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

if st.session_state["last_run"]:
    st.info(f"🔁 Last pipeline run: {st.session_state['last_run']}")

# ---------- File upload and sample option ----------
uploaded_file = st.file_uploader("📂 Choose CSV file (creditcard.csv) to run pipeline", type=["csv"])
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

# ---------- Run pipeline safely (prevents overlapping runs) ----------
def safe_run(df):
    if st.session_state["is_processing"]:
        st.warning("Pipeline already running — please wait for it to finish.")
        return None
    try:
        st.session_state["is_processing"] = True
        results = run_pipeline(df)
        # update last run timestamp from pipeline result if present
        run_ts = results.get("run_ts") if isinstance(results, dict) else None
        st.session_state["last_run"] = run_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return results
    finally:
        st.session_state["is_processing"] = False

# Manual run button
run_now = st.button("▶️ Run pipeline now (manual)")

# ---------- Main UI flow ----------
if df is not None:
    st.header("1️⃣ Data Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

    # Decide whether to run:
    should_run = False
    # 1) manual button clicked
    if run_now:
        should_run = True

    # 2) if last_run is None -> run
    if st.session_state["last_run"] is None:
        should_run = True
    else:
        # If last run older than 110 seconds, allow auto-run (auto-refresh every 120s)
        try:
            last = datetime.fromisoformat(st.session_state["last_run"]) if "T" in st.session_state["last_run"] else datetime.strptime(st.session_state["last_run"], "%Y-%m-%d %H:%M:%S")
            delta = (datetime.now() - last).total_seconds()
            if delta >= 110:
                should_run = True
        except Exception:
            # if parsing fails, just allow run
            should_run = True

    if should_run:
        with st.spinner("Processing pipeline..."):
            results = safe_run(df)
    else:
        results = None
        st.info("Waiting for scheduled run (or click 'Run pipeline now').")

    # ---------- Display pipeline results ----------
    if results is None:
        st.warning("No results to show right now.")
    else:
        if "error" in results:
            st.error(f"Pipeline error: {results['error']}")
        else:
            st.header("2️⃣ Summary Statistics")
            # results["summary"] may be a dict (from df.describe().to_dict()); convert when convenient
            try:
                summary_df = pd.DataFrame(results["summary"])
                # transposed summary sometimes easier to read
                st.dataframe(summary_df)
            except Exception:
                st.write(results.get("summary"))

            st.header("3️⃣ Missing Values")
            st.json(results.get("missing", {}))

            st.header("4️⃣ Data Types")
            st.json(results.get("dtypes", {}))

            st.header("5️⃣ Generated Visualizations (saved as images)")
            images = ["correlation_heatmap.png", "class_distribution.png", "feature_importance.png", "univariate_plot.png", "bivariate_plot.png"]
            cols = st.columns(2)
            for i, img in enumerate(images):
                p = Path(img)
                if p.exists():
                    with cols[i % 2]:
                        st.image(str(p), use_column_width=True, caption=img)
                else:
                    with cols[i % 2]:
                        st.write(f"ℹ️ {img} not available for this run.")

            st.header("6️⃣ Pipeline Logs")
            st.text_area("📜 data_pipeline.log", value=results.get("logs", ""), height=240)

            st.markdown("---")
            st.header("🧭 Application Insights")
            import platform, sys
            app_details = {
                "Application Name": "Credit Card Data Pipeline Dashboard",
                "Deployed Environment": os.getenv("STREAMLIT_SERVER_ADDRESS", "Local/Streamlit Cloud"),
                "Python Version": sys.version.split()[0],
                "Platform": platform.system() + " " + platform.release(),
                "Streamlit Version": st.__version__,
                "Current Working Directory": os.getcwd(),
                "Last Pipeline Run": results.get("run_ts", st.session_state["last_run"])
            }
            for k, v in app_details.items():
                st.write(f"**{k}:** {v}")

else:
    st.info("Please upload a CSV file or enable 'Use packaged sample dataset' to run the demo.")

# ---------- Footer ----------
st.markdown("---")
st.caption("DataOps: This app runs the pipeline automatically every 2 minutes (auto-refresh).")



