# app.py
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import pandas as pd
import altair as alt
import streamlit as st

# Local modules
from data_pipeline import run_dataops_pipeline, DATA_PATH, LOGS_DIR
from ml_pipeline import run_mlops_pipeline, METRICS_DIR

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Credit Card Fraud — DataOps & MLOps",
    page_icon="💳",
    layout="wide",
)

# --- Constants ---
AUTO_REFRESH_INTERVAL_MS = 120_000  # 2 minutes

# --- Session State Initialization ---
if "last_dataops_run" not in st.session_state:
    st.session_state.last_dataops_run = None
if "last_mlops_run" not in st.session_state:
    st.session_state.last_mlops_run = None
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True  # default ON

# --- Header ---
st.title("💳 Credit Card Fraud — Data & ML Pipelines")
st.caption(
    "Two pipelines: DataOps (EDA & preprocessing) and MLOps (model training & monitoring). "
    "Runs auto every 2 minutes (toggle below)."
)

# --- Auto Refresh Toggle ---
col0 = st.columns(3)
with col0[0]:
    st.toggle(
        "Auto-refresh every 2 minutes",
        value=st.session_state.auto_refresh_enabled,
        key="auto_refresh_enabled",
        help="When ON, the app reruns every 2 minutes and schedules pipelines if due.",
    )

if st.session_state.auto_refresh_enabled:
    st.autorefresh(interval=AUTO_REFRESH_INTERVAL_MS, key="global_autorefresh")

# --- Dataset presence check ---
DATA_PATH.mkdir(parents=True, exist_ok=True)
if not Path(DATA_PATH).exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`. "
        "Please add `creditcard.csv` to the `data/` folder in the repository."
    )
    st.stop()

# --- Scheduling helper ---
def should_run(last_run_ts: datetime | None, every_minutes: int = 2) -> bool:
    if last_run_ts is None:
        return True
    return datetime.utcnow() - last_run_ts >= timedelta(minutes=every_minutes)

# --- Controls ---
st.sidebar.header("Controls")
run_dataops_click = st.sidebar.button("▶ Run DataOps Pipeline (Sub-Objective 1)")
run_mlops_click = st.sidebar.button("▶ Run ML Pipeline (Sub-Objective 2)")

# --- Main layout ---
tab1, tab2, tab3 = st.tabs(["📊 DataOps", "🤖 MLOps", "🪵 Logs"])

# =========================
# Tab 1: DataOps Pipeline
# =========================
with tab1:
    st.subheader("DataOps Pipeline (Preprocessing + EDA + Artifacts)")
    next_run_info = (
        "next scheduled run in 2 minutes"
        if st.session_state.last_dataops_run
        else "first run pending"
    )
    st.caption(
        f"Last run: {st.session_state.last_dataops_run or '—'} | {next_run_info}"
    )

    # Auto schedule if due
    if st.session_state.auto_refresh_enabled and should_run(st.session_state.last_dataops_run):
        with st.spinner("Auto-running DataOps pipeline…"):
            dataops_result = run_dataops_pipeline()
            st.session_state.last_dataops_run = datetime.utcnow()
    elif run_dataops_click:
        with st.spinner("Running DataOps pipeline…"):
            dataops_result = run_dataops_pipeline()
            st.session_state.last_dataops_run = datetime.utcnow()
    else:
        dataops_result = None

    if dataops_result:
        st.success("DataOps pipeline completed ✅")

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{dataops_result['summary']['rows']:,}")
        with c2:
            st.metric("Columns", f"{dataops_result['summary']['cols']}")
        with c3:
            st.metric("Missing Values", f"{dataops_result['summary']['missing_total']:,}")
        with c4:
            st.metric("Imputed Columns", ", ".join(dataops_result['summary']['imputed_cols']) or "None")

        # Data types
        st.markdown("### Data Types")
        st.dataframe(pd.DataFrame({
            "column": list(dataops_result['dtypes'].keys()),
            "dtype": list(dataops_result['dtypes'].values())
        }))

        # Normalization info
        st.markdown("### Normalization (StandardScaler)")
        st.write("Scaled numeric columns:", ", ".join(dataops_result['normalized']['scaled_cols']))

        # EDA visualizations
        st.markdown("### EDA Visualizations")
        vcols = st.columns(2)
        with vcols[0]:
            st.image(str(dataops_result['artifacts']['eda']['class_distribution']), caption="Class Distribution (Imbalanced)")
            st.image(str(dataops_result['artifacts']['eda']['amount_hist']), caption="Amount Distribution")
        with vcols[1]:
            st.image(str(dataops_result['artifacts']['eda']['corr_heatmap']), caption="Correlation Heatmap (sampled)")
            st.image(str(dataops_result['artifacts']['eda']['time_amount_scatter']), caption="Time vs Amount (colored by Class)")

        # EDA report
        st.markdown("### EDA Report (JSON)")
        with open(dataops_result['artifacts']['eda_report_path'], "r") as f:
            eda_json = json.load(f)
        st.json(eda_json)

        # Processed data sample
        st.markdown("### Processed Data Preview")
        processed_path = dataops_result['artifacts']['processed_csv']
        df_preview = pd.read_csv(processed_path).head(100)
        st.dataframe(df_preview)

# =========================
# Tab 2: MLOps Pipeline
# =========================
with tab2:
    st.subheader("ML Pipeline (Train + Evaluate + Monitor)")

    next_run_info_ml = (
        "next scheduled run in 2 minutes"
        if st.session_state.last_mlops_run
        else "first run pending"
    )
    st.caption(
        f"Last run: {st.session_state.last_mlops_run or '—'} | {next_run_info_ml}"
    )

    # Auto schedule if due
    if st.session_state.auto_refresh_enabled and should_run(st.session_state.last_mlops_run):
        with st.spinner("Auto-running ML pipeline…"):
            ml_result = run_mlops_pipeline()
            st.session_state.last_mlops_run = datetime.utcnow()
    elif run_mlops_click:
        with st.spinner("Running ML pipeline…"):
            ml_result = run_mlops_pipeline()
            st.session_state.last_mlops_run = datetime.utcnow()
    else:
        ml_result = None

    if ml_result:
        st.success("ML pipeline completed ✅")

        # Show metrics for both models
        st.markdown("### Latest Metrics")
        for model_name, metrics in ml_result['metrics'].items():
            st.write(f"**{model_name}**")
            cols = st.columns(5)
            cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
            cols[1].metric("Precision", f"{metrics['precision']:.4f}")
            cols[2].metric("Recall", f"{metrics['recall']:.4f}")
            cols[3].metric("F1 Score", f"{metrics['f1']:.4f}")
            cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            st.caption(f"Confusion Matrix: {metrics['confusion_matrix']}")
            st.divider()

        # Plots
        pcols = st.columns(2)
        with pcols[0]:
            st.image(str(ml_result['artifacts']['roc_curve']), caption="ROC Curves")
        with pcols[1]:
            st.image(str(ml_result['artifacts']['rf_feature_importance']), caption="Random Forest Feature Importance")

        # Metrics trend (history)
        st.markdown("### MLOps Monitoring — Metrics History")
        hist_csv = Path(METRICS_DIR) / "ml_metrics_history.csv"
        if hist_csv.exists():
            hist_df = pd.read_csv(hist_csv)
            # line chart for F1 over time
            line = alt.Chart(hist_df).mark_line(point=True).encode(
                x="timestamp:T",
                y="f1:Q",
                color="model:N",
                tooltip=["timestamp:T", "model:N", "accuracy:Q", "precision:Q", "recall:Q", "f1:Q", "roc_auc:Q"]
            ).properties(height=300)
            st.altair_chart(line, use_container_width=True)

            st.dataframe(hist_df.sort_values("timestamp", ascending=False))
        else:
            st.info("No metrics history found yet. Run the ML pipeline to start logging.")

# =========================
# Tab 3: Logs
# =========================
with tab3:
    st.subheader("Pipeline Logs")
    dataops_log = Path(LOGS_DIR) / "dataops.log"
    mlops_log = Path(LOGS_DIR) / "mlops.log"

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**DataOps Log**")
        if dataops_log.exists():
            st.code(dataops_log.read_text()[-20_000:])  # tail
        else:
            st.info("No DataOps log yet.")

    with cols[1]:
        st.markdown("**MLOps Log**")
        if mlops_log.exists():
            st.code(mlops_log.read_text()[-20_000:])
        else:
            st.info("No MLOps log yet.")

st.caption("© Your Project — Streamlit Cloud deployment ready.")
``