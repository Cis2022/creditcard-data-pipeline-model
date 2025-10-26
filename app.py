import streamlit as st
import time
from data_pipeline import run_dataops_pipeline
from ml_pipeline import load_preprocessed_data, train_models
from PIL import Image
import os

st.set_page_config(page_title="Credit Card Fraud DataOps & MLOps", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")

st.markdown("""
This dashboard runs **DataOps Pipeline** (Sub-Objective 1) and **MLOps Pipeline** (Sub-Objective 2)
with auto-refresh every 2 minutes.
""")

# Auto-refresh every 2 minutes
st_autorefresh = st.experimental_rerun

def display_images(pattern):
    import glob
    images = glob.glob(pattern)
    for img_path in images:
        image = Image.open(img_path)
        st.image(image, caption=os.path.basename(img_path))

# DataOps pipeline button
if st.button("Run DataOps Pipeline"):
    with st.spinner("Running DataOps Pipeline..."):
        df, log = run_dataops_pipeline()
        st.success("DataOps Pipeline Completed!")
        st.write(log)
        display_images("univariate_*.png")
        display_images("bivariate_*.png")
        if os.path.exists("fraud_vs_nonfraud.png"):
            st.image("fraud_vs_nonfraud.png", caption="Fraud vs Non-Fraud")

# MLOps pipeline button
if st.button("Run MLOps Pipeline"):
    with st.spinner("Running MLOps Pipeline..."):
        df = load_preprocessed_data()
        metrics_log = train_models(df)
        st.success("MLOps Pipeline Completed!")
        st.write(metrics_log)

# Auto-refresh every 2 minutes
st_autorefresh(interval=120000)  # 120000 ms = 2 minutes
