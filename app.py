import streamlit as st
import pandas as pd
from data_pipeline import run_pipeline
from pathlib import Path
import glob
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 2 minutes ---
st_autorefresh(interval=120000, limit=None, key="refresh")

# --- Page setup ---
st.set_page_config(page_title="Credit Card Fraud DataOps", layout="wide")
st.title("💳 Credit Card Fraud DataOps Dashboard")

# --- File uploader ---
uploaded_file = st.file_uploader("📂 Upload your creditcard.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("1️⃣ Data Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")

    st.subheader("2️⃣ Running Data Pipeline...")
    with st.spinner("Processing data..."):
        results = run_pipeline(df)
    st.success("✅ Pipeline Completed Successfully!")

    st.subheader("3️⃣ Summary Statistics")
    st.dataframe(results["summary"])

    st.subheader("4️⃣ Missing Values & Data Types")
    col1, col2 = st.columns(2)
    with col1:
        st.json(results["missing"])
    with col2:
        st.json(results["dtypes"])

    st.subheader("5️⃣ Univariate Analysis")
    for img in sorted(glob.glob("plots/univariate/*.png")):
        st.image(img, caption=Path(img).name, use_container_width=True)

    st.subheader("6️⃣ Bivariate Analysis")
    for img in sorted(glob.glob("plots/bivariate/*.png")):
        st.image(img, caption=Path(img).name, use_container_width=True)

    st.subheader("7️⃣ Pipeline Logs")
    st.text_area("Logs", results["logs"], height=250)

else:
    st.info("Please upload the CSV file to start the automated DataOps pipeline.")
