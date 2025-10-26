import streamlit as st
import pandas as pd
from data_pipeline import run_pipeline
from pathlib import Path
st.set_page_config(page_title="Data Pipeline Dashboard", layout="wide")
st.title("📊 Automated Data Pipeline Dashboard")
uploaded_file = st.file_uploader("📂 Choose CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("1️⃣ Data Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
st.subheader("2️⃣ Running Data Pipeline...")
    with st.spinner("Processing..."):
        results = run_pipeline(df)
    st.success("✅ Pipeline Completed!")
st.subheader("3️⃣ Summary Statistics")
    st.dataframe(results["summary"])
st.subheader("4️⃣ Missing Values")
    st.json(results["missing"])
st.subheader("5️⃣ Data Types")
    st.json(results["dtypes"])
st.subheader("6️⃣ Generated Visualizations")
    for img in ["correlation_heatmap.png", "class_distribution.png", "feature_importance.png"]:
        if Path(img).exists():
            st.image(img, use_column_width=True)
st.subheader("7️⃣ Pipeline Logs")
    st.text_area("Logs", results["logs"], height=200)
else:
    st.info("Please upload a CSV file to start the pipeline.")
