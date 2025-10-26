# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from data_pipeline import run_pipeline


st.set_page_config(page_title="Data Pipeline Dashboard", layout="wide")
st.title("📊 Automated Data Pipeline — Upload CSV")

st.write("Upload a CSV file (e.g., creditcard.csv). The app will run preprocessing, EDA, log actions, and show visuals.")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

    if st.button("Run pipeline now"):
        with st.spinner("Running pipeline..."):
            out = run_pipeline(df)
        if out.get("status") == "success":
            st.success("Pipeline completed successfully")
        else:
            st.error("Pipeline failed - check logs")

        st.subheader("Summary Statistics")
        st.dataframe(out.get("summary"))

        st.subheader("Missing Values")
        st.json(out.get("missing"))

        st.subheader("Data Types")
        st.json(out.get("dtypes"))

        st.subheader("Generated Plots")
        for fn in ["class_distribution.png", "correlation_heatmap.png", "feature_importance.png"]:
            if Path(fn).exists():
                st.image(fn, use_column_width=True)
            else:
                st.info(f"{fn} not generated")

        st.subheader("Pipeline Logs (tail)")
        st.text_area("Logs", out.get("logs", ""), height=300)
    else:
        st.info("Click 'Run pipeline now' to execute preprocessing and EDA.")
else:
    st.info("Please upload a CSV to proceed.")
