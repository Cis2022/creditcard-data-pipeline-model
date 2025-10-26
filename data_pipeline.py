import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Load dataset
def load_data(file_path="creditcard.csv"):
    df = pd.read_csv(file_path)
    return df

# Preprocessing
def preprocess_data(df):
    # Summary stats
    summary = df.describe()
    
    # Missing values
    missing = df.isnull().sum()
    
    # Impute missing numeric data
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Data types
    dtypes = df.dtypes
    
    # Normalization
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, summary, missing, dtypes

# EDA
def perform_eda(df):
    correlation = df.corr()
    
    # Univariate plots
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"univariate_{col}.png")
        plt.close()
    
    # Bivariate plots (top 5 correlated features)
    corr_target = correlation['Class'].abs().sort_values(ascending=False)[1:6]
    for col in corr_target.index:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df['Class'])
        plt.title(f"{col} vs Class")
        plt.tight_layout()
        plt.savefig(f"bivariate_{col}.png")
        plt.close()
    
    # Fraud vs Non-Fraud
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=df)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.tight_layout()
    plt.savefig("fraud_vs_nonfraud.png")
    plt.close()
    
    return correlation

# DataOps workflow
def run_dataops_pipeline():
    df = load_data()
    df, summary, missing, dtypes = preprocess_data(df)
    correlation = perform_eda(df)
    
    log = f"""
    DataOps Pipeline Run:
    - Shape: {df.shape}
    - Summary Stats: {summary.head().to_dict()}
    - Missing Values: {missing.to_dict()}
    - Data Types: {dtypes.to_dict()}
    """
    return df, log
