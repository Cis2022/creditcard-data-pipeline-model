# from flask import Flask, jsonify
# import os
# import platform
# import pandas as pd
# import json

# app = Flask(__name__)

# # Load sample data (or replace with actual data loading logic)
# DATA_PATH = "sample_data.csv"
# df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()

# # Dummy model metrics (replace with actual metrics from ml_pipeline)
# model_metrics = {
#     "Logistic Regression": {
#         "Accuracy": 0.9389,
#         "Precision": 0.92,
#         "Recall": 0.91,
#         "F1 Score": 0.915
#     },
#     "Random Forest": {
#         "Accuracy": 0.6032,
#         "Precision": 0.60,
#         "Recall": 0.59,
#         "F1 Score": 0.595
#     }
# }

# # ---------- Endpoint 1: Application Info ----------
# @app.route("/api/info", methods=["GET"])
# def app_info():
#     info = {
#         "App Name": "Credit Card Fraud Detection",
#         "Version": "1.0",
#         "Python": platform.python_version(),
#         "Platform": platform.system() + " " + platform.release(),
#         "Deployment": os.getenv("DEPLOY_ENV", "Local")
#     }
#     return jsonify(info)

# # ---------- Endpoint 2: Model Performance ----------
# @app.route("/api/metrics", methods=["GET"])
# def get_metrics():
#     return jsonify(model_metrics)

# # ---------- Endpoint 3: Dataset Info ----------
# @app.route("/api/dataset", methods=["GET"])
# def dataset_info():
#     stats = {
#         "Shape": df.shape,
#         "Columns": list(df.columns),
#         "Missing Values": df.isnull().sum().to_dict()
#     }
#     return jsonify(stats)

# # ---------- Endpoint 4: Pipeline Status ----------
# @app.route("/api/pipeline/status", methods=["GET"])
# def pipeline_status():
#     status = {
#         "Data Pipeline": "Completed",
#         "ML Pipeline": "Completed",
#         "Last Run": "2025-10-26 01:20:00",
#         "Auto-Refresh": "Every 2 minutes"
#     }
#     return jsonify(status)

# # ---------- Endpoint 5: Models List ----------
# @app.route("/api/models", methods=["GET"])
# def models_list():
#     return jsonify(list(model_metrics.keys()))

# # ---------- Endpoint 6: Health Check ----------
# @app.route("/api/health", methods=["GET"])
# def health_check():
#     return jsonify({"status": "OK", "message": "API is running"})

# # ---------- Endpoint 8: Home/Docs ----------
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "Welcome to Credit Card Fraud Detection API",
#         "endpoints": [
#             "/api/info", "/api/metrics", "/api/dataset",
#             "/api/pipeline/status", "/api/models", "/api/health"
#         ]
#     })

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, jsonify
from data_pipeline import run_pipeline
from ml_pipeline import run_ml_pipeline
import pandas as pd
import os
import platform

app = Flask(__name__)

# Load latest uploaded dataset
DATA_PATH = "latest_uploaded.csv"
df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()

# Run pipelines once at startup
pipeline_result = run_pipeline(df) if not df.empty else {}
ml_result = run_ml_pipeline(df) if not df.empty else {}

# ---------- Endpoint 1: Application Info ----------
@app.route("/api/info", methods=["GET"])
def app_info():
    info = {
        "App Name": "Credit Card Fraud Detection",
        "Version": "1.0",
        "Python": platform.python_version(),
        "Platform": platform.system() + " " + platform.release(),
        "Deployment": os.getenv("DEPLOY_ENV", "Local")
    }
    return jsonify(info)

# ---------- Endpoint 2: Model Performance ----------
@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(ml_result.get("metrics", {}))

# ---------- Endpoint 3: Dataset Info ----------
@app.route("/api/dataset", methods=["GET"])
def dataset_info():
    stats = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict()
    }
    return jsonify(stats)

# ---------- Endpoint 4: Pipeline Status ----------
@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    status = {
        "Data Pipeline": "Completed" if pipeline_result else "Not Run",
        "ML Pipeline": "Completed" if ml_result else "Not Run",
        "Last Run": pipeline_result.get("run_ts", "N/A"),
        "Auto-Refresh": "Every 2 minutes"
    }
    return jsonify(status)

# ---------- Endpoint 5: Models List ----------
@app.route("/api/models", methods=["GET"])
def models_list():
    return jsonify(list(ml_result.get("metrics", {}).keys()))

# ---------- Endpoint 6: Health Check ----------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "API is running"})

# ---------- Endpoint 7: Logs ----------
@app.route("/api/logs", methods=["GET"])
def logs():
    return jsonify({
        "Data Pipeline Logs": pipeline_result.get("logs", "")[-1000:] if pipeline_result else "No logs",
        "ML Pipeline Logs": ml_result.get("logs", "")[-1000:] if ml_result else "No logs"
    })

# ---------- Endpoint 8: Home/Docs ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to Credit Card Fraud Detection API",
        "endpoints": [
            "/api/info", "/api/metrics", "/api/dataset",
            "/api/pipeline/status", "/api/models", "/api/health", "/api/logs"
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)