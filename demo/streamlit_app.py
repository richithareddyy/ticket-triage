"""Entry point for Streamlit Community Cloud (free hosting, single process).

Runs the Streamlit UI with the prediction API loaded in-process, so no separate
API server is needed. Main file path on Streamlit Cloud: demo/streamlit_app.py
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MODEL_PATH", os.path.join(ROOT, "artifacts", "model.joblib"))
os.environ.setdefault("METRICS_PATH", os.path.join(ROOT, "artifacts", "metrics.json"))

runpy.run_path(os.path.join(ROOT, "ui", "streamlit_app.py"), run_name="__main__")
