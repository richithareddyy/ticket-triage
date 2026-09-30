# syntax=docker/dockerfile:1
# Targets:
#   api   - Flask REST API served by gunicorn (default)
#   ui    - Streamlit front end
#   space - API + UI in one container (single-service Docker hosts)
#   docker build --target api -t ticket-triage-api .
#   docker build --target ui  -t ticket-triage-ui .

FROM python:3.11-slim-bookworm AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/app/nltk_data MODEL_PATH=/app/artifacts/model.joblib METRICS_PATH=/app/artifacts/metrics.json
WORKDIR /app
# xgboost wheels need the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY triage/ triage/
RUN python -c "from triage.text import ensure_nltk; ensure_nltk()"
COPY api/ api/
COPY artifacts/model.joblib artifacts/metrics.json artifacts/
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fs http://localhost:5000/health || exit 1
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "60", \
     "--access-logfile", "-", "api.wsgi:app"]

FROM python:3.11-slim-bookworm AS ui
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 API_URL=http://api:5000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt
COPY .streamlit/ .streamlit/
COPY ui/ ui/
RUN useradd --create-home appuser
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false", "--client.toolbarMode=viewer"]

# Single-container image (API + UI) for hosts that run one Docker service.
#   docker build --target space -t ticket-triage-space . && docker run -p 7860:7860 ticket-triage-space
FROM api AS space
USER root
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt
COPY .streamlit/ .streamlit/
COPY ui/ ui/
COPY deploy/start-space.sh /app/start-space.sh
RUN chmod +x /app/start-space.sh && chown -R appuser /app
USER appuser
ENV PORT=7860 API_URL=http://127.0.0.1:5000
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:7860/_stcore/health || exit 1
CMD ["/app/start-space.sh"]
