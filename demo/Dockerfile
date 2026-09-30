# Lean single-process demo image (Streamlit UI with the API loaded in-process).
# Fits small free instances (peaks at ~450 MB under a 512 MB limit).
#   docker build -f demo/Dockerfile -t ticket-triage-demo . && docker run -p 10000:10000 ticket-triage-demo
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MALLOC_ARENA_MAX=2 NLTK_DATA=/app/nltk_data PORT=10000
WORKDIR /app
COPY demo/requirements.txt demo/requirements.txt
RUN pip install -r demo/requirements.txt
COPY triage/ triage/
RUN python -c "from triage.text import ensure_nltk; ensure_nltk()"
COPY api/ api/
COPY ui/ ui/
COPY demo/streamlit_app.py demo/streamlit_app.py
COPY .streamlit/ .streamlit/
COPY artifacts/model.joblib artifacts/metrics.json artifacts/
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 10000
CMD streamlit run demo/streamlit_app.py --server.port="$PORT" --server.address=0.0.0.0 --server.headless=true
