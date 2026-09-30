#!/usr/bin/env bash
# Runs the API (internal only) and the Streamlit UI (public) in one container.
set -euo pipefail

gunicorn --bind 127.0.0.1:5000 --workers 1 --threads 4 --timeout 60 api.wsgi:app &
API_PID=$!

# The API warms up its models before accepting requests; wait so the UI doesn't start disconnected.
for _ in $(seq 1 60); do
  curl -fs http://127.0.0.1:5000/health >/dev/null 2>&1 && break
  kill -0 "$API_PID" 2>/dev/null || { echo "API failed to start" >&2; exit 1; }
  sleep 1
done

exec streamlit run ui/streamlit_app.py --server.port="${PORT:-7860}" --server.address=0.0.0.0 \
  --server.headless=true --browser.gatherUsageStats=false --client.toolbarMode=viewer
