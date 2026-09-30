"""Flask REST API for ticket triage.

    GET  /health            liveness + loaded model info
    GET  /model             test-set metrics and global SHAP importance
    POST /predict           one ticket object, or {"tickets": [...]} (max 100)
                            ?explain=true adds SHAP contributions
"""
import json
import logging
import os
import time

import pandas as pd
from flask import Flask, jsonify, request

from triage.predict import Predictor

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
METRICS_PATH = os.environ.get("METRICS_PATH", "artifacts/metrics.json")
MAX_BATCH = 100
MAX_TEXT = 10_000
ENUMS = {
    "channel": {"email", "web", "chat", "phone"},
    "customer_tier": {"free", "pro", "enterprise"},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("triage-api")


def validate_ticket(t, i):
    errors = []
    if not isinstance(t, dict):
        return [f"tickets[{i}]: must be an object"]
    for field in ("subject", "description"):
        if not isinstance(t.get(field), str):
            errors.append(f"tickets[{i}].{field}: required string")
        elif len(t[field]) > MAX_TEXT:
            errors.append(f"tickets[{i}].{field}: longer than {MAX_TEXT} characters")
    if isinstance(t.get("subject"), str) and isinstance(t.get("description"), str) \
            and not (t["subject"].strip() or t["description"].strip()):
        errors.append(f"tickets[{i}]: subject and description are both empty")
    for field in ("channel", "customer_tier", "product"):
        if not isinstance(t.get(field), str):
            errors.append(f"tickets[{i}].{field}: required string")
    for field, allowed in ENUMS.items():
        if isinstance(t.get(field), str) and t[field].lower() not in allowed:
            errors.append(f"tickets[{i}].{field}: must be one of {sorted(allowed)}")
    for field in ("prior_tickets_30d", "attachments"):
        v = t.get(field, 0)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            errors.append(f"tickets[{i}].{field}: must be a non-negative integer")
    if "created_at" in t:
        try:
            pd.Timestamp(t["created_at"])
        except (ValueError, TypeError):
            errors.append(f"tickets[{i}].created_at: must be an ISO-8601 timestamp")
    return errors


def create_app(predictor=None):
    app = Flask(__name__)
    app.json.sort_keys = False
    predictor = predictor or Predictor(MODEL_PATH)
    # Warm up lazy loads (WordNet, VADER lexicon, SHAP explainers) before serving traffic.
    predictor.predict([{"subject": "warmup", "description": "warmup request", "channel": "web",
                        "product": "API", "customer_tier": "free"}], explain=True)
    log.info("loaded model %s", predictor.info["models"])

    @app.get("/health")
    def health():
        return jsonify(status="ok", **predictor.info)

    @app.get("/model")
    def model_info():
        if not os.path.exists(METRICS_PATH):
            return jsonify(error="metrics not available"), 404
        with open(METRICS_PATH) as f:
            m = json.load(f)
        keep = ("selected_model", "final_test", "comparison", "shap_importance")
        return jsonify(priority={k: m["priority"][k] for k in keep + ("confusion_matrix",)},
                       resolution_time={k: m["resolution_time"][k] for k in keep})

    @app.post("/predict")
    def predict():
        body = request.get_json(silent=True)
        if body is None:
            return jsonify(error="request body must be JSON"), 400
        batch = isinstance(body, dict) and "tickets" in body
        tickets = body["tickets"] if batch else [body]
        if not isinstance(tickets, list) or not tickets:
            return jsonify(error="'tickets' must be a non-empty list"), 400
        if len(tickets) > MAX_BATCH:
            return jsonify(error=f"at most {MAX_BATCH} tickets per request"), 400
        errors = [e for i, t in enumerate(tickets) for e in validate_ticket(t, i)]
        if errors:
            return jsonify(error="validation failed", details=errors), 400

        explain = request.args.get("explain", "false").lower() in ("1", "true", "yes")
        start = time.perf_counter()
        results = predictor.predict(tickets, explain=explain)
        ms = round((time.perf_counter() - start) * 1000, 1)
        log.info("scored %d ticket(s) explain=%s in %sms", len(tickets), explain, ms)
        for t, r in zip(tickets, results):
            if "ticket_id" in t:
                r["ticket_id"] = t["ticket_id"]
        payload = {"predictions": results, "latency_ms": ms} if batch else {**results[0], "latency_ms": ms}
        return jsonify(payload)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="not found"), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return jsonify(error="method not allowed"), 405

    @app.errorhandler(500)
    def server_error(e):
        log.exception("unhandled error: %s", e)
        return jsonify(error="internal server error"), 500

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))
