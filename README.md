# Support Ticket Triage

A support ticket arrives. How urgent is it, and how long might it take to resolve? This project estimates both from the ticket text and details such as its channel and product, then shows which inputs influenced each prediction. The demo uses synthetic tickets.

Python, XGBoost, SHAP, Flask, Streamlit, PySpark.

The first version represented text with 3,000 word features. Its SHAP explanations often pointed to missing words: “the word *would* isn’t in this ticket.” That wasn’t useful. Now a small text model turns the message into a few scores, which XGBoost combines with the other ticket details. During training, each score comes from a model that did not train on that ticket. This avoids leaking its answer into the features.

## Results and scope

Training and evaluation use **20,000 synthetic tickets**, split into 14,000 training, 3,000 validation, and 3,000 test examples. The final XGBoost models report:

| Task | Test result | Baseline |
| --- | --- | --- |
| Priority | Macro-F1 0.705 | Majority class: 0.130 |
| Resolution time | MAE 12.2 hours | Median: 24.4 hours |

These results describe the generated dataset, not performance on a real support queue. [Committed metrics](artifacts/metrics.json) contain the candidate comparisons and final results. SHAP shows how inputs influenced the model’s prediction; it does not show what caused a ticket to take longer.

## Run locally

Requires Python 3.9+ and Docker. On macOS, XGBoost also needs `libomp`.

```bash
make setup
make data
make train
make up
```

Open http://localhost:8501; the Flask API runs on port 5050. [Try the hosted demo](https://richithareddyy-ticket-triage-demostreamlit-app-3uubea.streamlit.app).

## API and checks

`POST /predict` accepts a ticket or a batch of up to 100. Add `?explain=true` to request SHAP contributions. Predictions include a resolution-time estimate and a comparison with the configured SLA target.

Run `make test` for pytest and the Docker-based Spark/pandas parity check. `make spark` builds batch features; `make train-spark` trains from that output. API timing is per-request instrumentation, not a committed latency benchmark.

[Full setup, API example, and deployment](GUIDE.md) · [Training code](triage/train.py) · [Feature definitions](triage/features.py)
