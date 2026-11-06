# Support Ticket Triage

A support ticket arrives, and someone has to decide how urgent it is and how long it might take to resolve. This project predicts both from the ticket text and metadata, then shows which inputs pushed the predictions up or down. It is trained and evaluated on synthetic tickets, so the demo is a way to inspect the approach, not evidence that it works on a real support queue.

Python, XGBoost, SHAP, Flask, Streamlit, PySpark.

The first version fed 3,000 raw TF-IDF columns straight into XGBoost. Accuracy was fine, but the SHAP explanations were dominated by words being *absent* (“the word *would* isn’t in this ticket”), which tells an agent nothing. Now a linear text model compresses the text into a few dense scores, generated out of fold so the final model never sees predictions fit on its own labels, and XGBoost learns how those scores interact with ticket metadata. Accuracy stayed about the same and the explanations became readable.

## Results and scope

Training and evaluation use **20,000 synthetic tickets**, split into 14,000 training, 3,000 validation, and 3,000 test examples. The final XGBoost models report:

| Task | Test result | Baseline |
| --- | --- | --- |
| Priority | Macro-F1 0.705 | Majority class: 0.130 |
| Resolution time | MAE 12.2 hours | Median: 24.4 hours |

These results describe the generated dataset, not performance on a real support queue. [Committed metrics](artifacts/metrics.json) contain the candidate comparisons and final results. SHAP contributions explain model behavior, not causal effects.

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
