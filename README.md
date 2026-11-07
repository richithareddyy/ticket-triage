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

## Real-data check

The results above are the easy case: the models are learning rules that were written into the data generator. To test the pipeline on real data, it was also run on **50,855 fixed bug reports** from the public Apache Jira tracker (Hive, Spark, Kafka, HBase, HDFS, YARN, Flink, Cassandra), training on the oldest 70% and testing on the newest 15%.

| Test metric | Synthetic | Apache Jira | Jira baseline |
| --- | --- | --- | --- |
| Priority macro-F1 | 0.705 | 0.360 | 0.212 |
| Priority accuracy | 68.8% | 65.6% | 73.5% |
| Resolution MAE | 12.2 hours | 1,005 hours | 1,020 hours |
| Resolution median error | 4.6 hours | 165 hours | 130 hours |

Priority has a real but modest signal: macro-F1 beats the baseline because the model finds some of the rarer levels, while accuracy falls below always answering Medium, which covers 74% of the test set. Resolution time is not useful on this data. MAE barely moves and median error is worse than the baseline, so the opening text and details say little about how long an open-source bug stays open.

These are bug reports rather than customer-support tickets. Jira has no customer tier or channel, so those fields are `unknown`. Only fixed bugs are included. Priorities, components, and attachment counts are their current values, which may have changed after a ticket was opened, so results may be optimistic. Random Forest beat XGBoost on validation here. The [Jira metrics](artifacts/jira/metrics.json) are committed; the hosted demo still uses the synthetic model. `make jira train-jira` reproduces the run.

## Run locally

Requires Python 3.9+ and Docker. On macOS, XGBoost also needs `libomp`.

```bash
make setup
make data
make train
make up
```

Open http://localhost:8501; the Flask API runs on port 5050. [Try the hosted demo](https://tickettriage.streamlit.app).

## API and checks

`POST /predict` accepts a ticket or a batch of up to 100. Add `?explain=true` to request SHAP contributions. Predictions include a resolution-time estimate and a comparison with the configured SLA target.

Run `make test` for pytest and the Docker-based Spark/pandas parity check. `make spark` builds batch features; `make train-spark` trains from that output. API timing is per-request instrumentation, not a committed latency benchmark.

[Full setup, API example, and deployment](GUIDE.md) · [Training code](triage/train.py) · [Feature definitions](triage/features.py)
