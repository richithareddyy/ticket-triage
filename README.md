# Support Ticket Triage

A support ticket arrives. How urgent is it, and how long might it take to resolve? This project estimates both from the ticket text and details such as its channel and product, then shows which inputs influenced each prediction. The demo uses synthetic tickets.

**[Live demo → richithareddyy-ticket-triage.streamlit.app](https://richithareddyy-ticket-triage.streamlit.app)**

Python, XGBoost, SHAP, Flask, Streamlit, PySpark.

[![Home page of the demo](docs/home.jpg)](https://richithareddyy-ticket-triage.streamlit.app)

Pick a sample ticket or write your own, then press **Triage**. The ticket is stamped with a priority, an expected resolution time, and its SLA status:

![A triaged ticket](docs/triage.jpg)

The paper trail lists the main factors behind each prediction as SHAP contributions, plus the words that drove the text scores:

![The paper trail with SHAP factors](docs/paper-trail.jpg)

The hosted app may take up to a minute to wake if nobody has opened it recently.

The first version represented text with 3,000 word features. Its SHAP explanations often pointed to missing words: “the word *would* isn’t in this ticket.” That wasn’t useful. Now a small text model turns the message into a few scores, which XGBoost combines with the other ticket details. During training, each score comes from a model that did not train on that ticket. So a ticket's own label never trains the model that scores its text.

## Results and scope

Training and evaluation use **20,000 synthetic tickets**, split into 14,000 training, 3,000 validation, and 3,000 test examples. The final XGBoost models report:

| Task | Test result | Baseline |
| --- | --- | --- |
| Priority | Macro-F1 0.705 | Majority class: 0.130 |
| Resolution time | MAE 12.2 hours | Median: 24.4 hours |

These results describe the generated dataset, not performance on a real support queue. [Committed metrics](artifacts/metrics.json) contain the candidate comparisons and final results. SHAP shows how inputs influenced the model’s prediction; it does not show what caused a ticket to take longer.

## Real-data check

The results above are the easy case: the models are learning rules that were written into the data generator. To test the pipeline on real data, it runs on **96,086 bug reports** from the public Apache Jira tracker (Hive, Spark, Kafka, HBase, HDFS, YARN, Flink, Cassandra), of which 93,519 were still open 15 minutes after filing.

This is a forward-in-time evaluation with inputs rebuilt as filed:

- **Inputs** are the summary, description, component and attachments as they stood 15 minutes after filing, rebuilt from each issue's changelog, plus project, filing time and the reporter's bug count over the previous 30 days.
- **Priority** is the value 14 days after filing (5.4% changed in that time). **Resolution time** is hours to first resolution, capped at 90 days; a bug still open at the cap is labelled 90 days, never zero. Every outcome is kept: fixed, won't-fix, duplicate and never resolved (16.8%).
- **Each year from 2019 to 2025 is predicted by models trained on bugs whose outcomes had settled before that year began.** A bug filed in November is not used for resolution time until February, even if it closed the next day.
- **Text scores inside the stacked model are also forward-only.** Training rows are cut into five chronological blocks, and each block is scored by a text model fit only on outcomes settled before it. The first block trains the text model but is left out of the final model.
- **2019 picks the model family** (Random Forest for priority, XGBoost for resolution time); 2020–2025 test it unchanged.

Averages over the six test years:

| Metric | Model | Majority / median | Text only | Metadata only |
| --- | --- | --- | --- | --- |
| Priority macro-F1 | **0.333** | 0.214 | 0.296 | 0.297 |
| Priority accuracy | 71.5% | **74.6%** | 71.7% | 46.8% |
| Resolution MAE | 867 h | 916 h | **867 h** | 897 h |
| Resolution median error | 508 h | **374 h** | 520 h | 491 h |

Priority has a modest, real signal. The model beats every simple baseline on macro-F1 in five of six years (0.33–0.36 against at most 0.31), but not in 2025 (0.266 against 0.298), and its accuracy is below always answering Medium. Resolution time is not useful: the full model is no better than the text-only model, and its median error is worse than predicting the median. In plain terms, always guessing the majority class gets higher priority accuracy than the model (74.6% against 71.5%). The resolution-time model does not beat the text-only baseline on MAE (867 hours each), and while it beats the median baseline on MAE (867 against 916 hours), it loses to it on median error (508 against 374 hours). The most influential inputs for both tasks are the project and the text scores. In the last year's models, grouped SHAP values sum to the model output to within 1e-5. Feature code is unchanged, so the Spark/pandas parity test still applies.

These are bug reports, not customer-support tickets. Jira has no customer tier or channel, so those fields are `unknown`. The population is issues whose type is Bug today, minus any that became Bugs after filing; issues filed as bugs and later re-typed are missing. Per-year results are in the [rolling metrics](artifacts/jira_temporal/metrics.json); `make jira train-jira` reproduces them. The hosted demo still uses the synthetic model.

**Correction:** components removed within 15 minutes of filing were counted as present; fixing that changed the category on 137 of 96,086 bugs and moved average priority macro-F1 from 0.332 to 0.333 and resolution MAE by 0.2 hours, with no change to the conclusions.

**Earlier result, kept for the record.** A first version tested on 50,855 fixed bugs with a single 70/15/15 split by filing date and reported priority macro-F1 0.360 and resolution MAE 1,005 hours ([metrics](artifacts/jira/metrics.json)). It is not comparable to the table above. It used only fixed bugs, labelled current priorities and last resolution dates, used later field values as inputs, counted training outcomes that weren't known at the test cutoff, and shuffled the inner stacking folds. Its population, targets and test period also differ.

## Run locally

Requires Python 3.9+ and Docker. On macOS, XGBoost also needs `libomp`.

```bash
make setup
make data
make train
make up
```

Open http://localhost:8501; the Flask API runs on port 5050.

## API and checks

`POST /predict` accepts a ticket or a batch of up to 100. Add `?explain=true` to request SHAP contributions. Predictions include a resolution-time estimate and a comparison with the configured SLA target.

Run `make test` for pytest and the Docker-based Spark/pandas parity check. `make spark` builds batch features; `make train-spark` trains from that output. API timing is per-request instrumentation, not a committed latency benchmark.

[Full setup, API example, and deployment](GUIDE.md) · [Training code](triage/train.py) · [Feature definitions](triage/features.py)
