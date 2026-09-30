---
title: Support Ticket Triage
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Predicts ticket priority and resolution time, explained with SHAP
---

# Support Ticket Triage & Resolution-Time Predictor

Paste a support ticket and get:

- **Priority** (Low / Medium / High / Critical) from an XGBoost classifier
- **Predicted time to resolution** and an SLA-breach flag
- **SHAP explanations** showing which fields and words drove each prediction

Models: Logistic Regression, Random Forest and XGBoost were compared; XGBoost was selected
(macro-F1 0.705 on a held-out test set vs 0.130 for a majority-class baseline). Features use NLTK
(lemmatization, VADER sentiment) with a matching PySpark batch pipeline. The container serves a Flask
REST API and this Streamlit interface.

Trained on a synthetic ticket dataset with a known causal structure (real ticket data is proprietary).
