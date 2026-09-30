import json

import numpy as np
import pandas as pd

from triage.features import build_features
from triage.model import Explainer

from .conftest import SAMPLE_TICKET


def test_training_outputs_and_beats_baselines(artifacts):
    m = json.load(open(artifacts / "metrics.json"))
    p, r = m["priority"], m["resolution_time"]
    assert p["selected_model"] in ("logistic_regression", "random_forest", "xgboost")
    assert p["final_test"]["macro_f1"] > p["comparison"]["baseline_majority"]["test"]["macro_f1"] + 0.3
    assert r["final_test"]["mae_hours"] < r["comparison"]["baseline_median"]["test"]["mae_hours"]
    assert (artifacts / "shap_priority.png").exists() and (artifacts / "shap_resolution.png").exists()


def test_prediction_shape(predictor):
    res = predictor.predict([SAMPLE_TICKET], explain=True)[0]
    assert res["priority"] in predictor.classes
    assert abs(sum(res["priority_probabilities"].values()) - 1) < 1e-3
    assert res["resolution_hours"] > 0
    assert len(res["explanation"]["priority"]) == 6
    assert res["explanation"]["priority_key_terms"]


def test_obvious_cases_are_ordered(predictor):
    outage, request = predictor.predict([SAMPLE_TICKET, {
        **SAMPLE_TICKET, "subject": "Would love dark mode", "customer_tier": "free", "channel": "web",
        "description": "Please consider adding a dark mode. Thanks!", "category": "feature_request",
        "prior_tickets_30d": 0}])
    order = predictor.classes
    assert order.index(outage["priority"]) > order.index(request["priority"])
    assert outage["resolution_hours"] < request["resolution_hours"]


def test_grouped_shap_is_additive(predictor, tickets):
    """Grouped contributions (one-hot fields summed) + base value must equal the model output."""
    X = build_features(pd.DataFrame(tickets.head(20)))
    exp = Explainer(predictor.reg, predictor._background)
    values = exp.shap_values(X)
    grouped = np.column_stack([v for _, _, v in exp._grouped(values, X)])
    margin = predictor.reg.predict(X)
    base = np.ravel(exp.explainer.expected_value)[0]
    np.testing.assert_allclose(grouped.sum(axis=1) + base, margin, atol=1e-3)
