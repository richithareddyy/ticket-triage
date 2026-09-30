import pandas as pd
import pytest

from triage.features import FEATURE_COLUMNS, build_features
from triage.text import clean_text


def make(**overrides):
    row = dict(subject="Site DOWN", description="Error 500 everywhere!! Is this urgent? yes, URGENT.",
               created_at="2026-09-26 22:30:00", channel="Email", product="API", customer_tier="pro",
               category="outage", prior_tickets_30d=3, attachments=0)
    row.update(overrides)
    return pd.DataFrame([row])


def test_structured_features():
    f = build_features(make()).iloc[0]
    assert list(f.index) == FEATURE_COLUMNS
    assert f["exclamation_count"] == 2
    assert f["question_count"] == 1
    assert f["urgency_count"] == 3          # down, urgent, urgent
    assert f["has_error_code"] == 1
    assert f["day_of_week"] == 5 and f["is_weekend"] == 1   # 2026-09-26 is a Saturday
    assert f["hour"] == 22 and f["is_business_hours"] == 0
    assert f["channel"] == "email"          # categoricals are normalized to lowercase
    assert 0 < f["upper_ratio"] < 1


def test_clean_text_lemmatizes_and_keeps_negations():
    out = clean_text("The exports are NOT working, error 404 on all dashboards")
    assert "not" in out.split() and "all" in out.split()
    assert "export" in out.split() and "dashboard" in out.split()
    assert "errcode" in out.split() and "the" not in out.split()


def test_missing_values_are_tolerated():
    f = build_features(make(description=None, category=None, attachments=None)).iloc[0]
    assert f["category"] == "unknown" and f["attachments"] == 0


def test_missing_column_raises():
    with pytest.raises(ValueError, match="channel"):
        build_features(make().drop(columns=["channel"]))
