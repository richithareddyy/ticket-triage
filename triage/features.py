"""Feature engineering shared by training and online inference.

`triage.spark_features` implements the same features in PySpark for batch
processing; `tests/test_spark_parity.py` checks the two agree.
"""
import pandas as pd

from .text import ERROR_CODE_RE, URGENCY_RE, clean_text, sentiment

RAW_COLUMNS = ["subject", "description", "created_at", "channel", "product", "customer_tier",
               "category", "prior_tickets_30d", "attachments"]
TEXT_FEATURE = "clean_text"
CATEGORICAL_FEATURES = ["channel", "product", "customer_tier", "category"]
NUMERIC_FEATURES = ["text_len", "word_count", "exclamation_count", "question_count", "upper_ratio",
                    "urgency_count", "has_error_code", "sentiment", "hour", "day_of_week",
                    "is_weekend", "is_business_hours", "prior_tickets_30d", "attachments"]
FEATURE_COLUMNS = [TEXT_FEATURE] + CATEGORICAL_FEATURES + NUMERIC_FEATURES


def validate(df):
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")


def build_features(df):
    """Raw ticket rows -> model-ready feature frame (one row per ticket)."""
    validate(df)
    text = df["subject"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)
    lower = text.str.lower()
    created = pd.to_datetime(df["created_at"])
    letters = text.str.count(r"[A-Za-z]")

    out = pd.DataFrame(index=df.index)
    out[TEXT_FEATURE] = text.map(clean_text)
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].fillna("unknown").astype(str).str.lower()
    out["text_len"] = text.str.len()
    out["word_count"] = text.str.count(r"\S+")
    out["exclamation_count"] = text.str.count("!")
    out["question_count"] = text.str.count(r"\?")
    out["upper_ratio"] = text.str.count(r"[A-Z]") / letters.clip(lower=1)
    out["urgency_count"] = lower.str.count(URGENCY_RE)
    out["has_error_code"] = lower.str.contains(ERROR_CODE_RE, regex=True).astype(int)
    out["sentiment"] = text.map(sentiment).round(4)
    out["hour"] = created.dt.hour
    out["day_of_week"] = created.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_business_hours"] = ((out["hour"] >= 8) & (out["hour"] < 18) & (out["is_weekend"] == 0)).astype(int)
    out["prior_tickets_30d"] = pd.to_numeric(df["prior_tickets_30d"]).fillna(0).astype(int)
    out["attachments"] = pd.to_numeric(df["attachments"]).fillna(0).astype(int)
    return out[FEATURE_COLUMNS]
