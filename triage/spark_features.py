"""Batch feature engineering with PySpark.

Produces the same feature table as `triage.features.build_features`, but
distributed: structured features use native Spark SQL expressions, and the NLTK
steps (lemmatization, VADER sentiment) run as vectorized pandas UDFs.

    python -m triage.spark_features --input data/tickets.csv --output data/features.parquet

Runs locally (`local[*]`) or on a cluster / EMR via spark-submit.
"""
import argparse

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TEXT_FEATURE
from .text import ERROR_CODE_RE, URGENCY_RE

LABEL_COLUMNS = ["ticket_id", "priority", "resolution_hours"]


@F.pandas_udf(StringType())
def clean_text_udf(text: pd.Series) -> pd.Series:
    from .text import clean_text
    return text.map(clean_text)


@F.pandas_udf(DoubleType())
def sentiment_udf(text: pd.Series) -> pd.Series:
    from .text import sentiment
    return text.map(sentiment).round(4)


def _count(col, pattern):
    """Non-overlapping regex match count (split keeps trailing empties, so parts - 1 = matches)."""
    return F.size(F.split(col, pattern, -1)) - 1


def build_features(df):
    text = F.concat_ws(" ", F.coalesce("subject", F.lit("")), F.coalesce("description", F.lit("")))
    df = df.withColumn("_text", text).withColumn("_lower", F.lower("_text"))
    created = F.to_timestamp("created_at")
    # Spark dayofweek is 1=Sunday..7=Saturday; convert to pandas convention 0=Monday..6=Sunday.
    dow = (F.dayofweek(created) + 5) % 7
    letters = F.length(F.regexp_replace("_text", "[^A-Za-z]", ""))

    cols = [clean_text_udf("_text").alias(TEXT_FEATURE)]
    cols += [F.lower(F.coalesce(F.col(c).cast("string"), F.lit("unknown"))).alias(c)
             for c in CATEGORICAL_FEATURES]
    cols += [
        F.length("_text").alias("text_len"),
        F.size(F.filter(F.split("_text", r"\s+"), lambda x: x != "")).alias("word_count"),
        _count(F.col("_text"), "!").alias("exclamation_count"),
        _count(F.col("_text"), r"\?").alias("question_count"),
        (F.length(F.regexp_replace("_text", "[^A-Z]", "")) / F.greatest(letters, F.lit(1))).alias("upper_ratio"),
        _count(F.col("_lower"), URGENCY_RE).alias("urgency_count"),
        F.col("_lower").rlike(ERROR_CODE_RE).cast("int").alias("has_error_code"),
        sentiment_udf("_text").alias("sentiment"),
        F.hour(created).alias("hour"),
        dow.alias("day_of_week"),
        (dow >= 5).cast("int").alias("is_weekend"),
        ((F.hour(created) >= 8) & (F.hour(created) < 18) & (dow < 5)).cast("int").alias("is_business_hours"),
        F.coalesce(F.col("prior_tickets_30d").cast("int"), F.lit(0)).alias("prior_tickets_30d"),
        F.coalesce(F.col("attachments").cast("int"), F.lit(0)).alias("attachments"),
    ]
    keep = [c for c in LABEL_COLUMNS if c in df.columns]
    return df.select(*keep, *cols).select(*keep, *FEATURE_COLUMNS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/tickets.csv")
    ap.add_argument("--output", default="data/features.parquet")
    ap.add_argument("--master", default="local[*]")
    args = ap.parse_args()

    spark = (SparkSession.builder.appName("ticket-triage-features").master(args.master)
             .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    # multiLine + escape handles descriptions containing quotes/commas.
    raw = spark.read.csv(args.input, header=True, multiLine=True, escape='"')
    feats = build_features(raw)
    feats.write.mode("overwrite").parquet(args.output)
    print(f"wrote {feats.count():,} rows -> {args.output}")
    feats.groupBy("priority").agg(F.count("*").alias("n"), F.round(F.avg("resolution_hours"), 1)
                                  .alias("avg_hours"), F.round(F.avg("urgency_count"), 2).alias("avg_urgency")
                                  ).orderBy("priority").show()
    spark.stop()


if __name__ == "__main__":
    main()
