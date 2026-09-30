"""PySpark and pandas feature pipelines must produce identical features.

Needs pyspark + Java; skipped otherwise. Run anywhere with:
    docker compose --profile batch run --rm --entrypoint pytest spark tests/test_spark_parity.py
"""
import subprocess

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")


def _java_available():
    # macOS ships a /usr/bin/java stub even without a JDK, so check that it actually runs.
    try:
        return subprocess.run(["java", "-version"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _java_available():
    pytest.skip("Java runtime not found", allow_module_level=True)

from triage.data import generate  # noqa: E402
from triage.features import FEATURE_COLUMNS, NUMERIC_FEATURES, build_features  # noqa: E402


def test_spark_matches_pandas(tmp_path):
    from pyspark.sql import SparkSession

    from triage.spark_features import build_features as spark_build

    raw = generate(n=300, seed=3)
    path = tmp_path / "t.csv"
    raw.to_csv(path, index=False)

    spark = (SparkSession.builder.master("local[2]").config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.enabled", "false").getOrCreate())
    try:
        sdf = spark.read.csv(str(path), header=True, multiLine=True, escape='"')
        got = spark_build(sdf).toPandas().set_index("ticket_id").loc[raw["ticket_id"], FEATURE_COLUMNS]
    finally:
        spark.stop()
    expected = build_features(pd.read_csv(path))
    got = got.reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        if col in NUMERIC_FEATURES:
            pd.testing.assert_series_equal(got[col].astype(float), expected[col].astype(float),
                                           check_names=False, atol=1e-4, obj=col)
        else:
            assert (got[col] == expected[col]).all(), col
