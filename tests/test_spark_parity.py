"""PySpark and pandas feature pipelines must produce identical features.

Needs pyspark + Java 17; skipped otherwise, except in CI (CI=true), where that is a failure.
Run anywhere with:
    docker compose --profile batch run --rm --entrypoint pytest spark tests/test_spark_parity.py
"""
import importlib.util
import os
import re
import subprocess

import pandas as pd
import pytest

REQUIRED_JAVA = 17


def _java_major():
    """Major version of the Java that Spark would start, or None if there isn't one."""
    home = os.environ.get("JAVA_HOME")  # Spark prefers it to the java on PATH
    java = os.path.join(home, "bin", "java") if home else "java"
    # macOS ships a /usr/bin/java stub even without a JDK, so check that it actually runs.
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=10,
                             check=True)
    except (OSError, subprocess.SubprocessError):  # missing, timed out or exited non-zero
        return None
    version = re.search(r'version "(?:1\.)?(\d+)', out.stderr + out.stdout)  # Java 8 is "1.8"
    return int(version.group(1)) if version else None


def _missing_requirement():
    if importlib.util.find_spec("pyspark") is None:
        return "pyspark is not installed"
    major = _java_major()
    if major is None:
        return f"Spark parity needs Java {REQUIRED_JAVA}, but no working Java runtime was found"
    if major != REQUIRED_JAVA:
        return f"Spark parity needs Java {REQUIRED_JAVA}, but the local Java is {major}"
    return None


if reason := _missing_requirement():
    if os.environ.get("CI") == "true":
        pytest.fail(f"{reason} (this test must run in CI)", pytrace=False)
    pytest.skip(reason, allow_module_level=True)

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
        got = (
            spark_build(sdf)
            .toPandas()
            .set_index("ticket_id")
            .loc[raw["ticket_id"], FEATURE_COLUMNS]
        )
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
