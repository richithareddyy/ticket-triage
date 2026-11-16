import numpy as np
import pandas as pd
import pytest

from triage.features import TEXT_FEATURE, build_features
from triage.model import classifiers, regressors
from triage.temporal import (RESOLUTION_CAP_HOURS, evaluate_window, fit_stacked, forward_text_scores, in_queue,
                             labels, shap_check, window_rows)

CUTOFF = pd.Timestamp("2021-01-01")
SNAPSHOT = pd.Timestamp("2022-06-01")


def resolution_rows(rows):
    df = pd.DataFrame(rows, columns=["created_at", "first_resolved_at"]).apply(pd.to_datetime)
    df["priority"] = "Medium"
    return df


def test_old_ticket_resolving_after_the_cutoff_cannot_train_that_cutoff():
    df = resolution_rows([
        ("2020-11-01", "2021-01-10"),  # 70 days: still open at the cutoff, outcome unknown
        ("2020-06-01", "2021-01-10"),  # open past the 90-day cap before the cutoff: label is the cap
        ("2020-11-20", "2020-11-21"),  # resolved quickly, but filed under 90 days before the cutoff
        ("2020-03-01", "2020-03-03"),
    ])
    y, ready = labels(df, "regression")
    train, _ = window_rows(df["created_at"], ready, in_queue(df), CUTOFF, CUTOFF, SNAPSHOT)
    assert train.tolist() == [1, 3]
    assert y[1] == RESOLUTION_CAP_HOURS and y[3] == 48


def test_unresolved_tickets_are_censored_at_the_cap_never_zero():
    df = resolution_rows([("2020-01-01", None), ("2022-05-01", None), ("2020-01-01", "2020-01-01 00:05")])
    y, ready = labels(df, "regression")
    assert y[0] == RESOLUTION_CAP_HOURS and (y > 0).all()
    assert in_queue(df).tolist() == [True, True, False]  # resolved before prediction time: never queued
    _, test = window_rows(df["created_at"], ready, in_queue(df), pd.Timestamp("2020-01-01"),
                          pd.Timestamp("2023-01-01"), SNAPSHOT)
    assert test.tolist() == [0]  # the 2022 ticket hadn't reached the cap by the snapshot


def test_priority_label_matures_fourteen_days_after_filing():
    df = resolution_rows([("2020-12-10", None), ("2020-12-25", None)])
    _, ready = labels(df, "classification")
    train, _ = window_rows(df["created_at"], ready, in_queue(df), CUTOFF, CUTOFF, SNAPSHOT)
    assert train.tolist() == [0]


@pytest.fixture(scope="module")
def history(tickets):
    """Synthetic tickets spread over two years, with resolution times turned into timestamps."""
    df = tickets.head(1500).copy()
    df["created_at"] = pd.date_range("2019-01-01", "2020-12-31", periods=len(df))
    df["first_resolved_at"] = df["created_at"] + pd.to_timedelta(df["resolution_hours"], unit="h")
    return df.reset_index(drop=True), build_features(df).reset_index(drop=True)


def test_forward_scores_ignore_later_tickets_and_unsettled_labels(history):
    df, X = history
    y, ready = labels(df, "regression")
    created = df["created_at"].to_numpy()
    scores, scored = forward_text_scores("regression", X[TEXT_FEATURE], y, created, ready, blocks=3, min_rows=100)
    assert not scored[:500].any() and scored[500:].all()  # the first block is warm-up

    # Rewrite the last block, and the labels of earlier rows that hadn't settled when block 2 began.
    block2_start = created[500]
    unsettled = (ready > block2_start) & (created < block2_start)
    assert unsettled.any()
    text2, y2 = X[TEXT_FEATURE].copy(), y.copy()
    text2.iloc[1000:] = "completely different words"
    y2[1000:] = 1.0
    y2[unsettled] = 9999.0
    again, _ = forward_text_scores("regression", text2, y2, created, ready, blocks=3, min_rows=100)
    np.testing.assert_array_equal(scores[:1000], again[:1000])


def test_window_predictions_do_not_depend_on_the_future(history):
    df, X = history
    y, ready = labels(df, "classification")
    created = df["created_at"].to_numpy()
    cutoff, end = pd.Timestamp("2020-07-01"), pd.Timestamp("2021-01-01")
    train, test = window_rows(created, ready, in_queue(df), cutoff, end, SNAPSHOT)

    def run(labels_):
        _, fitted, _ = evaluate_window("classification", X, labels_, created, ready, train, test, ["logistic_regression"])
        return fitted["logistic_regression"].predict_proba(X.iloc[test])

    changed = y.copy()
    changed[ready > np.datetime64(cutoff)] = 0  # outcomes that settle after the cutoff
    np.testing.assert_array_equal(run(y), run(changed))


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_shap_stays_additive_with_forward_stacking(history, task):
    df, X = history
    y, ready = labels(df, task)
    created = df["created_at"].to_numpy()
    inner = forward_text_scores(task, X[TEXT_FEATURE], y, created, ready, blocks=3, min_rows=100)
    est = (classifiers() if task == "classification" else regressors())["xgboost"]
    pipe = fit_stacked(est, task, X, y, inner)
    gap, importance = shap_check(pipe, task, X.tail(40), X.head(100))
    assert gap < 1e-3 and importance
    assert pipe.named_steps["prep"].named_transformers_["text"].oof is None
