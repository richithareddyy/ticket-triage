import numpy as np
import pandas as pd

from triage.jira import to_tickets
from triage.train import make_split


def issue(key, priority="Major", created="2020-01-01T10:00:00.000+0000", resolved="2020-01-02T10:00:00.000+0000",
          reporter="ann"):
    return {"key": key, "fields": {
        "summary": "Broker crashes", "description": "x" * 9000, "created": created, "resolutiondate": resolved,
        "priority": {"name": priority} if priority else None, "project": {"key": "KAFKA"},
        "components": [{"name": "core"}], "attachment": [{}, {}], "reporter": {"name": reporter},
    }}


def test_maps_priorities_and_resolution_time():
    df = to_tickets([issue("A-1", "Blocker"), issue("A-2", "Critical"), issue("A-3", "Major"), issue("A-4", "Trivial")])
    assert df.set_index("ticket_id")["priority"].to_dict() == {
        "A-1": "Critical", "A-2": "High", "A-3": "Medium", "A-4": "Low"}
    assert (df["resolution_hours"] == 24).all()
    assert (df["customer_tier"] == "unknown").all() and (df["channel"] == "unknown").all()
    assert df["description"].str.len().max() == 4000


def test_drops_records_that_cannot_be_used():
    df = to_tickets([
        issue("OK-1"),
        issue("BAD-1", priority=None),
        issue("BAD-2", priority="Unknown"),
        issue("BAD-3", created="0010-04-03T23:04:00.000+0000"),
        issue("BAD-4", resolved=None),
        issue("BAD-5", created="2021-01-01T00:00:00.000+0000", resolved="2020-01-01T00:00:00.000+0000"),
    ])
    assert df["ticket_id"].tolist() == ["OK-1"]


def test_prior_tickets_only_look_back_and_drop_reporter_names():
    df = to_tickets([
        issue("A-3", created="2020-03-20T00:00:00.000+0000", resolved="2020-04-01T00:00:00.000+0000"),
        issue("A-1", created="2020-01-01T00:00:00.000+0000"),
        issue("A-2", created="2020-01-10T00:00:00.000+0000", resolved="2020-02-01T00:00:00.000+0000"),
        issue("B-1", created="2020-01-11T00:00:00.000+0000", resolved="2020-02-01T00:00:00.000+0000", reporter="bo"),
    ])
    assert df.set_index("ticket_id")["prior_tickets_30d"].to_dict() == {"A-1": 0, "A-2": 1, "B-1": 0, "A-3": 0}
    assert "reporter" not in df.columns


def test_time_split_tests_on_the_newest_tickets():
    created = pd.Series(pd.date_range("2020-01-01", periods=100)).sample(frac=1, random_state=0).reset_index(drop=True)
    tr, va, te = make_split(np.zeros(100, dtype=int), created, "time")
    assert (len(tr), len(va), len(te)) == (70, 15, 15)
    assert created[tr].max() < created[va].min() and created[va].max() < created[te].min()
