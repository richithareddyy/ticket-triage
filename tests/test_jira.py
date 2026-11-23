import numpy as np
import pandas as pd

from triage.jira import to_bugs
from triage.train import make_split


def change(when, field, frm=None, to=None, frm_id=None, to_id=None):
    return {"created": when, "items": [{"field": field, "fromString": frm, "toString": to,
                                        "from": frm_id, "to": to_id}]}


def issue(key, priority="Major", created="2020-01-01T10:00:00.000+0000", resolved="2020-01-02T10:00:00.000+0000",
          reporter="ann", history=(), components=({"id": "1", "name": "core"},), attachments=None):
    if attachments is None:
        attachments = [{"id": "a1"}, {"id": "a2"}]
    if resolved and not any(i["field"] == "resolution" for h in history for i in h["items"]):
        history = [*history, change(resolved, "resolution", to="Fixed", to_id="1")]
    return {"key": key, "changelog": {"total": len(history), "histories": list(history)}, "fields": {
        "summary": "Broker crashes", "description": "x" * 9000, "created": created, "resolutiondate": resolved,
        "priority": {"name": priority} if priority else None, "project": {"key": key.split("-")[0]},
        "components": list(components), "attachment": attachments, "reporter": {"name": reporter},
    }}


def bugs(*issues):
    return to_bugs(issues).set_index("ticket_id")


def test_maps_priorities_including_cassandras_scheme():
    df = bugs(issue("A-1", "Blocker"), issue("A-2", "Critical"), issue("A-3", "Major"), issue("A-4", "Trivial"),
              issue("C-1", "Urgent"), issue("C-2", "Normal"))
    assert df["priority"].to_dict() == {
        "A-1": "Critical", "A-2": "High", "A-3": "Medium", "A-4": "Low", "C-1": "Critical", "C-2": "Medium"}
    assert (df["customer_tier"] == "unknown").all() and (df["channel"] == "unknown").all()
    assert df["description"].str.len().max() == 4000


def test_priority_label_is_the_value_fourteen_days_after_filing():
    df = bugs(
        issue("A-1", "Blocker", history=[change("2020-01-03T00:00:00.000+0000", "priority", "Major", "Blocker")]),
        issue("A-2", "Blocker", history=[change("2020-03-01T00:00:00.000+0000", "priority", "Major", "Blocker")]),
    )
    assert df["filed_priority"].to_dict() == {"A-1": "Medium", "A-2": "Medium"}
    assert df["priority"].to_dict() == {"A-1": "Critical", "A-2": "Medium"}  # A-2 changed after the horizon
    assert (df["priority_known_at"] == pd.Timestamp("2020-01-15 10:00")).all()


def test_inputs_are_rebuilt_as_filed():
    df = bugs(issue(
        "A-1", components=[{"id": "2", "name": "streams"}],
        attachments=[{"id": "a1"}, {"id": "a3"}],
        history=[
            change("2020-01-01T10:05:00.000+0000", "Attachment", to="first.log", to_id="a1"),
            change("2020-01-01T10:06:00.000+0000", "Attachment", to="gone.log", to_id="a2"),
            change("2020-01-01T12:00:00.000+0000", "description", "Original text", "Edited text"),
            change("2020-01-01T12:00:00.000+0000", "summary", "Old title", "Broker crashes"),
            change("2020-01-02T00:00:00.000+0000", "Component", frm="core", frm_id="1"),
            change("2020-01-02T00:00:00.000+0000", "Component", to="streams", to_id="2"),
            change("2020-01-03T00:00:00.000+0000", "Attachment", frm="gone.log", frm_id="a2"),
            change("2020-01-03T00:00:00.000+0000", "Attachment", to="later.patch", to_id="a3"),
        ])).loc["A-1"]
    assert (df["subject"], df["description"], df["category"]) == ("Old title", "Original text", "core")
    assert df["attachments"] == 2  # a1 and a2 were there when filed; a3 came later


def test_component_removed_within_the_grace_period_does_not_count():
    swap = [change("2020-01-01T10:05:00.000+0000", "Component", frm="core", frm_id="1"),
            change("2020-01-01T10:05:00.000+0000", "Component", to="streams", to_id="2")]
    df = bugs(
        issue("A-1", components=[{"id": "2", "name": "streams"}], history=swap),
        issue("A-2", components=[], history=swap[:1]),
        issue("A-3", components=[], history=[change("2020-01-01T10:20:00.000+0000", "Component", frm="core", frm_id="1")]),
    )
    # A-3 lost its component after prediction time, so it still counts there.
    assert df["category"].to_dict() == {"A-1": "streams", "A-2": "unknown", "A-3": "core"}


def test_keeps_every_outcome_and_drops_unusable_records():
    df = bugs(
        issue("OK-1"),
        issue("OPEN-1", resolved=None),
        issue("WONTFIX-1", history=[change("2020-01-05T00:00:00.000+0000", "resolution", to="Won't Fix", to_id="2")],
              resolved="2020-01-05T00:00:00.000+0000"),
        issue("BAD-1", priority=None),
        issue("BAD-2", priority="Unknown"),
        issue("BAD-3", created="0010-04-03T23:04:00.000+0000"),
        issue("RETYPED-1", history=[change("2020-01-02T00:00:00.000+0000", "issuetype", "Improvement", "Bug")]),
    )
    assert sorted(df.index) == ["OK-1", "OPEN-1", "WONTFIX-1"]
    assert pd.isna(df.loc["OPEN-1", "first_resolved_at"])


def test_reopened_bug_uses_its_first_resolution():
    df = bugs(issue("A-1", resolved="2020-03-01T00:00:00.000+0000", history=[
        change("2020-01-04T00:00:00.000+0000", "resolution", to="Fixed", to_id="1"),
        change("2020-01-10T00:00:00.000+0000", "resolution", frm="Fixed", frm_id="1"),
        change("2020-03-01T00:00:00.000+0000", "resolution", to="Fixed", to_id="1"),
    ]))
    assert df.loc["A-1", "first_resolved_at"] == pd.Timestamp("2020-01-04")


def test_prior_tickets_count_all_earlier_bugs_and_drop_reporter_names():
    df = bugs(
        issue("A-3", created="2020-03-20T00:00:00.000+0000", resolved="2020-04-01T00:00:00.000+0000"),
        issue("A-1", created="2020-01-01T00:00:00.000+0000", resolved=None),
        issue("A-2", created="2020-01-10T00:00:00.000+0000", resolved="2020-02-01T00:00:00.000+0000"),
        issue("B-1", created="2020-01-11T00:00:00.000+0000", resolved="2020-02-01T00:00:00.000+0000", reporter="bo"),
    )
    assert df["prior_tickets_30d"].to_dict() == {"A-1": 0, "A-2": 1, "B-1": 0, "A-3": 0}
    assert "reporter" not in df.columns


def test_time_split_tests_on_the_newest_tickets():
    created = pd.Series(pd.date_range("2020-01-01", periods=100)).sample(frac=1, random_state=0).reset_index(drop=True)
    tr, va, te = make_split(np.zeros(100, dtype=int), created, "time")
    assert (len(tr), len(va), len(te)) == (70, 15, 15)
    assert created[tr].max() < created[va].min() and created[va].max() < created[te].min()
