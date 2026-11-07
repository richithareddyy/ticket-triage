"""Build a real-data benchmark from the public Apache Jira tracker.

    python -m triage.jira --out data/jira_tickets.csv

These are open-source bug reports rather than customer-support tickets, but they
have what the synthetic set can't offer: text written by real users, priorities
set by real maintainers, and real open/close times. Jira has no customer tier or
contact channel, so those columns stay "unknown" instead of being made up.
"""
import argparse
import os
import time
from collections import defaultdict, deque

import pandas as pd
import requests

SEARCH_URL = "https://issues.apache.org/jira/rest/api/2/search"
PROJECTS = ["HIVE", "SPARK", "KAFKA", "HBASE", "HDFS", "YARN", "FLINK", "CASSANDRA"]
FIELDS = "summary,description,priority,created,resolutiondate,project,components,reporter,attachment"
PAGE_SIZE = 1000
MAX_DESCRIPTION = 4000  # long descriptions are mostly pasted stack traces
PRIORITY_MAP = {"Blocker": "Critical", "Critical": "High", "Major": "Medium", "Minor": "Low", "Trivial": "Low"}


def fetch_issues(projects=PROJECTS):
    jql = f"project in ({','.join(projects)}) AND issuetype = Bug AND resolution = Fixed ORDER BY created ASC, key ASC"
    issues, start = [], 0
    while True:
        r = requests.get(SEARCH_URL, timeout=60,
                         params={"jql": jql, "fields": FIELDS, "startAt": start, "maxResults": PAGE_SIZE})
        if r.status_code in (429, 502, 503, 504):
            time.sleep(30)
            continue
        r.raise_for_status()
        page = r.json()
        issues += page["issues"]
        start += len(page["issues"])
        print(f"fetched {start:,} of {page['total']:,}", flush=True)
        if not page["issues"] or start >= page["total"]:
            return issues
        time.sleep(1)  # shared ASF infrastructure, so don't hammer it


def to_tickets(issues):
    """Map raw Jira issues onto the ticket schema, keeping only fields known when a ticket is opened.

    Reporter names are used to count recent tickets and then dropped.
    """
    rows = []
    for issue in issues:
        f = issue["fields"]
        priority = PRIORITY_MAP.get((f.get("priority") or {}).get("name"))
        if priority is None or not f.get("resolutiondate"):
            continue
        components = f.get("components") or []
        rows.append(dict(
            ticket_id=issue["key"], created_at=f["created"], resolved_at=f["resolutiondate"],
            subject=f.get("summary") or "", description=(f.get("description") or "")[:MAX_DESCRIPTION],
            channel="unknown", product=f["project"]["key"], customer_tier="unknown",
            category=components[0]["name"] if components else "unknown",
            attachments=len(f.get("attachment") or []), priority=priority,
            reporter=(f.get("reporter") or {}).get("name"),
        ))
    df = pd.DataFrame(rows)
    # Early records include impossible dates such as the year 0010.
    for col in ("created_at", "resolved_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["created_at", "resolved_at"])
    df = df[(df["created_at"] >= "2005-01-01") & (df["resolved_at"] >= df["created_at"])]
    df = df.sort_values("created_at").reset_index(drop=True)

    df["resolution_hours"] = ((df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600).clip(lower=0.25).round(2)
    df["prior_tickets_30d"] = _prior_tickets(df)
    return df.drop(columns=["resolved_at", "reporter"])


def _prior_tickets(df):
    """Tickets the same reporter opened in the previous 30 days; expects rows sorted by created_at."""
    window = pd.Timedelta(days=30)
    recent = defaultdict(deque)
    counts = []
    for reporter, created in zip(df["reporter"], df["created_at"]):
        if reporter is None:
            counts.append(0)
            continue
        seen = recent[reporter]
        while seen and created - seen[0] > window:
            seen.popleft()
        counts.append(len(seen))
        seen.append(created)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/jira_tickets.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = to_tickets(fetch_issues())
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} tickets -> {args.out}")
    print(df["priority"].value_counts().to_string())
    print(df["resolution_hours"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
