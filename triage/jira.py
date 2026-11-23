"""Build a real-data benchmark from the public Apache Jira tracker.

    python -m triage.jira --out data/jira_bugs.parquet

These are open-source bug reports rather than customer-support tickets, but they
have what the synthetic set can't offer: text written by real users, priorities
set by real maintainers, and real open/close times. Jira has no customer tier or
contact channel, so those columns stay "unknown" instead of being made up.

Jira returns each field's current value. A model in production would only see a
ticket as it was filed, so every input is rebuilt from the issue's changelog as it
stood FILED_GRACE after creation. Labels keep their timestamps so that
`triage.temporal` can tell which outcomes were known at a given training cutoff:

    priority             priority PRIORITY_HORIZON after creation (known from then on)
    first_resolved_at    first time any resolution was set; NaT if never

All bugs are kept, whatever their resolution. Selecting fixed bugs only would
filter on an outcome that is unknown when a ticket is filed.
"""
import argparse
import glob
import gzip
import json
import os
import time
from collections import defaultdict, deque

import pandas as pd
import requests

SEARCH_URL = "https://issues.apache.org/jira/rest/api/2/search"
ISSUE_URL = "https://issues.apache.org/jira/rest/api/2/issue/{key}"
PROJECTS = ["HIVE", "SPARK", "KAFKA", "HBASE", "HDFS", "YARN", "FLINK", "CASSANDRA"]
FIELDS = ("summary,description,priority,created,resolutiondate,project,components,reporter,"
          "attachment,issuetype")
PAGE_SIZE = 1000
MAX_DESCRIPTION = 4000  # long descriptions are mostly pasted stack traces
PRIORITY_MAP = {
    "Blocker": "Critical",
    "Critical": "High",
    "Major": "Medium",
    "Minor": "Low",
    "Trivial": "Low",
    # Cassandra's newer scheme
    "Urgent": "Critical",
    "High": "High",
    "Normal": "Medium",
    "Low": "Low",
}
FILED_GRACE = pd.Timedelta(
    minutes=15
)  # attachments and quick fixes to the text land just after filing
PRIORITY_HORIZON = pd.Timedelta(days=14)
EARLIEST = pd.Timestamp(
    "2005-01-01"
)  # early records include impossible dates such as the year 0010


def fetch_issues(cache_dir, projects=PROJECTS):
    """Download every Bug with its changelog, one gzipped JSON file per page so reruns resume."""
    os.makedirs(cache_dir, exist_ok=True)
    jql = f"project in ({','.join(projects)}) AND issuetype = Bug ORDER BY created ASC, key ASC"
    start = 0
    while True:
        path = os.path.join(cache_dir, f"page_{start:06d}.json.gz")
        if os.path.exists(path):
            with gzip.open(path, "rt") as f:
                page = json.load(f)
        else:
            r = requests.get(
                SEARCH_URL,
                timeout=300,
                params={
                    "jql": jql,
                    "fields": FIELDS,
                    "expand": "changelog",
                    "startAt": start,
                    "maxResults": PAGE_SIZE,
                },
            )
            if r.status_code in (429, 502, 503, 504):
                time.sleep(30)
                continue
            r.raise_for_status()
            page = r.json()
            _complete_changelogs(page["issues"])
            page["fetched_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None).isoformat()
            with gzip.open(path, "wt") as f:
                json.dump(page, f)
            time.sleep(1)  # shared ASF infrastructure, so don't hammer it
        start += len(page["issues"])
        print(f"fetched {start:,} of {page['total']:,}", flush=True)
        if not page["issues"] or start >= page["total"]:
            return


def _complete_changelogs(issues):
    """Search results can truncate long changelogs; refetch those issues one at a time."""
    for issue in issues:
        log = issue.get("changelog") or {}
        if log.get("total", 0) > len(log.get("histories", [])):
            r = requests.get(ISSUE_URL.format(key=issue["key"]), timeout=120,
                             params={"fields": "created", "expand": "changelog"})
            r.raise_for_status()
            issue["changelog"] = r.json()["changelog"]
            time.sleep(0.2)


def _pages(cache_dir):
    for path in sorted(glob.glob(os.path.join(cache_dir, "page_*.json.gz"))):
        with gzip.open(path, "rt") as f:
            yield json.load(f)


def load_issues(cache_dir):
    """Cached issues, one page in memory at a time."""
    for page in _pages(cache_dir):
        yield from page["issues"]


def snapshot_time(cache_dir):
    """The earliest page download: every page holds at least the events up to then."""
    return min(pd.Timestamp(page["fetched_at"]) for page in _pages(cache_dir))


def _ts(value):
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts.tz_localize(None) if ts is not pd.NaT else pd.NaT


def _changes(issue):
    """Changelog items as (time, field, item), oldest first."""
    out = []
    for h in (issue.get("changelog") or {}).get("histories", []):
        at = _ts(h["created"])
        out += [(at, item["field"], item) for item in h["items"]]
    return sorted(out, key=lambda c: c[0])


def value_at(current, changes, field, when, key="fromString"):
    """A single-valued field at `when`: the "from" side of the first later change, else current."""
    for at, name, item in changes:
        if name == field and at > when:
            return item.get(key)
    return current


def ids_at(current_ids, changes, field, when):
    """A multi-valued field (components, attachments) at `when`, undoing later changes by id."""
    ids = set(current_ids)
    for at, name, item in reversed(changes):
        if name != field or at <= when:
            continue
        if item.get("to"):
            ids.discard(str(item["to"]))
        if item.get("from"):
            ids.add(str(item["from"]))
    return ids


def first_resolved(changes, resolutiondate):
    for at, name, item in changes:
        if name == "resolution" and item.get("to"):
            return at
    # Some migrated issues carry a resolution date but no changelog for it.
    return _ts(resolutiondate) if resolutiondate else pd.NaT


def to_bugs(issues):
    """One row per bug: inputs as filed, timestamped labels. Reporter names are dropped."""
    rows = []
    for issue in issues:
        f = issue["fields"]
        created = _ts(f.get("created"))
        if created is pd.NaT or created < EARLIEST:
            continue
        changes = _changes(issue)
        filed = created + FILED_GRACE
        if value_at("Bug", changes, "issuetype", filed) != "Bug":
            continue  # re-typed into a Bug later, so not a bug report when filed
        current_priority = (f.get("priority") or {}).get("name")
        priority = PRIORITY_MAP.get(
            value_at(current_priority, changes, "priority", created + PRIORITY_HORIZON)
        )
        if priority is None:
            continue
        components = {str(c["id"]): c["name"] for c in f.get("components") or []}
        filed_ids = ids_at(components, changes, "Component", filed)
        for _, name, item in changes:  # names of components removed since filing
            if name == "Component":
                for side in ("from", "to"):
                    if item.get(side):
                        components.setdefault(
                            str(item[side]), item.get(side + "String") or "unknown"
                        )
        filed_components = sorted(components[i] for i in filed_ids)
        attachments = ids_at(
            [str(a["id"]) for a in f.get("attachment") or []], changes, "Attachment", filed
        )
        resolved = first_resolved(changes, f.get("resolutiondate"))
        rows.append(
            dict(
                ticket_id=issue["key"],
                created_at=created,
                subject=value_at(f.get("summary") or "", changes, "summary", filed) or "",
                description=(
                    value_at(f.get("description") or "", changes, "description", filed) or ""
                )[:MAX_DESCRIPTION],
                channel="unknown",
                product=issue["key"].split("-")[0],
                customer_tier="unknown",
                category=filed_components[0] if filed_components else "unknown",
                attachments=len(attachments),
                priority=priority,
                priority_known_at=created + PRIORITY_HORIZON,
                filed_priority=PRIORITY_MAP.get(
                    value_at(current_priority, changes, "priority", filed)
                ),
                first_resolved_at=resolved
                if resolved is not pd.NaT and resolved >= created
                else pd.NaT,
                reporter=(f.get("reporter") or {}).get("name"),
            )
        )
    df = pd.DataFrame(rows).sort_values(["created_at", "ticket_id"]).reset_index(drop=True)
    df["prior_tickets_30d"] = _prior_tickets(df)
    return df.drop(columns=["reporter"])


def _prior_tickets(df):
    """Bugs the same reporter filed in the previous 30 days; expects rows sorted by created_at.

    Counts every bug in the population, whatever its later outcome, so it only uses
    information available when the ticket is filed.
    """
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", default="data/jira_raw")
    ap.add_argument("--out", default="data/jira_bugs.parquet")
    args = ap.parse_args()
    fetch_issues(args.cache)
    snapshot = snapshot_time(args.cache)
    df = to_bugs(load_issues(args.cache))
    df.attrs["snapshot_at"] = snapshot.isoformat()
    df.to_parquet(args.out, index=False)
    print(f"snapshot at {snapshot}")
    print(f"wrote {len(df):,} bugs -> {args.out}")
    print(df["priority"].value_counts().to_string())
    print(f"priority changed within {PRIORITY_HORIZON.days} days of filing: "
          f"{(df['priority'] != df['filed_priority']).mean():.1%}")
    print(f"never resolved: {df['first_resolved_at'].isna().mean():.1%}")


if __name__ == "__main__":
    main()
